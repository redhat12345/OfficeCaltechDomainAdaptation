#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random

import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing


###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

featuresToUse = "GoogleNet1024"  # surf, CaffeNet4096, GoogleNet1024
numberIteration = 10
adaptationAlgoUsed = ["NA", "SA"]
# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def generateSubset(X, Y, nPerClass):
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
    return (X[idx, :], Y[idx])


def adaptData(algo, Sx, Sy, Tx, Ty):
    if algo == "NA":  # No Adaptation
        sourceAdapted = Sx
        targetAdapted = Tx
    if algo == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.
        from sklearn.decomposition import PCA
        d = 80  # subspace dimension
        pcaS = PCA(d).fit(Sx)
        pcaT = PCA(d).fit(Tx)
        XS = np.transpose(pcaS.components_)[:, :d]  # source subspace matrix
        XT = np.transpose(pcaT.components_)[:, :d]  # target subspace matrix
        Xa = XS.dot(np.transpose(XS)).dot(XT)  # align source subspace
        sourceAdapted = Sx.dot(Xa)  # project source in aligned subspace
        targetAdapted = Tx.dot(XT)  # project target in target subspace

    return sourceAdapted, targetAdapted


def pairwiseEuclidean(a, b, squared=False):
    """
    Compute the pairwise euclidean distance between matrices a and b.


    Parameters
    ----------
    a : np.ndarray (n, f)
        first matrix
    b : np.ndarray (m, f)
        second matrix
    squared : boolean, optional (default False)
        if True, return squared euclidean distance matrix


    Returns
    -------
    c : (n x m) np.ndarray
        pairwise euclidean distance distance matrix
    """
    # a is shape (n, f) and b shape (m, f). Return matrix c of shape (n, m).
    # First compute in c the squared euclidean distance. And return its
    # square root. At each cell [i,j] of c, we want to have
    # sum{k in range(f)} ( (a[i,k] - b[j,k])^2 ). We know that
    # (a-b)^2 = a^2 -2ab +b^2. Thus we want to have in each cell of c:
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2).

    # Multiply a by b transpose to obtain in each cell [i,j] of c the
    # value sum{k in range(f)} ( a[i,k]b[j,k] )
    c = a.dot(b.T)
    # multiply by -2 to have sum{k in range(f)} ( -2a[i,k]b[j,k] )
    np.multiply(c, -2, out=c)

    # Compute the vectors of the sum of squared elements.
    a = np.power(a, 2).sum(axis=1)
    b = np.power(b, 2).sum(axis=1)

    # Add the vectors in each columns (respectivly rows) of c.
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] )
    c += a.reshape(-1, 1)
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2)
    c += b

    if not squared:
        np.sqrt(c, out=c)

    return c


def getAccuracy(trainData, trainLabels, testData, testLabels):
    # ------------ Accuracy evaluation by performing a 1NearestNeighbor
    dist = pairwiseEuclidean(trainData, testData, squared=True)
    minIDX = np.argmin(dist, axis=0)
    prediction = trainLabels[minIDX]
    accuracy = 100 * float(sum(prediction == testLabels)) / len(testData)
    return accuracy


# ---------------------------- DATA Loading Part ------------------------------
domainNames = ['amazon', 'caltech10', 'dslr', 'webcam']
tests = []
data = {}

for sourceDomain in domainNames:
    possible_data = loadmat(os.path.join(".", "features", featuresToUse,
                                         sourceDomain + '.mat'))
    if featuresToUse == "surf":
        # Normalize the surf histograms
        feat = (possible_data['fts'].astype(float) /
                np.tile(np.sum(possible_data['fts'], 1),
                        (np.shape(possible_data['fts'])[1], 1)).T)
    else:
        feat = possible_data['fts'].astype(float)

    # Z-score
    feat = preprocessing.scale(feat)

    labels = possible_data['labels'].ravel()
    data[sourceDomain] = [feat, labels]
    for targetDomain in domainNames:
        if sourceDomain != targetDomain:
            perClassSource = 20
            if sourceDomain == 'dslr':
                perClassSource = 8
            tests.append([sourceDomain, targetDomain, perClassSource])

meansAcc = {}
stdsAcc = {}
print("Feature used: ", featuresToUse)
print("Number of iterations: ", numberIteration)
print("Adaptation algorithms used: ", end="")
for name in adaptationAlgoUsed:
    meansAcc[name] = []
    stdsAcc[name] = []
    print(" ", name, end="")
print("")

# -------------------- Main testing loop --------------------------------------
for test in tests:
    startTime = time.time()
    Sname = test[0]
    Tname = test[1]
    perClassSource = test[2]
    print(Sname.upper()[:1] + '->' + Tname.upper()[:1], end=" ")

    # --------------------II. prepare data-------------------------------------
    Sx = data[Sname][0]
    Sy = data[Sname][1]
    Tx = data[Tname][0]
    Ty = data[Tname][1]

    # --------------------III. run experiments---------------------------------
    results = {}
    for name in adaptationAlgoUsed:
        results[name] = []
    for iteration in range(numberIteration):
        (subSx, subSy) = generateSubset(Sx, Sy, perClassSource)
        for name in adaptationAlgoUsed:
            # Apply domain adaptation algorithm
            subSa, Ta = adaptData(name, subSx, subSy, Tx, Ty)
            # Compute the accuracy classification
            results[name].append(getAccuracy(subSa, subSy, Ta, Ty))
        print(".", end="")

    currentTime = time.time()
    print(" {:6.2f}".format(currentTime - startTime) + "s")

    for name in adaptationAlgoUsed:
        meanAcc = np.mean(results[name])
        stdAcc = np.std(results[name])
        meansAcc[name].append(meanAcc)
        stdsAcc[name].append(stdAcc)
        print("     {:4.1f}".format(meanAcc), " {:3.1f}".format(stdAcc), name)

print("")
print("Mean results:")
for name in adaptationAlgoUsed:
    meanMean = np.mean(meansAcc[name])
    meanStd = np.mean(stdsAcc[name])
    print("     {:4.1f}".format(meanMean), " {:3.1f}".format(meanStd), name)
