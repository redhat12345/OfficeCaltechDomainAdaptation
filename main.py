#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random

import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances


###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

featuresToUse = "surf"  # surf, CaffeNet4096, GoogleNet1024
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
    elif algo == "SA":
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
    elif algo == "OT":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        import ot  # https://github.com/rflamary/POT
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=2, reg_cl=1, norm="median")
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)
        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx

    return sourceAdapted, targetAdapted


def getAccuracy(trainData, trainLabels, testData, testLabels):
    # ------------ Accuracy evaluation by performing a 1NearestNeighbor
    dist = euclidean_distances(trainData, testData, squared=True)
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
