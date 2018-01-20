# OfficeCaltechDomainAdaptation

## What is this ?
This dataset is part of the Computer Vision problematic consisting in making machines learn to detect the presence of an object in an image. Here, we want to learn a classification model that takes as input an image and return the category of the object it contains.

The Office Caltech dataset contains four different domains: amazon, caltech10, dslr and webcam. These domains contain respectively 958, 1123, 295 and 157 images. Each image contains only one object among a list of 10 objects: backpack, bike, calculator, headphones, keyboard, laptop, monitor, mouse, mug and projector.

With this benchmark dataset in Domain Adaptation, we repeatedly take one of the four domains as Source domain S and one of the three remaining as target T. The aim is then to learn to classify images with the data from S to correctly classify the images in T.

## What is available in this repository ?
In addition to the images, we also give features that were extracted from the images to describe them. We give different sets of features that describe all the images in the corresponding folder.

We propose some code in python3 to show how to evaluate the benchmark. What is usually evaluated with this benchmark are Domain Adaptation algorithms. We provide code for a few of them.

## Dependencies
Python3 and some python3 libraries:
 - numpy
 - scipy
 - sklearn

## Example of execution

Program launched by executing the main.py script with python:
```
python3 main.py
```

For each adaptation problem among the 12 possible, each adaptation algorithm chosen at the beginning of the file is applied. Then are reported the mean accuracy and standard deviation. Results (using the default surf [1] features):
```
Feature used:  surf
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT
A->C ..........
     22.3  2.4  NA     0.12s
     34.1  1.4  SA     0.95s
     33.7  1.6  TCA   18.60s
     29.3  1.3  OT     1.65s
A->D ..........
     23.6  2.4  NA     0.02s
     31.8  3.6  SA     0.38s
     29.4  3.4  TCA    0.81s
     40.2  3.2  OT     0.37s
A->W ..........
     23.2  1.8  NA     0.03s
     29.4  3.2  SA     0.46s
     30.6  3.2  TCA    1.67s
     36.4  3.8  OT     0.47s
C->A ..........
     21.4  2.5  NA     0.10s
     35.5  2.3  SA     0.84s
     36.4  2.0  TCA   12.71s
     37.1  2.3  OT     1.41s
C->D ..........
     22.5  2.2  NA     0.02s
     35.2  4.2  SA     0.38s
     32.9  5.2  TCA    0.82s
     44.1  4.1  OT     0.40s
C->W ..........
     21.5  3.5  NA     0.03s
     30.3  4.7  SA     0.45s
     30.6  2.4  TCA    1.67s
     34.4  3.0  OT     0.49s
D->A ..........
     27.3  1.3  NA     0.07s
     32.9  0.9  SA     0.69s
     33.5  1.0  TCA    9.64s
     29.1  1.9  OT     0.76s
D->C ..........
     24.3  1.3  NA     0.08s
     30.9  1.0  SA     0.75s
     31.9  1.0  TCA   14.08s
     29.9  1.0  OT     0.87s
D->W ..........
     52.4  2.3  NA     0.02s
     79.6  2.2  SA     0.34s
     75.1  2.3  TCA    0.96s
     68.2  1.6  OT     0.34s
W->A ..........
     22.9  0.8  NA     0.10s
     32.6  2.9  SA     0.84s
     31.4  2.4  TCA   12.76s
     38.1  1.1  OT     1.42s
W->C ..........
     19.0  0.8  NA     0.11s
     29.4  1.3  SA     0.92s
     29.3  0.7  TCA   18.53s
     34.0  0.7  OT     1.63s
W->D ..........
     52.1  2.7  NA     0.02s
     82.7  2.5  SA     0.39s
     79.9  2.3  TCA    0.82s
     71.3  1.6  OT     0.35s

Mean results and total time
     27.7  2.0  NA     0.73s
     40.4  2.5  SA     7.40s
     39.6  2.3  TCA   93.05s
     41.0  2.1  OT    10.16s
```

By modifying the feature used in the script with CaffeNet [2] features:
```
Feature used:  CaffeNet4096
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT
A->C ..........
     71.2  2.8  NA     0.35s
     78.7  0.9  SA     6.54s
     79.8  0.8  TCA   19.49s
     82.4  0.9  OT     4.14s
A->D ..........
     76.8  4.7  NA     0.08s
     82.1  3.0  SA     3.17s
     88.4  1.8  TCA    0.90s
     93.9  1.4  OT     1.12s
A->W ..........
     68.0  4.0  NA     0.11s
     77.7  2.8  SA     3.57s
     82.1  2.8  TCA    1.90s
     91.5  0.9  OT     1.50s
C->A ..........
     83.4  1.8  NA     0.30s
     86.3  2.2  SA     5.99s
     87.7  1.9  TCA   13.24s
     88.8  1.2  OT     3.51s
C->D ..........
     77.2  3.3  NA     0.08s
     80.3  1.6  SA     3.17s
     84.1  2.9  TCA    0.89s
     93.1  1.0  OT     1.12s
C->W ..........
     73.9  5.2  NA     0.11s
     79.9  1.8  SA     3.49s
     83.0  3.1  TCA    1.87s
     90.4  1.9  OT     1.45s
D->A ..........
     70.0  4.6  NA     0.20s
     83.3  1.6  SA     5.80s
     87.5  1.2  TCA   10.02s
     86.4  1.3  OT     2.57s
D->C ..........
     68.2  2.2  NA     0.24s
     75.8  1.9  SA     6.36s
     78.1  1.5  TCA   14.82s
     80.5  2.6  OT     3.02s
D->W ..........
     91.5  1.7  NA     0.08s
     96.8  1.1  SA     3.50s
     97.1  1.3  TCA    1.10s
     96.5  0.6  OT     1.06s
W->A ..........
     68.7  2.2  NA     0.29s
     82.2  1.2  SA     5.89s
     87.1  1.1  TCA   13.30s
     86.7  1.7  OT     3.57s
W->C ..........
     61.1  1.7  NA     0.34s
     73.7  1.2  SA     6.43s
     76.6  0.9  TCA   18.56s
     76.4  1.0  OT     4.05s
W->D ..........
     95.5  1.6  NA     0.08s
     99.7  0.3  SA     3.15s
     99.6  0.4  TCA    0.90s
     97.5  0.8  OT     1.09s

Mean results and total time
     75.4  3.0  NA     2.24s
     83.0  1.6  SA    57.05s
     85.9  1.6  TCA   97.00s
     88.7  1.3  OT    28.20s
```

and with GoogleNet [3] features:
```
Feature used:  GoogleNet1024
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT
A->C ..........
     85.4  0.7  NA     0.13s
     86.6  1.1  SA     1.41s
     87.9  1.5  TCA   18.28s
     89.5  0.6  OT     1.88s
A->D ..........
     89.0  2.3  NA     0.03s
     89.1  2.7  SA     0.50s
     90.4  2.6  TCA    0.83s
     94.1  0.6  OT     0.49s
A->W ..........
     82.3  2.7  NA     0.04s
     83.4  1.6  SA     0.59s
     86.4  1.8  TCA    1.69s
     95.1  1.1  OT     0.62s
C->A ..........
     89.9  1.1  NA     0.10s
     90.9  0.8  SA     1.26s
     92.6  0.7  TCA   12.41s
     93.8  0.4  OT     1.62s
C->D ..........
     87.2  2.9  NA     0.03s
     88.7  1.3  SA     0.50s
     91.5  1.5  TCA    0.82s
     94.1  0.7  OT     0.47s
C->W ..........
     83.9  2.5  NA     0.04s
     88.0  2.5  SA     0.59s
     91.7  2.2  TCA    1.68s
     96.9  0.4  OT     0.60s
D->A ..........
     82.3  2.0  NA     0.07s
     87.3  1.7  SA     1.07s
     88.9  1.3  TCA    9.40s
     91.7  0.6  OT     1.03s
D->C ..........
     77.8  1.9  NA     0.09s
     83.7  1.7  SA     1.22s
     84.3  1.8  TCA   13.92s
     89.6  0.8  OT     1.19s
D->W ..........
     96.9  1.4  NA     0.03s
     97.5  1.5  SA     0.45s
     97.7  1.3  TCA    0.95s
     97.7  0.5  OT     0.42s
W->A ..........
     86.4  1.2  NA     0.11s
     89.9  0.5  SA     1.26s
     92.3  0.7  TCA   12.60s
     93.0  0.3  OT     1.83s
W->C ..........
     79.1  1.5  NA     0.12s
     83.7  0.9  SA     1.41s
     87.5  0.5  TCA   18.04s
     90.5  1.0  OT     1.83s
W->D ..........
     99.2  0.7  NA     0.03s
     99.5  0.5  SA     0.52s
     99.4  0.3  TCA    0.82s
     97.8  0.8  OT     0.49s

Mean results and total time
     86.6  1.8  NA     0.81s
     89.0  1.4  SA    10.78s
     90.9  1.3  TCA   91.45s
     93.6  0.6  OT    12.47s
```
[1] Gong, B., Grauman, K., & Sha, F. (2014). Learning kernels for unsupervised domain adaptation with applications to visual object recognition. International Journal of Computer Vision, 109(1-2), 3-27.

[2] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., ... & Darrell, T. (2014, November). Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 675-678). ACM.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
