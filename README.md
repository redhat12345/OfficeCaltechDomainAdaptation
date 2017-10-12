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
Adaptation algorithms used:   NA  SA
A->C ..........   1.43s
     21.1  1.2 NA
     33.0  1.6 SA
A->D ..........   0.50s
     22.3  3.5 NA
     32.2  2.7 SA
A->W ..........   0.64s
     25.4  2.2 NA
     31.7  3.0 SA
C->A ..........   1.24s
     20.4  1.9 NA
     35.3  2.5 SA
C->D ..........   0.50s
     22.5  3.2 NA
     36.9  3.0 SA
C->W ..........   0.62s
     20.4  3.9 NA
     28.1  3.7 SA
D->A ..........   0.99s
     27.6  2.3 NA
     32.5  1.2 SA
D->C ..........   1.16s
     24.8  1.9 NA
     30.0  1.3 SA
D->W ..........   0.47s
     52.7  1.5 NA
     79.5  2.0 SA
W->A ..........   1.33s
     23.1  1.1 NA
     32.2  2.5 SA
W->C ..........   1.43s
     19.3  0.7 NA
     28.4  0.8 SA
W->D ..........   0.51s
     50.6  4.3 NA
     82.9  1.7 SA

Mean results:
     27.5  2.3 NA
     40.2  2.2 SA
```

By modifying the feature used in the script with CaffeNet [2] features:
```
Feature used:  CaffeNet4096
Number of iterations:  10
Adaptation algorithms used:   NA  SA
A->C ..........   7.97s
     72.0  2.5 NA
     78.8  1.0 SA
A->D ..........   3.51s
     77.4  4.0 NA
     82.7  2.6 SA
A->W ..........   3.94s
     66.6  4.3 NA
     79.0  2.0 SA
C->A ..........   7.32s
     81.2  2.2 NA
     85.2  1.2 SA
C->D ..........   3.64s
     74.6  5.4 NA
     79.3  1.2 SA
C->W ..........   4.07s
     71.5  3.8 NA
     76.8  2.4 SA
D->A ..........   6.61s
     70.3  2.5 NA
     83.4  1.4 SA
D->C ..........   7.28s
     67.2  2.0 NA
     75.3  1.1 SA
D->W ..........   3.66s
     92.0  1.9 NA
     97.0  1.3 SA
W->A ..........   6.87s
     68.9  2.9 NA
     82.4  1.2 SA
W->C ..........   7.64s
     61.0  0.9 NA
     73.1  1.1 SA
W->D ..........   3.30s
     95.4  1.2 NA
     99.7  0.3 SA

Mean results:
     74.8  2.8 NA
     82.7  1.4 SA
```

and with GoogleNet [3] features:
```
Feature used:  GoogleNet1024
Number of iterations:  10
Adaptation algorithms used:   NA  SA
A->C ..........   1.93s
     84.5  1.0 NA
     85.8  1.1 SA
A->D ..........   0.64s
     88.9  1.8 NA
     87.4  2.0 SA
A->W ..........   0.81s
     83.4  2.2 NA
     83.7  2.6 SA
C->A ..........   1.76s
     90.8  1.3 NA
     91.4  0.4 SA
C->D ..........   0.65s
     87.8  1.7 NA
     88.9  2.0 SA
C->W ..........   0.90s
     85.3  2.5 NA
     88.4  3.1 SA
D->A ..........   1.59s
     83.1  1.7 NA
     88.5  1.5 SA
D->C ..........   1.69s
     76.8  2.7 NA
     83.5  1.7 SA
D->W ..........   0.57s
     97.4  1.2 NA
     97.8  1.3 SA
W->A ..........   1.75s
     86.8  0.9 NA
     89.7  0.7 SA
W->C ..........   1.88s
     79.4  0.7 NA
     84.0  0.4 SA
W->D ..........   0.63s
     99.4  0.4 NA
     99.4  0.4 SA

Mean results:
     87.0  1.5 NA
     89.0  1.4 SA
```
[1] Gong, B., Grauman, K., & Sha, F. (2014). Learning kernels for unsupervised domain adaptation with applications to visual object recognition. International Journal of Computer Vision, 109(1-2), 3-27.

[2] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., ... & Darrell, T. (2014, November). Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 675-678). ACM.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
