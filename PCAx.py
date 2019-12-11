# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, cov, cumsum, dot, linalg, size, argsort, real
from pylab import imread, subplot, imshow, title, gray, figure, show, NullLocator, plot, axis
# from pylab import plot, axis


def prin_Comp(Img, Npc):
    adjImg = np.transpose(Img-mean((Img), axis=0)) 
    [EigVal, EigVec] = linalg.eig(cov(adjImg))  # eigenvalues/vectors of covariance
    idx = argsort(EigVal)  # sort eigenvalues
    idx = idx[::-1] 
    EigVec = EigVec[:, idx]  # sort eigenvectors
    EigVal = EigVal[idx]
    numEigVec = size(EigVec, axis=1)
    if Npc >= 0 and Npc < numEigVec:
        EigVec = EigVec[:, range(Npc)]  # subset of PAs
    proj = dot(EigVec.T, adjImg)  # projection into reduced space

    return EigVec, proj, EigVal

# Read image and iteratively reduce number of Principal Components
Img = imread('F18.jpg')   # load image
Img = mean(Img, 2) 
totalpc = size(Img, axis=1)  # total number of all the principal components
for Npc in range(0, totalpc+10, 10):  # 0, 10, 20, up to full_pc
    EigVec, proj, EigVal = prin_Comp(Img, Npc)
    recImg = np.transpose(dot(EigVec, proj)) + mean(Img, axis=0)

    maxpc = 100
    if Npc <= maxpc:
        figure()
        imshow(real(recImg))
        title('Principal Components = ' + str(Npc))
        gray()
        plt.savefig('Principal Components = ' + str(Npc) + '.png')


figure()
imshow(real(Img))
title('All ' + str(totalpc)+' Principal Components')
gray()
plt.savefig('Full Resolution ' + str(totalpc) + ' Principal Components.png')


perExp = np.divide(cumsum(EigVal), sum(EigVal))
figure()
plt.plot(range(len(perExp)), perExp, 'g')
plt.axis([0, totalpc, 0, 1.1])
plt.xlabel('Number of Prinicpal Components')
plt.ylabel('Variance Explained')
plt.savefig('Variance Explained 2.png')
plt.show()
plt.close
