# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:38:22 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat("bird_small.mat")

A = data["A"]

X = (A/255).reshape(128*128,3)

def runKmeans(X,initial_centroids,num_iters,K):
    
    idx=findClosestCentroid(X,initial_centroids)
    for i in range(num_iters):
        centroids = computeCentroids(X,idx,K)
        idx = findClosestCentroid(X,initial_centroids)
    
    return centroids,idx

def kMeansInitialCentroid(X,K):
    m,n=X.shape[0],X.shape[1]
    centroid=np.zeros((K,n))
    for i in range(K):
        centroid[i] = X[np.random.randint(0,m),:]
    
    return centroid

def computeCentroids(X,idx,K):
    m,n=X.shape[0],X.shape[1]
    centroids=np.zeros((K,n))
    count=np.zeros((K,1))
    
    for i in range(m):
        index = int((idx[i]-1))
        centroids[index,:]+=X[i,:]
        count[index]+=1
    return centroids/count

def findClosestCentroid(X,centroids):
    K=centroids.shape[0]
    idx=np.zeros((X.shape[0],1))
    temp=np.zeros((centroids.shape[0],1))
    for i in range(X.shape[0]):
        for j in range(K):
            dist=X[i,:]-centroids[j,:]
            length=np.sum(dist**2)
            temp[j]=length
        idx[i]=np.argmin(temp)+1

    return idx
    
K=16
num_iters=10
initial_centroids=kMeansInitialCentroid(X,K)
centroids,idx=runKmeans(X,initial_centroids,num_iters,K)

m,n=X.shape[0],X.shape[1]
X_recovered=X.copy()
for i in range (1,K+1):
    X_recovered[(idx==i).ravel(),:]=centroids[i-1]

X_recovered=X_recovered.reshape(128,128,3)

import matplotlib.image as mpimg
fig,ax=plt.subplots(1, 2)
ax[0].imshow(X.reshape(128,128,3))
ax[1].imshow(X_recovered)
