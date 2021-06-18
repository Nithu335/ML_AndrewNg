# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:50:47 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data1=loadmat("ex7data2.mat")

X=data1["X"]

def findClosestCentroid(X,centroids):
    K=centroids.shape[0]
    idx=np.zeros((X.shape[0],1))
    temp=np.zeros((centroids.shape[0],1))
    
    for i in range (X.shape[0]):
        for j in range(K):
            dist=X[i,:] - centroids[j,:]
            length = np.sum(dist**2)
            temp[j]=length
        idx[i]=np.argmin(temp)+1
    return idx

def computeCentroid(X,idx,K):
    m,n=X.shape[0],X.shape[1]
    centroids=np.zeros((K,n))
    count=np.zeros((K,1))
    
    for i in range(m):
        index = int((idx[i]-1))
        centroids[index,:]+=X[i,:]
        count[index]+=1
    return centroids/count

def plotKmeans(X,centroids,idx,K,num_iters):
    m,n=X.shape[0],X.shape[1]
    fig,ax=plt.subplots(nrows=num_iters,ncols=1,figsize=(6,36))
    
    for i in range(num_iters):
        color="rgb"
        for k in range(1,K+1):
            grp=(idx==k).reshape(m,1)
            ax[i].scatter(X[grp[:,0],0],X[grp[:,0],1],c=color[k-1],s=15)
        ax[i].scatter(centroids[:,0],centroids[:,1],s=120,marker="x",c="black",linewidth=3)
        title="Iteration Number" + str(i)
        ax[i].set_title(title)
        
        centroids = computeCentroid(X,idx,K)
        
        idx=findClosestCentroid(X,centroids)
    
    plt.tight_layout()
            
def kmeansInitialCentroids(X,K):
    m,n=X.shape[0],X.shape[1]
    centroids=np.zeros((K,n))
    for i in range(K):
        centroids[i] = X[np.random.randint(0,m),:]
        
    return centroids
        
K=3
initial_centroids= np.array([[3,3],[6,2],[8,5]])
idx=findClosestCentroid(X,initial_centroids)
print("Closest centroids for the first 3 examples:\n",idx[0:3])

centroids = computeCentroid(X,idx,K)
print("Centroids after intial finding of closest centroids:",centroids)

m,n=X.shape[0],X.shape[1]
centroids = kmeansInitialCentroids(X,K)
idx=findClosestCentroid(X,centroids)
plotKmeans(X,centroids,idx,K,10)