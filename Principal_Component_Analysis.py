# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:30:51 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd
import matplotlib.image as mpimg
data=loadmat("ex7data1.mat")
X=data["X"]
plt.figure(1)
plt.scatter(X[:,0],X[:,1],marker="o",facecolors="none",edgecolors="b")

def featureNormalisation(X):
    
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm = (X-mu)/sigma
    
    return X_norm,mu,sigma

def pca(X):
    m,n=X.shape[0],X.shape[1]
    
    sigma=(1/m) * X.T @ X
    
    U,S,V =svd(sigma)
    
    return U,S,V
def projectData(X,U,K):
    m=X.shape[0]
    U_reduced=U[:,:K]
    Z=np.zeros((m,K))
    for i in range(m):
        for j in range(K):
            Z[i,j]=X[i,:] @ U_reduced[:,j]
            
    return Z

def recoverData(Z,U,K):
    m,n=Z.shape[0],U.shape[1]
    X_rec=np.zeros((m,n))
    U_reduced=U[:,:K]
    for i in range(m):
        X_rec[i,:] = Z[i,:] @ U_reduced.T
        
    return X_rec
            
X_norm,mu,std=featureNormalisation(X)
U,S=pca(X_norm)[:2]

K=1
Z=projectData(X_norm,U,K)
print("Projection of the first example:",Z[0][0])

X_rec=recoverData(Z,U,K)
print("Approximation of the first example:",X_rec[0,:])

mat=loadmat("ex7faces.mat")
X2=mat["X"]
plt.figure(2)
fig, ax = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax[int(i/10),j].imshow(X2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax[int(i/10),j].axis("off")

X_norm2=featureNormalisation(X2)[0]

U2=pca(X_norm2)[0]

U_reduced = U2[:,:36].T

plt.figure(3)
fig2,ax2=plt.subplots(6,6,figsize=(8,8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].imshow(U_reduced[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/6),j].axis("off")
        
K2=100
Z2=projectData(X_norm2,U2,K2)
print("The project data Z has size:",Z2.shape)

X_rec2=recoverData(Z2,U2,K2)
plt.figure(4)
fig3, ax3 = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax3[int(i/10),j].imshow(X_rec2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax3[int(i/10),j].axis("off")

