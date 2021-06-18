# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:40:51 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import matplotlib.image as mpimg

def sigmoid(z):
    value=(1/(1+np.exp(-z)))
    return value

def costFunction(theta,X,y,Lambda):
    m=len(y)
    predictions=sigmoid(np.dot(X,theta))
    error=(-y * np.log(predictions)) -((1-y) * np.log(1-predictions))
    cost = (1/m) * sum(error)
    regcost = cost + (Lambda/(2*m)) * sum(theta[1:]**2)
    
    j_0= (1/m) * np.dot(X.transpose(), (predictions-y))[0]
    j_1= (1/m) * np.dot(X.transpose(), (predictions-y))[1:] + (Lambda/m) * theta[1:]
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regcost, grad

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        cost,grad = costFunction(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    
    return theta,J_history

def onevsAll(X,y,num_labels,Lambda):
    m,n = X.shape[0], X.shape[1]
    initial_theta=np.zeros((n+1,1))
    all_theta=[]
    all_J=[]
    X=np.hstack((np.ones((m,1)),X))
    for i in range(1,num_labels+1):
        theta,J_history=gradientDescent(X,np.where(y==i,1,0),initial_theta,1,300,Lambda)
        all_theta.append(theta)
        all_J.append(J_history)
    
    return np.array(all_theta).reshape(num_labels,n+1),all_J
def predictOneVsAll(all_theta, X):
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    predictions= np.dot(X,all_theta.transpose())
    
    return np.argmax(predictions,axis=1)+1

data=loadmat("ex3data1.mat")
X=data["X"]
y=data["y"]
all_theta, all_J=onevsAll(X,y,10,3)
plt.figure(1)
fig,axis=plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"),cmap="hot")
        axis[i,j].axis("off")
        
pred=predictOneVsAll(all_theta,X)
print("Train Accuracy:",(sum(pred[:,np.newaxis]==y)/5000 *100),"%")