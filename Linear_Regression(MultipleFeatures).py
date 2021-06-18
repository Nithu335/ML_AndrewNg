# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:12:19 2021

@author: Nishant.Kumar2
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex1data2.txt", header=None)
fig, axes= plt.subplots(figsize=(12,4),nrows=1,ncols=2)

axes[0].scatter(data[0],data[2],color="b")
axes[0].set_xlabel("Size")
axes[0].set_ylabel("Prices")
axes[0].set_title("House prices against size of the house")
axes[1].scatter(data[1],data[2],color="r")
axes[1].set_xlabel("Number of bedroom")
axes[1].set_ylabel("Prices")
axes[1].set_title("House prices against no of bedrooms")
#plt.tight_layout()
def featureNormalisation(X):
    
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm = (X-mean)/std
    
    return X_norm, mean, std

data_n=data.values
m=len(data_n[:,-1])
X=data_n[:,0:2].reshape(m,2)
X, mean , std = featureNormalisation(X)
X= np.append(np.ones((m,1)),X,axis=1)
y= data_n[:,-1].reshape(m,1)
theta=np.zeros((3,1))

def computeCost(X,y,theta):
    m=len(y)
    predictions=X.dot(theta)
    square_err= (predictions - y)**2
    cost=(1/(2*m)) * np.sum(square_err)
    return cost

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent= alpha * (1/m) * error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    
    return theta, J_history

def predictions(x,theta):
    predictions= np.dot(theta.transpose(),x)
    
    return predictions[0]

Total_cost = computeCost(X,y,theta)

theta,J_history = gradientDescent(X,y,theta,0.01,400)

print("The optimised function is h(x) = {} + {}x_1 + {}x_2".format(round(theta[0,0]),round(theta[1,0]),round(theta[2,0])))

x_sample =(np.array([1650,3])-mean)/std
x_sample=np.append(np.ones(1),x_sample)
predict3=predictions(x_sample,theta)
print("For size of house = 1650, Number of bedroom = 3, we predict a house value of $"+str(round(predict3,0)))
