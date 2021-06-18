# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:51:43 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def mapFeature(x1,x2,degree):
    out=np.ones(len(x1)).reshape(len(x1),1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2 **j).reshape(len(x1),1)
            out=np.hstack((out,terms))
    return out 

def sigmoid(z):
    value = 1/(1 + np.exp(-z))
    return value

def costFunction(X,y,theta,Lambda):
    m=len(y)
    y=y[:,np.newaxis]
    predictions=sigmoid(np.dot(X,theta))
    error = (- y * np.log(predictions)) -((1-y) * np.log(1-predictions))
    cost = (1/m) * sum(error)
    regcost = cost + (Lambda/(2*m)) * sum(theta[1:]**2)
    
    j_0 = (1/m) * np.dot(X.transpose(),(predictions-y))[0]
    j_1 = (1/m) * np.dot(X.transpose(),(predictions-y))[1:] + (Lambda/m) * theta[1:]
    grad = np.vstack((j_0[:,np.newaxis],j_1))
    return regcost,grad

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        cost,grad=costFunction(X,y,theta,Lambda)
        theta = theta - (alpha*grad)
        J_history.append(cost)
    return theta,J_history

def classifierPredict(theta,X):
    predictions=X.dot(theta)
    return predictions>0

def predictions(x,theta):
    predictions= np.dot(theta.transpose(),x)
    
    return predictions[0]
  
data=pd.read_csv("ex2data2.txt", header=None)

data_n=data.values
X=data_n[:,0:2]
y=data_n[:,-1]

plt.figure(1)
pos,neg=(y==1).reshape(118,1),(y==0).reshape(118,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o")
plt.xlabel("Test 1")
plt.ylabel("Test2")
plt.legend(["Accepted","Rejected"],loc=0)



X = mapFeature(X[:,0], X[:,1],6)
initial_theta = np.zeros((X.shape[1],1))

Lambda=1

cost,grad = costFunction(X,y,initial_theta,Lambda)

print("Cost at initial theta:{}".format(cost))

theta, J_history = gradientDescent(X,y,initial_theta,1,800,0.2)

print("The optimised theta is {}".format(theta))

p=classifierPredict(theta,X)
print("Train Accuracy:",(sum(p==y[:,np.newaxis])/len(y) *100),"%")

