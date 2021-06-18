# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:33:11 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat("ex5data1.mat")
X=data["X"]
y=data["y"]
Xtest=data["Xtest"]
ytest=data["ytest"]
Xval=data["Xval"]
yval=data["yval"]
m=len(y)
m1=len(yval)
m2=len(ytest)
X=np.append(np.ones((m,1)),X.reshape(m,1),axis=1)
Xval=np.append(np.ones((m1,1)),Xval.reshape(m1,1),axis=1)
Xtest=np.append(np.ones((m2,1)),Xtest.reshape(m2,1),axis=1)
def costFunction(X,y,theta,Lambda):
    m=len(y)
    prediction = X.dot(theta)
    error =(prediction - y)**2
    cost = (1/(2*m)) * sum(error) 
    reg_cost = cost + (Lambda/(2*m)) * sum(theta[1:]**2)
    j_0 = (1/m) * np.dot(X.transpose(),(prediction-y))[0]
    j_1 = (1/m) * np.dot(X.transpose(),(prediction-y))[1:] + (Lambda/m) * theta[1:]
    grad = np.vstack((j_0[:,np.newaxis],j_1))
    return reg_cost, grad

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        reg_cost,grad=costFunction(X,y,theta,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(reg_cost)
    
    return theta, J_history
def predictions(x,theta):
    predictions= np.dot(theta.transpose(),x)
    
    return predictions[0]


theta=np.zeros((2,1))
theta, J_history = gradientDescent(X,y,theta,0.001,7000,0.0001)
print("The optimised theta is {}".format(theta))

costFunction_train, grad_val = costFunction(X,y, theta,0.0001)

print("The validation error is {}".format(costFunction_train))
costFunction_val, grad_val = costFunction(Xval,yval,theta,0.0001)

print("The validation error is {}".format(costFunction_val))

costFunction_test, grad_val = costFunction(Xtest,ytest,theta,0.0001)

print("The test error is {}".format(costFunction_val))


    