# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:13:26 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex2data1.txt",header=None)

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def costFunction(theta,X,y):
    m=len(y)
    predictions=sigmoid(np.dot(X,theta))
    error=(-y * np.log(predictions))-((1-y)*np.log(1-predictions))
    cost = (1/m) * sum(error)
    grad = (1/m) * np.dot(X.transpose(),(predictions-y))
    return cost, grad

def featureNormalization(X):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm=(X-mean)/std
    return X_norm,mean,std

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        cost,grad= costFunction(theta,X,y)
        theta = theta - (alpha*grad)
        J_history.append(cost)
    return theta,J_history

def predictions(x,theta):
    predictions= np.dot(theta.transpose(),x)
    
    return predictions[0]

data_n=data.values
X=data_n[:,0:2]
y=data_n[:,-1]
plt.figure(1)
pos, neg = (y==1).reshape(100,1), (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c='b',marker="o")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)
m,n=X.shape[0],X.shape[1]
X,mean,std = featureNormalization(X)
X=np.append(np.ones((m,1)),X,axis=1)
y=y.reshape(m,1)
initial_theta=np.zeros((n+1,1))
cost,grad=costFunction(initial_theta,X,y)
print("Cost of initial theta is :{}".format(cost))
print("Gradient at initial theta is :{}".format(grad))

theta,J_history=gradientDescent(X,y,initial_theta,1,400)
plt.figure(2)
plt.plot(J_history)

print("The optimised theta is :{}".format(theta))

plt.figure(3)
pos, neg = (y==1).reshape(100,1), (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="r",marker="+")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c='b',marker="o")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)
x_values=np.array([np.min(X[:,1]),np.max(X[:,1])])
y_values = -(theta[0] + theta[1] *x_values)/theta[2]
plt.plot(x_values,y_values)

x_test=np.array([45,85])
x_test=(x_test-mean)/std
x_test = np.append(np.ones(1),x_test)
prob=sigmoid(x_test.dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of {}".format(prob))