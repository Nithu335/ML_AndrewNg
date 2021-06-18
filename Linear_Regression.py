# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:36:13 2021

@author: Nishant.Kumar2
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex1data1.txt", header=None)


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

data_n=data.values
m=len(data_n[:,-1])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

Total_cost = computeCost(X,y,theta)

theta,J_history = gradientDescent(X,y,theta,0.01,1500)

print ("h(x) = {} + {}".format(round(theta[0,0],2),round(theta[1,0],2)))

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1] + theta [0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City")
plt.ylabel("Profit")
plt.title("Profit Vs Population")  

predict = predictions(np.array([1,3.5]),theta)*10000
print(predict)