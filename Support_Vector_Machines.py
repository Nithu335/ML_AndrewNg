# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:17:54 2021

@author: Nishant.Kumar2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import loadmat 
from sklearn.svm import SVC
data=loadmat("ex6data1.mat")
X=data["X"]
y=data["y"]
plt.figure(1)
m, n = X.shape[0],X.shape[1]
pos,neg = (y==1).reshape(m,1),(y==0).reshape(m,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o")

classifier1 = SVC(kernel="linear")
classifier1.fit(X,np.ravel(y))

X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier1.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)

classifier2 = SVC(C=100,kernel="linear")
classifier2.fit(X,np.ravel(y))
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier2.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)

data2=loadmat("ex6data2.mat")
X2=data2["X"]
y2=data2["y"]
m,n=X2.shape[0],X2.shape[1]
plt.figure(2)
pos,neg=(y2==1).reshape(m,1),(y2==0).reshape(m,1)
plt.scatter(X2[pos[:,0],0],X2[pos[:,0],1],c="r",marker="+")
plt.scatter(X2[neg[:,0],0],X2[neg[:,0],1],c="b",marker="o")

classifier3 = SVC(kernel="rbf",gamma=30)
classifier3.fit(X2,np.ravel(y2))
X_1,X_2 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier3.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)

classifier4 = SVC(C=100,kernel="rbf",gamma=30)
classifier4.fit(X2,np.ravel(y2))
X_1,X_2 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier4.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)

classifier5 = SVC(C=100,kernel="rbf",gamma=10)
classifier5.fit(X2,np.ravel(y2))
X_1,X_2 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier5.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)

classifier6 = SVC(C=100,kernel="rbf",gamma=60)
classifier6.fit(X2,np.ravel(y2))
X_1,X_2 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier6.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)

data3=loadmat("ex6data3.mat")
print(data3)
X3=data3["X"]
y3=data3["y"]
Xval=data3["Xval"]
yval=data3["yval"]
m,n=X3.shape[0],X3.shape[1]
pos,neg=(y3==1).reshape(m,1),(y3==0).reshape(m,1)
plt.figure(3)
plt.scatter(X3[pos[:,0],0],X3[pos[:,0],1],c="r",marker="+")
plt.scatter(X3[neg[:,0],0],X3[neg[:,0],1],c="g",marker="o")

def dataset3Params(X3,y3,Xval,yval,vals):
    acc=0
    best_C=0
    best_gamma=0
    for i in vals:
        C=i
        for j in vals:
            gamma=1/j
            classifier = SVC(C=C,gamma=gamma)
            classifier.fit(X3,y3.ravel())
            prediction=classifier.predict(Xval)
            score=classifier.score(Xval,yval)
            if score>acc:
                acc=score
                best_C=C
                best_gamma=gamma
    
    return best_C,best_gamma

vals=[0.01,0.03,0.1,0.3,1,3,10,30]
C,gamma=dataset3Params(X3,y3,Xval,yval,vals)
classifier7 = SVC(C=C,kernel="rbf",gamma=gamma)
classifier7.fit(X3,np.ravel(y3))
X_1,X_2 = np.meshgrid(np.linspace(X3[:,0].min(),X3[:,0].max(),num=100),np.linspace(X3[:,1].min(),X3[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier7.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
