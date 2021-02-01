#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:47:55 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from assignment import *

#%%################ test complexe #################

#data generation 

n = 10000
m = 8000

Y1 = np.random.randn(2,3*(n//8))
Y1 /= np.linalg.norm(Y1, axis=0, ord=np.inf)
Y1 = 0.5*Y1.T

r = 1
theta = 2*np.pi*np.random.rand(1,3*(n//8))
Y2 = np.vstack((np.cos(theta)*r,np.sin(theta)*r)).T

Y3 = np.random.uniform(-1/np.sqrt(2),1/np.sqrt(2),size = (1,3*(n//16)))
Y3 = np.vstack((Y3,Y3)).T

Y4 = np.random.uniform(-0.5,0,size = (1,n-3*(n//16)-6*(n//8)))
Y4 = np.vstack((-Y4,Y4)).T


Y = np.concatenate((Y1,Y2,Y3,Y4),axis=0)
np.random.shuffle(Y)

X = Y[:m].copy().dot(np.array([[np.cos(np.pi/3),np.sin(np.pi/3)],[-np.sin(np.pi/3),np.cos(np.pi/3)]]))*0.3-[2,-0.5]
np.random.shuffle(X)

plt.figure()

n_iter = 50
X_match = FIST_2D_similarity(X,Y,n_iter,plot=2)

plt.scatter(Y[:,0],Y[:,1],label = "Y data")
plt.scatter(X[:,0],X[:,1],label = "X data")
plt.scatter(X_match[:,0],X_match[:,1], label = "iteration "+str(n_iter))
plt.legend()


#%%################ test cross + circle #################

#data generation 

n = 10000
m = 100

Y1 = np.random.uniform(-1,1,size=(n//3,1))
Y1 = np.concatenate((Y1,np.zeros((n//3,1))),axis=1)

Y2 = np.random.uniform(-1,1,size=(n//3,1))
Y2 = np.concatenate((np.zeros((n//3,1)),Y2),axis=1)

Y3 = 2*np.pi*np.random.uniform(0,1,size=n-2*n//3)
Y3 = 0.5*np.vstack((np.cos(Y3),np.sin(Y3))).T+[1,1]

Y = np.concatenate((Y1,Y2,Y3),axis=0)
np.random.shuffle(Y)

X = Y[:m].copy().dot(np.array([[np.cos(np.pi/3),np.sin(np.pi/3)],[-np.sin(np.pi/3),np.cos(np.pi/3)]]))*0.3-[2,-0.5]
np.random.shuffle(X)

plt.figure()

n_iter = 20
X_match = FIST_2D_similarity(X,Y,n_iter,plot=2)

plt.scatter(Y[:,0],Y[:,1],label = "Y data")
plt.scatter(X[:,0],X[:,1],label = "X data")
plt.scatter(X_match[:,0],X_match[:,1], label = "iteration "+str(n_iter))
plt.legend()

#%%################ test line #################

#data generation 

n = 10000
m = 8000

Y = np.random.uniform(-1,1,size=(n,1))
Y = np.concatenate((Y,np.zeros((n,1))),axis=1)

np.random.shuffle(Y)
plt.figure()

X = Y[:m].copy().dot(np.array([[np.cos(np.pi/3),np.sin(np.pi/3)],[-np.sin(np.pi/3),np.cos(np.pi/3)]]))*0.3-[2,-0.5]
np.random.shuffle(X)

n_iter = 20
X_match = FIST_2D_similarity(X,Y,n_iter,plot=2)

plt.scatter(Y[:,0],Y[:,1],label = "Y data")
plt.scatter(X[:,0],X[:,1],label = "X data")
plt.scatter(X_match[:,0],X_match[:,1], label = "iteration "+str(n_iter))
plt.legend()
