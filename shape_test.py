#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:47:55 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from assigment import *

#%%################ test fist 2D shape matching #################

#data generation 

n = 10000
m = 500

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


n_iter = 100
X_match = FIST_2D_similarity(X,Y,n_iter,plot=None)

plt.scatter(Y[:,0],Y[:,1],label = "Y data")
plt.scatter(X[:,0],X[:,1],label = "X data")
plt.scatter(X_match[:,0],X_match[:,1], label = "matched X")
plt.legend()
