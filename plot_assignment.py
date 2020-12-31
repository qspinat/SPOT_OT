#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:33:04 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_assignment(X,Y,t,title=None):
    plt.figure()
    plt.scatter(X,np.ones(X.shape[0]), color='blue')
    plt.scatter(Y,np.zeros(Y.shape[0]), color="red")
    for i in range(t.shape[0]):
        plt.plot([X[i],Y[t[i]]],[1,0],color="black")
    plt.title(title)
    plt.axis("off")
    plt.show()
    
def plot_assignment_decomp(X,Y,A,title=None,colors=['b','g','r','c','m','y']):
    plt.figure()
    for i in range(A.shape[0]):
        plt.scatter(X[A[i][0]:A[i][1]+1],np.ones(A[i][1]-A[i][0]+1), color=colors[i%len(colors)])
        plt.scatter(Y[A[i][2]+1:A[i][3]+1],np.zeros(A[i][3]-A[i][2]-1+1), color=colors[i%len(colors)])
        plt.plot([X[A[i][0]],X[A[i][1]]],[1,1],color=colors[i%len(colors)])
        plt.plot([Y[A[i][2]+1],Y[A[i][3]]],[0,0],color=colors[i%len(colors)])
        plt.plot([X[A[i][0]],Y[A[i][2]+1]],[1,0],color=colors[i%len(colors)])
        plt.plot([X[A[i][1]],Y[A[i][3]]],[1,0],color=colors[i%len(colors)])
    plt.title(title)
    plt.axis("off")
    plt.show()