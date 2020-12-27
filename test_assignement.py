#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:32:21 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

#%%


def test_assignment(assigment_func,alpha,n_iter,n_moy):
    size = np.logspace(1,6,n_iter,dtype=np.int64)
    liste_time = np.zeros(n_iter)
    X = np.sort(np.random.uniform(10000,size=5))
    Y = np.sort(np.random.uniform(10000,size=10))
    t = assigment_func(X, Y)
    for i,n in enumerate(size):
        print("n =",n)
        m = int(alpha*n)
        Y = np.sort(np.random.uniform(10000,size=int(n)))
        for k in range(n_moy):
            X = np.sort(np.random.uniform(10000,size=int(m)))
            start = time.perf_counter()
            t = assigment_func(X, Y)
            end =  time.perf_counter()
            liste_time[i]+=end-start
        liste_time[i]/=n_moy
    return size,liste_time


#%% nearest neighbor

from assigment import *

alpha = 0.75

size,time_t = test_assignment(assignment_opt, alpha, 20, 100)

plt.figure()
plt.loglog(size,time_t,label="t")
plt.xlabel("n")
plt.ylabel("time (s)")
plt.grid(True, which="both")
plt.title("Time for Nearest Neighbor Assigment")
plt.legend()
plt.show()

print("ordre :",(np.log(time_t[-1])-np.log(time_t[-2]))/(np.log(size[-1])-np.log(size[-2])))

#%% quadratic optimal assignement

alpha = 0.5

size,time_a = test_assignment(quad_part_opt_ass_preprocessed, alpha=alpha, n_iter=20, n_moy=100)

plt.figure()
plt.loglog(size,time_a,label="a")
plt.xlabel("n")
plt.ylabel("time (s)")
plt.grid(True, which="both")
plt.title("Time for quadratic optimal injective Assigment with nearest neighbor")
plt.legend()
plt.show()

print("ordre :",(np.log(time_a[-1])-np.log(time_a[-2]))/(np.log(size[-1])-np.log(size[-2])))

#%%

plt.figure()
plt.loglog(size,time_a-time_t,label="a")
plt.xlabel("n")
plt.ylabel("time (s)")
plt.grid(True, which="both")
plt.title("Time for quadratic optimal injective Assigment without nearest neighbor")
plt.legend()
plt.show()

print("ordre :",(np.log(time_a[-1]-time_t[-1])-np.log(time_a[-2]-time_t[-2]))/(np.log(size[-1])-np.log(size[-2])))


#%% optimal assignement

alpha = 0.9

size,time_a_bis = test_assignment(assignment, alpha=alpha, n_iter=20, n_moy=100)

plt.figure()
plt.loglog(size,time_a_bis,label="a")
plt.xlabel("n")
plt.ylabel("time (s)")
plt.grid(True, which="both")
plt.title("Time for optimal injective Assigment")
plt.legend()
plt.show()

print("ordre :",(np.log(time_a_bis[-1])-np.log(time_a_bis[-2]))/(np.log(size[-1])-np.log(size[-2])))

