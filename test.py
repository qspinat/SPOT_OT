#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:34:45 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from assigment import *
from plot_assignment import *
from PIL import Image

#%%################################################################
      
rng = np.random.default_rng()
        
X = np.sort(np.random.uniform(1000,size=int(430)))
Y = np.sort(np.random.uniform(800,size=int(2400)))

#######################

start1 = time.time()
print(0, "starting optimal assigment")
t = assignment_opt_jit(X, Y)
end1 =  time.time()
print(end1-start1, "optimal assigment fisnished")
print("total time :", end1-start1)
print("cost :",cost(X,Y,t))
print()
#print(t)

start2 = time.time()
print(time.time()-start1, "starting injective optimal assigment")
a = quad_part_opt_ass_preprocessed(X,Y)
end2 =  time.time()
print(end2-start1, "injective optimal assigment finished")
print("total time :", end2-start2)
print("cost :",cost(X,Y,a))
print()
#print(a)

start3 = time.time()
print(time.time()-start1, "starting second injective optimal assigment")
a_bis = quad_part_opt_ass(X,Y)     
end3 =  time.time()
print(end3-start1, "second injective optimal assigment finished")
print("total time :", end3-start3)
print("cost :",cost(X,Y,a_bis))
print()

#%%

start4 = time.time()
print(time.time()-start4, "starting third injective optimal assigment with subproblem decomposition")
a_ter = assignment(X,Y)
end4 =  time.time()
print(end4-start4, "third injective optimal assigment finished")
print("total time :", end4-start4)
print("cost :",cost(X,Y,a_ter))

#plot_assignment(X,Y,t,'t')
#plot_assignment(X,Y,a,'a')
#plot_assignment(X,Y,a_bis,'a_bis')
#plot_assignment(X,Y,a_ter,'a_ter')

#%%################## test assignment decomposition #####################

start5 = time.time()
print(time.time()-start5, "starting subproblem decomposition")
A = assignment_decomp(X, Y)
end5 =  time.time()
print(end5-start5, "subproblem decomposition finished")
print("total time :", end5-start5)

#plot_assignment_decomp(X,Y,A)

#%%################# test FIST image matching ########################

X = np.array(Image.open('Images/mountains3.png'),dtype=float)[:,:,:3]
Y = np.array(Image.open('Images/mountains.jpg'),dtype=float)
Y2 = np.array(Image.open('Images/galaxy.jpg'),dtype=float)

mean_X = X.mean(axis=(0,1))
mean_Y = Y.mean(axis=(0,1))

n_iter = 300
c = 7

X_match = FIST_histogram_matching(X,Y,n_iter,c).clip(0,255)
X_match2 = FIST_histogram_matching(X,Y2,n_iter,c).clip(0,255)
#Y_match = FIST_histogram_matching(Y,Y2,n_iter,c).clip(0,255)

plt.figure()
plt.imshow(X.astype(np.int))

plt.figure()
plt.imshow(Y.astype(np.int))
plt.figure()
plt.imshow(X_match.astype(np.int))

plt.figure()
plt.imshow(Y2.astype(np.int))
plt.figure()
plt.imshow(X_match2.astype(np.int))

#plt.figure()
#plt.imshow(Y_match.astype(np.int))