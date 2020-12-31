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

n = 30
alpha = 0.75
m = int(alpha*n)

X = np.sort(np.random.uniform(10,size=int(m)))
Y = np.sort(np.random.uniform(10,size=int(n)))

#%%################## true optimal assignement ##############

# start0 = time.time()
# print(0, "starting optimal assigment brut force")
# a_bf = brut_force(X, Y)
# end0 =  time.time()
# print(end0-start0, "optimal assigment fisnished")
# print("total time :", end0-start0)
# print("cost :",cost(X,Y,a_bf))
# print()

####################### 

start1 = time.time()
print(0, "starting optimal assigment")
t = assignment_opt(X, Y)
end1 =  time.time()
print(end1-start1, "optimal assigment fisnished")
print("total time :", end1-start1)
print("cost :",cost(X,Y,t))
print()
#print(t)

plot_assignment(X,Y,t)#,'t')

#%%

start2 = time.time()
print(time.time()-start1, "starting injective optimal assigment")
a = quad_part_opt_ass_preprocessed(X,Y)
end2 =  time.time()
print(end2-start1, "injective optimal assigment finished")
print("total time :", end2-start2)
print("cost :",cost(X,Y,a))
print()
#print(a)

plot_assignment(X,Y,a)#,'a')


#%%

start3 = time.time()
print(time.time()-start1, "starting second injective optimal assigment")
a_bis = quad_part_opt_ass(X,Y)     
end3 =  time.time()
print(end3-start1, "second injective optimal assigment finished")
print("total time :", end3-start3)
print("cost :",cost(X,Y,a_bis))
print()

#

start4 = time.time()
print(time.time()-start4, "starting third injective optimal assigment with subproblem decomposition")
a_ter = assignment(X,Y)
end4 =  time.time()
print(end4-start4, "third injective optimal assigment finished")
print("total time :", end4-start4)
print("cost :",cost(X,Y,a_ter))
print()

#

# start5 = time.time()
# print(time.time()-start5, "starting fourth injective optimal assigment with subproblem decomposition")
# a_quad = assignment_bis(X,Y)
# end5 =  time.time()
# print(end5-start5, "fourth injective optimal assigment finished")
# print("total time :", end5-start5)
# print("cost :",cost(X,Y,a_quad))
# print()

#plot_assignment(X,Y,t,'t')
#plot_assignment(X,Y,a,'a')
#plot_assignment(X,Y,a_bis,'a_bis')
#plot_assignment(X,Y,a_ter,'a_ter')

#%%################## test assignment decomposition #####################

start6 = time.time()
print(time.time()-start6, "starting subproblem decomposition")
A = assignment_decomp(X, Y)
end6 =  time.time()
print(end6-start6, "subproblem decomposition finished")
print("total time :", end6-start6)

plot_assignment_decomp(X,Y,A)

#%%################ test fist 2D shape matching #################

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


n_iter = 5000
c = 0.1
X_match = FIST_2D_solid(X,Y,n_iter,alpha=0.5,plot=5)

plt.scatter(Y[:,0],Y[:,1],label = "Y data")
plt.scatter(X[:,0],X[:,1],label = "X data")
plt.scatter(X_match[:,0],X_match[:,1], label = "matched X")
plt.legend()


