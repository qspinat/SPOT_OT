#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:34:45 2020

@author: qspinat
"""
import numpy as np
import time

from assigment import *
from plot_assignment import *

#%%################################################################
      
rng = np.random.default_rng()
        
X = np.sort(rng.choice(int(5*10e5),size=int(1*10e3),replace=False))
Y = np.sort(rng.choice(int(5*10e5),size=int(5*10e3),replace=False))

#%%######################

start1 = time.time()
print(0, "starting optimal assigment")
t = assignment_opt(X, Y)
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

#%%################## test assigment decomposition #####################

start5 = time.time()
print(time.time()-start5, "starting subproblem decomposition")
A = assignment_decomp(X, Y)
end5 =  time.time()
print(end5-start5, "subproblem decomposition finished")
print("total time :", end5-start5)

#plot_assignment_decomp(X,Y,A)
