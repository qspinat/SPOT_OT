#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:13:21 2020

@author: qspinat
"""


import numpy as np
import numba as nb
from numba import jit
import matplotlib.pyplot as plt
import time

@jit(nopython=True)
def assignment_opt_jit(X,Y,t):
    """
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement

    Returns function t wich assign X[i] to Y[t[i]]
    -------
    None.

    """
    m = X.shape[0]
    n= Y.shape[0]
    
    #t = nb.int32(np.zeros(m))
    i=0
    j=0
    while(i<m):
        if j==n-1:
            t[i]=j
            i+=1
        elif X[i]>Y[j]:
            j+=1
        elif j==0:
            t[i] = j
            i+=1
        else:
            if np.abs(X[i]-Y[j])<np.abs(X[i]-Y[j-1]):
                t[i]=j
            else:
                t[i]=j-1
            i+=1
    return t

def assignment_opt(X,Y):
    t = np.zeros(X.shape[0],dtype=np.int)
    t = assignment_opt_jit(X, Y, t)
    return t

@jit(nopython=True)
def quad_part_opt_ass_preprocessed_jit(X,Y,t,a):
    """
    Parameters
    ----------
    X : sorted X, size m<n, allready preprocessed
    Y : sorted Y, size n, allready preprocessed
    t : optimal assigment X to Y (allready done)
    a : array of integer to stock the injective assignement

    Returns a injective optimal assignement
    -------
    None doesn't simplify the problem
    """
    
    m = X.shape[0]
    n = Y.shape[0]
#    a = np.zeros(m,dtype=np.int)

    
    #initialization
    a[0] = t[0]
    s = a[0]-1
    r = 0
    
    for i in range(0,m-1):
        if t[i+1]>a[i]:
            a[i+1] = t[i+1]
            if a[i+1]>a[i]+1:
                s = a[i+1]-1
                r = i+1
        else:
            #subcases
            if s<0:
                #case 2
                a[i+1] = a[i]+1
            elif a[i]==n-1:
                #case 1
                a[i+1]=a[i]
                a[r:i+1]=np.arange(s,a[i+1])
                # change s and r
                for j in range(r,-1,-1):
                    if j == 0:
                        s = a[0]-1
                        r = 0
                    elif a[j]>a[j-1]+1:
                        s = a[j]-1
                        r = j
                        break
            
            else:
            # compute sums
#                print(a[i])
#                print(Y.shape)
#                print(Y[a[i]])
                w1 = np.sum((X[r:i+1]-Y[a[r:i+1]-1])**2) + (X[i+1]-Y[a[i]])**2
                w2 = np.sum((X[r:i+1]-Y[a[r:i+1]])**2) + (X[i+1]-Y[a[i]+1])**2
                
                #case 1
                if w1<w2:
                    a[i+1]=a[i]
                    a[r:i+1]=np.arange(s,a[i+1])
                    # change s and r
                    for j in range(r,-1,-1):
                        if j == 0:
                            s = a[0]-1
                            r = 0
                        elif a[j]>a[j-1]+1:
                            s = a[j]-1
                            r = j
                            break
                #case 2
                else:
                    a[i+1]=a[i]+1
#        print(i,s,r)
#        print(a)
#        print()
    return a

def quad_part_opt_ass_preprocessed(X,Y):
    t = assignment_opt(X, Y)
    a = np.zeros(X.shape[0],dtype=np.int)
    a = quad_part_opt_ass_preprocessed_jit(X,Y,t,a)
    return a

@jit(nopython=True)
def quad_part_opt_ass_jit(X,Y,t,a):
    """

    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement
    a : array of integer to stock the injective assignement

    Returns a optimal injective assigment t
    -------
    None.

    """
    m = X.shape[0]
    n = Y.shape[0]
    
    #################  symplifying the problem ###############
    
    # n == m
    if n==m:
        for i in range(m):
            a[i] = i
            return a
    
    # m == n-1
    
        
#    a = np.zeros(m,dtype=np.int)
  
    # X before Y
    ind_min=0
    while X[ind_min]<=Y[ind_min]:
        a[ind_min]=ind_min
        ind_min+=1
        if ind_min == m:
            return a
    
    
    # X after Y
    ind_max=-1
    while X[m+ind_max]>=Y[n+ind_max]:
        a[m+ind_max]=n+ind_max
        ind_max-=1
        if m+ind_max == ind_min:
            return a
        
#    print("OK",ind_min,ind_max)
    
    t = t[ind_min:m+ind_max+1]
    t = assignment_opt_jit(X[ind_min:m+ind_max+1],Y[ind_min:n+ind_max+1],t)
    
    # number of non-injective values of t
    p = t.shape[0]-np.unique(t).shape[0]
    if p==0:
        a[ind_min:m+ind_max+1]=t+ind_min
        return a
       
    #ind_min_Y = np.max((0,t[0]-p))+ind_min
    #ind_max_Y = np.min((n+ind_max-ind_min,t[m+ind_max-ind_min]+p))+ind_min
    # for numba
    if t[0]-p>0:
        ind_min_Y = ind_min+t[0]-p
    else :
        ind_min_Y = ind_min
    
    if n+ind_max-ind_min<t[m+ind_max-ind_min]+p:
        ind_max_Y = n+ind_max-ind_min+ind_min
    else :
        ind_max_Y = t[m+ind_max-ind_min]+p+ind_min
#    print("OKOK",ind_min_Y,ind_max_Y-n)
    
    # assigment    
    a[ind_min:m+ind_max+1] = quad_part_opt_ass_preprocessed_jit(X[ind_min:m+ind_max+1], Y[ind_min_Y:ind_max_Y+1], t-ind_min_Y+ind_min,a[ind_min:m+ind_max+1])+ind_min_Y
    
    return a

def quad_part_opt_ass(X,Y):
    t = assignment_opt(X, Y)
    a = np.zeros(X.shape[0],dtype=np.int)
    a = quad_part_opt_ass_jit(X,Y,t,a)
    return a
  
#%%################################################################
      
rng = np.random.default_rng()
        
X = np.sort(rng.choice(int(6*10e1),size=int(1*10e1),replace=False))
Y = np.sort(rng.choice(int(6*10e1),size=int(2*10e1),replace=False))

#%%######################

start1 = time.time()
print(0, "starting optimal assigment")
t = assignment_opt(X, Y)
end1 =  time.time()
print(end1-start1, "optimal assigment fisnished")
print("total time :", end1-start1)
print()
#print(t)

start2 = time.time()
print(time.time()-start1, "starting injective optimal assigment")
a = quad_part_opt_ass_preprocessed(X,Y)
end2 =  time.time()
print(end2-start1, "injective optimal assigment finished")
print("total time :", end2-start2)
print()
#print(a)

start3 = time.time()
print(time.time()-start1, "starting second injective optimal assigment")
a_bis = quad_part_opt_ass(X,Y)     
end3 =  time.time()
print(end3-start1, "second injective optimal assigment finished")
print("total time :", end3-start3)
print()
#print(a_bis-a)

#%%################# plot ####################

def plot_assignment(X,Y,t,title=None):
    plt.figure()
    plt.scatter(X,np.ones(X.shape[0]), color='blue')
    plt.scatter(Y,np.zeros(Y.shape[0]), color="red")
    for i in range(t.shape[0]):
        plt.plot([X[i],Y[t[i]]],[1,0],color="black")
        plt.title(title)
    plt.show

plot_assignment(X,Y,t,'t')
plot_assignment(X,Y,a,'a')
plot_assignment(X,Y,a_bis,'a_bis')


#%%#################### Assigment Decomposition ####################

def assigment_decomp_test(X,Y,t,f,A_Y):
    """
    
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignementn size m
    f : array of boolean, size m
    A_Y : array of integer, A_Y[i] = Subproblem assignement of Y[i], size n

    Returns
    -------
    s = last available free spot
    l = currently last value considered

    """
    
    m = X.shape[0]
    n = Y.shape[0]
    
    for i in range(n):
        A_Y[i] = -1
    
    A = []
    
    for i in range(f.shape[0]):
        f[i] = True
        
    for i in range(m):
        # Nouveau subproblem
        if f[t[i]]:
            f[t[i]] = False
            # update s
            s = t[i]-1
            l = t[i]
            #create A_k
            A.append([[i],[t[i]],s,l])
            
            A_Y[t[i]] = len(A)-1

        
        # Mise a jour probleme
        else:
            # Premier cas : on extend à droite et à gauche
            k1=A_Y[t[i]]
            if t[i] == t[i-1]:
                # tant que s_k1 pris, on fusionne les problemes
                while A[k1][2]>=0 and not f[A[k1][2]] : 
                    k2 = A_Y[A[k1][2]] 
                    A[k1][0]=A[k2][0]+A[k1][0]
                    A[k1][1]=A[k2][1]+A[k1][1]
                    A[k1][2] = A[k2][2]
                    #A[k1][3] = A[k1][3]
                    for y in A[k2][1]:
#                        print(y)
                        A_Y[y] = k1
                    A[k2]=[]
                # tant que l_k1 +1 pris, on fusionne les problemes
                while A[k1][3]<n-1 and not f[A[k1][3]+1]:
                    k2 = A_Y[A[k1][3]+1]
                    A[k1][0]=A[k1][0]+A[k2][0]
                    A[k1][1]=A[k1][1]+A[k2][1]
                    #A[k1][2] = A[k1][2]
                    A[k1][3] = A[k2][3]
                    for y in A[k2][1]:
                        A_Y[y] = k1
                    A[k2]=[]
                A[k1][0].append(i)
                if A[k1][2] >=0 : 
                    f[A[k1][2]] = False
                    A_Y[A[k1][2]] = k1
                    A[k1][1].insert(0,A[k1][2])
                    A[k1][2]-=1
                if A[k1][3]<n-1 : 
                    f[A[k1][3]+1]=False
                    A_Y[A[k1][3]+1]=k1
                    A[k1][1].append(A[k1][3]+1)
                    A[k1][3]+=1
            # Deuxieme cas : on extend à droite uniquement
            else :
                # tant que l_k1 +1 pris, on fusionne les problemes
                while A[k1][3]<n-1 and not f[A[k1][3]+1]:
                    k2 = A_Y[A[k1][3]+1]
                    A[k1][0]=A[k1][0]+A[k2][0]
                    A[k1][1]=A[k1][1]+A[k2][1]
                    #A[k1][2] = A[k1][2]
                    A[k1][3] = A[k2][3]
                    for y in A[k2][1]:
                        A_Y[y] = k1                
                    A[k2]=[]
                A[k1][0].append(i)
                if A[k1][3]<n-1 : 
                    f[A[k1][3]+1]=False
                    A_Y[A[k1][3]+1]=k1
                    A[k1][1].append(A[k1][3]+1)
                    A[k1][3]+=1
            
    A_to_delete = []
    for i in range(len(A)):
        if not A[i]:
            A_to_delete.insert(0,i)
    for i in A_to_delete:
        A.pop(i)
            
    return A

@jit(nopython=True)
def assigment_decomp_jit(X,Y,t,f,A_Y):
    """
    
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignementn size m
    f : array of boolean, size m
    A_Y : array of integer, A_Y[i] = Subproblem assignement of Y[i], size n

    Returns
    -------
    s = last available free spot
    l = currently last value considered

    """
    
    m = X.shape[0]
    n = Y.shape[0]
    
#    for i in range(n):
#        A_Y[i] = -1

    AX = []
    AY = []
    As = []
    Al = []
    
#    for i in range(f.shape[0]):
#        f[i] = True
        
    for i in range(m):
        # Nouveau subproblem
        if f[t[i]]:
            f[t[i]] = False
            # update s
            s = t[i]-1
            l = t[i]
            #create A_k
            AX.append([i])
            AY.append([t[i]])
            As.append(s)
            Al.append(l)
            
            A_Y[t[i]] = len(AX)-1

        
        # Mise a jour probleme
        else:
            # Premier cas : on extend à droite et à gauche
            k1=A_Y[t[i]]
            if t[i] == t[i-1]:
                # tant que s_k1 pris, on fusionne les problemes
                while As[k1]>=0 and not f[As[k1]] : 
                    k2 = A_Y[As[k1]] 
                    AX[k1]=AX[k2]+AX[k1]
                    AY[k1]=AY[k2]+AY[k1]
                    As[k1] = As[k2]
                    # AX[k2]=[[i]]
                    # AY[k2]=[[t[i]]]
                    As[k2]=-2
                    Al[k2]=-2
                    for y in AY[k2]:
                        A_Y[y] = k1
                # tant que l_k1 +1 pris, on fusionne les problemes
                while Al[k1]<n-1 and not f[Al[k1]+1]:
                    k2 = A_Y[Al[k1]+1]
                    AX[k1]=AX[k1]+AX[k2]
                    AY[k1]=AY[k1]+AY[k2]
                    Al[k1] = Al[k2]
                    # AX[k2]=[]
                    # AY[k2]=[]
                    As[k2]=-2
                    Al[k2]=-2
                    for y in AY[k2]:
                        A_Y[y] = k1
                AX[k1].append(i)
                if As[k1] >=0 : 
                    f[As[k1]]=False
                    A_Y[As[k1]]=k1
                    AY[k1].insert(0,As[k1])
                    As[k1]-=1
                if Al[k1]<n-1 : 
                    f[Al[k1]+1]=False
                    A_Y[Al[k1]+1]=k1
                    AY[k1].append(Al[k1]+1)
                    Al[k1]+=1
            # Deuxieme cas : on extend à droite uniquement
            else :
                # tant que l_k1 +1 pris, on fusionne les problemes
                while Al[k1]<n-1 and not f[Al[k1]+1]:
                    k2 = A_Y[Al[k1]+1]
                    AX[k1]=AX[k1]+AX[k2]
                    AY[k1]=AY[k1]+AY[k2]
                    Al[k1] = Al[k2]
                    # AX[k2]=[]
                    # AY[k2]=[]
                    As[k2]=-2
                    Al[k2]=-2
                    for y in AY[k2]:
                        A_Y[y] = k1
                AX[k1].append(i)
                if Al[k1]<n-1 : f[Al[k1]+1]=False
                if Al[k1]<n-1 : AY[k1].append(Al[k1]+1)
                if Al[k1]<n-1 : Al[k1]+=1
            
    A_to_delete = []
    for i in range(len(As)):
        if As[i]==-2:
            A_to_delete.insert(0,i)
    for i in A_to_delete:
        AX.pop(i)
        AY.pop(i)
        As.pop(i)
        Al.pop(i)
          
    return AX,AY,As,Al

def assigment_decomp(X, Y):
    t = assignment_opt(X, Y)
    f = np.ones(Y.shape[0],dtype='bool')
    A_Y = -np.ones(Y.shape[0],dtype='int')
    return assigment_decomp_jit(X, Y, t, f, A_Y)
    
        
#%%################## test assigment decomposition #####################

f = np.ones(Y.shape[0],dtype='bool')
A_Y = -np.ones(Y.shape[0],dtype='int')

A = assigment_decomp(X, Y)

#%%##################### plot assigment decomp ########################

def plot_assignment_decomp(X,Y,A,title=None,colors=['b','g','r','c','m','y']):
    plt.figure()
    for i in range(len(A[0])):
        plt.scatter(X[A[0][i]],np.ones(len(A[0][i])), color=colors[i%len(colors)])
        plt.scatter(Y[A[1][i]],np.zeros(len(A[1][i])), color=colors[i%len(colors)])
        plt.plot([X[A[0][i][0]],X[A[0][i][-1]]],[1,1],color=colors[i%len(colors)])
        plt.plot([Y[A[1][i][0]],Y[A[1][i][-1]]],[0,0],color=colors[i%len(colors)])
        plt.plot([X[A[0][i][0]],Y[A[1][i][0]]],[1,0],color=colors[i%len(colors)])
        plt.plot([X[A[0][i][-1]],Y[A[1][i][-1]]],[1,0],color=colors[i%len(colors)])
    plt.title(title)
    plt.show()

plot_assignment_decomp(X,Y,A)