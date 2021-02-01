#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:38:38 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from assignment import *
from PIL import Image

#%%################# test FIST image matching ########################

X = np.array(Image.open('Images/landscape4.jpeg'),dtype=float)[:,:,:3]
#X2 = np.array(Image.open('Images/starry_night.png'),dtype=float)[:,:,:3]
Y_ = Image.open('Images/landscape1.jpeg')

scale_list = [1.,1.33,1.66,2.0]

fig, axes = plt.subplots(nrows=2, ncols=3)

#we display the images
axes[0,0].imshow(X.astype(np.int))
axes[0,0].set_title('Input Image')
axes[0,0].axis("off")
axes[1,-1].imshow(np.array(Y_,dtype=np.int))
axes[1,-1].set_title('Target Image')
axes[1,-1].axis("off")

n_iter = 300
c = 1

for i,scale in enumerate(scale_list):    
    Y = Y_.resize((int(scale*Y_.width),int(scale*Y_.height)),0)
    Y = np.array(Y,dtype=float)
    X_match = FIST_image(X,Y,n_iter).clip(0,255)
    axes[(i+1)//3,(i+1)%3].imshow(X_match.astype(np.int))
    axes[(i+1)//3,(i+1)%3].set_title("scale : "+str(scale))
    axes[(i+1)//3,(i+1)%3].axis("off")
    
fig.tight_layout()

#%%

X = np.array(Image.open('Images/landscape4.jpeg'),dtype=float)[:,:,:3]
#X2 = np.array(Image.open('Images/starry_night.png'),dtype=float)[:,:,:3]
Y_ = Image.open('Images/landscape1.jpeg')

n_iter = 200
c = 1
scale=1.5

Y = Y_.resize((int(scale*Y_.width),int(scale*Y_.height)),0)
Y = np.array(Y,dtype=float)
X_match = FIST_features(X,Y,n_iter).clip(0,255)

plt.figure()
plt.imshow(X_match.astype(np.int))
plt.show()
