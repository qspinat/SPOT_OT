#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:38:38 2020

@author: qspinat
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from assigment import *
from PIL import Image

#%%################# test FIST image matching ########################

X = np.array(Image.open('Images/landscape4.jpeg'),dtype=float)[:,:,:3]
#X2 = np.array(Image.open('Images/starry_night.png'),dtype=float)[:,:,:3]
Y_ = Image.open('Images/landscape1.jpeg')

scale_list = [1.,1.2,1.4,1.6,1.8,2.]

fig, axes = plt.subplots(nrows=2, ncols=4)

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
    axes[(i+1)//4,(i+1)%4].imshow(X_match.astype(np.int))
    axes[(i+1)//4,(i+1)%4].set_title("scale : "+str(scale))
    axes[(i+1)//4,(i+1)%4].axis("off")
    
fig.tight_layout()