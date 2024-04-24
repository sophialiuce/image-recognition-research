#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 06:27:59 2018

@author: sophialiu
"""

from skimage.feature import hog  
from sklearn.externals import joblib  
import numpy as np   
import cv2  
import os  
import time  
  
# define parameter  
normalize = True  
visualize = False  
block_norm = 'L2-Hys'  
cells_per_block = [3,3]  
pixels_per_cell = [20,20]  
orientations = 9

IMG_SIZE = 150
size = 2184

x_matrix = np.empty((size, 2025))

def rgb2gray(im):  
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140  
    return gray

def extractFeatures():
    dirpath = "./data/images"
    cnt = 0
    for filename in os.listdir(dirpath):
        print("start extracting features: " + filename)
        if filename != '.DS_Store':
            data = cv2.imread(dirpath + '/' +filename)
            data = cv2.resize(data,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)  
            data = np.reshape(data, (IMG_SIZE * IMG_SIZE,3))
            
            image = np.reshape(data, (IMG_SIZE, IMG_SIZE, 3)) 
            gray = rgb2gray(image)/255.0 # train image to gray
            
            # Extract HoG features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize)
            
            x_matrix[cnt] = fd
            cnt += 1
            
def saveNpy():
    np.save(".hog_features.npy", x_matrix)
            
if __name__ == '__main__':  
    t0 = time.time()  
    
    extractFeatures()
    saveNpy()
    
    t1 = time.time()
    
    print ('The cast of time is:%f'%(t1-t0))