# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:40:54 2019

@author: ryan4
"""

import cv2
import numpy as np
from sklearn.feature_extraction import image



def noOverlap_extract(train_img,train_labels,n_dim):
    """
    This function takes train_img and train_labels and splits it into n number 
    of image patches. There exists no overlap between patches. 
    
    Input:
        train_img, train_labels --> image name and corresponding labels to be split
        n_dim --> int:
            Desired input image size (n_dim,n_dim,3). Image will be resize if
            height/width is not divisible by n_dim.
            
    Output:
        img_patches,lab_patches:
            Total patches of train_img and corresponding train_labels with no overlap.
            Total patches = (resized_image/n_dim)^2
            
    """
    
    train_img = cv2.imread(train_img)
    train_labels = cv2.imread(train_labels)
    
    if train_img.shape[0] % n_dim != 0:
        new_size = int(np.round(train_img.shape[0]/n_dim) * n_dim)
        train_img = cv2.resize(train_img,(new_size,new_size))
        train_labels = cv2.resize(train_labels,(new_size,new_size))
        
    moves = int(train_img.shape[0]/n_dim)
    img_patches = []
    lab_patches = []
        
    for i in range(moves):
        for j in range(moves):
            img_patches.append(train_img[n_dim*j:n_dim*(1+j),n_dim*i:n_dim*(1+i)])
            lab_patches.append(train_labels[n_dim*j:n_dim*(1+j),n_dim*i:n_dim*(1+i)])
            
    labels = []
    for i in range(len(lab_patches)):
        maxes = []
        for j in range(n_dim):
            for k in range(n_dim):
                maxes.append(max(lab_patches[i][j][k]))
        labels.append(maxes)
        
    return np.array(img_patches),np.array(labels)