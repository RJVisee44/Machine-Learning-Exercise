# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:40:54 2019

@author: ryan4
"""

import cv2
import numpy as np
from sklearn.feature_extraction import image

#Preprocess the image for training
def extract_patches(train_img, train_labels,num_patches,n_dim):
    """
    This function takes train_img and train_labels and splits it into num_patches
    for training. Note that extract_patches_2d results in overlapping regions. 
    
    Input:
        train_img, train_labels --> image name and corresponding labels to be split
        num_patches --> int:
            Total number of patches that train_img and train_labels will be split into
        n_dim --> int:
            Desired height/width of input images into network
        
    Returns:
        img_patches,lab_patches:
            num_patches of train_img and corresponding train_labels        
    """
    
    
    train_img = cv2.imread(train_img)
    train_labels = cv2.imread(train_labels)

    img_patches = image.extract_patches_2d(train_img, (n_dim, n_dim),max_patches=num_patches,random_state=1)
    lab_patches = image.extract_patches_2d(train_labels, (n_dim, n_dim),max_patches=num_patches,random_state=1)

    return img_patches,lab_patches

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
            
    return img_patches,lab_patches