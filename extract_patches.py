# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:40:54 2019

@author: ryan4
"""

import cv2
import numpy as np
import random
import skimage as sk

def random_rotation(train_img,train_label):
    """
    This function randomly rotates the train_img and train_label between 25% on left and right
    """
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(train_img, random_degree), sk.transform.rotate(train_label, random_degree)

def random_noise(train_img):
    """
    This function randomly adds noise to the train_img
    """
    return sk.util.random_noise(train_img)

def horizontal_flip(train_img,train_label):
    """
    This function simply flips the image and labels horizontally
    """
    return train_img[:, ::-1],train_label[:, ::-1]

def extract_patches(train_img,train_labels,n_dim,data_augmentation=False):
    """
    This function takes train_img and train_labels and splits it into n number 
    of image patches. There exists no overlap between patches. 
    
    Input:
        train_img, train_labels --> image name and corresponding labels to be split
        n_dim --> int:
            Desired input image size (n_dim,n_dim,3). Image will be resize if
            height/width is not divisible by n_dim.
            
    Output:
        img_patches,labels:
            Total patches of train_img and corresponding train_labels with no overlap.
            Total patches = (resized_image/n_dim)^2
            img_patches dim: (Total patches,n_dim,n_dim,3)
            labels dim: (Total patches,n_dim*n_dim,1)
            
    """
    
    assert(n_dim <= 512 and n_dim >0),"Image dimensions must be less than 512 and greater than 0!"
    
    train_img = cv2.imread(train_img)
    train_labels = cv2.imread(train_labels)
    
    if train_img.shape[0] % n_dim != 0: #Ensure all data is the same size
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
    
    if data_augmentation == True:
        print("Augmenting data...")
        total_patches = int(len(img_patches)/3)
        for i in range(total_patches):
            img_rot,lab_rot = random_rotation(img_patches[i],lab_patches[i])
            img_ran = random_noise(img_patches[i])
            img_flip,lab_flip = horizontal_flip(img_patches[i],lab_patches[i])
            
            img_patches.extend([img_rot.astype(int),img_ran.astype(int),img_flip.astype(int)])
            lab_patches.extend([lab_rot.astype(int),lab_patches[i],lab_flip.astype(int)])
            
    labels = []
    for i in range(len(lab_patches)):
        l = lab_patches[i].reshape(n_dim*n_dim,3)
        labels.append(l[:,1]) #RGB values are all the same, need pixel-wise
        
    return np.array(img_patches),np.round(np.array(labels)/255)

def get_data(train_img,train_label,n_dim,augment=False):
    """
    Input:
        train_img/train_labels --> numpy arrays
        n_dim --> height/width of images
        augment --> bool: perform data augmentation? 
        
    Output:
        train_imgs/train_labels/test_imgs/test_labels --> numpy arrays
    
    """    
    
    random.seed(44) #For reproducibility
    
    train_imgs, train_labels = extract_patches(train_img,train_label,n_dim,data_augmentation=augment)
    
    #Leave some data out for testing
    rand_id = random.sample(range(len(train_imgs)),int(len(train_imgs)/10))
    not_in = [x for x in range(len(train_imgs)) if x not in rand_id]
    test_imgs,test_labels = train_imgs[rand_id],train_labels[rand_id]
    train_imgs,train_labels = train_imgs[not_in],train_labels[not_in]

    return train_imgs, train_labels, test_imgs, test_labels