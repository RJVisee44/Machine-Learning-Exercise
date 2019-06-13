# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:23:55 2019

@author: ryan4
"""

from extract_patches import get_data, extract_patches
from train import train_model
from prediction import prediction
import numpy as np
from PIL import Image

def run_code(train_img, train_label, n_dim, weights,train=False,valid=False):
    
    print("Getting images for training/testing...")
    train_imgs, train_labels, test_imgs, test_labels = get_data(train_img,train_label,n_dim,augment=True)
    
    if train == True:
        train_model(train_imgs,train_labels,n_dim,valid=valid)
        weights = 'Weights/weights.h5' #Make adaptive
        
    test_pred = prediction(test_imgs,test_labels,weights,n_dim)
    
    #To test on entire image:
    train_imgs, train_labels = extract_patches(train_img,train_label,n_dim)
    img_pred = prediction(train_imgs,train_labels,weights,n_dim)
    
    #Convert confidences to 0 or 1, use 20% threshold
    img_pred[np.where(img_pred >= 0.2)] = 1
    img_pred[np.where(img_pred < 0.2)] = 0
    
    #Reshape into image
    count = 0
    i = 0
    step = int((((5000*5000)/(n_dim*n_dim))/n_dim)/2)
    while i < 9990: #This can be constant for a 5000x5000x3 image
        img_hor = img_pred[i].reshape(n_dim,n_dim)
        for j in range(i+1,i+step):
            img_hor = np.concatenate((img_hor,img_pred[j].reshape(n_dim,n_dim)))
        if count == 0:
            img = img_hor
            count = 1
        else:
            img = np.concatenate((img,img_hor),axis=1)
        i += step
    
    img = Image.fromarray(img*255,'I')
    img.save("result.png")
    
    return test_pred
    
n_dim = 50
weights = 'Weights/weights.h5'
test_pred = run_code('image.tif','labels.tif',n_dim,weights,train=False,valid=False)
    
        
        
    