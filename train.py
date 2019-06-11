# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:17:39 2019

@author: ryan4
"""

from IAINet import IAINet
from extract_patches import noOverlap_extract
from sklearn.model_selection import KFold
from keras.optimizers import SGD
import random

n_dim = 512

print("Extracting images for training...")
train_imgs, train_labels = noOverlap_extract('image.tif','labels.tif',n_dim)

#Leave some data out for testing
rand_id = random.sample(range(len(train_imgs)),int(len(train_imgs)/10))
not_in = [x for x in range(len(train_imgs)) if x not in rand_id]
test_imgs,test_labels = train_imgs[rand_id],train_labels[rand_id]
train_imgs,train_labels = train_imgs[not_in],train_labels[not_in]

model = IAINet((n_dim,n_dim,3))

#Set up K-Fold cross-validation due to lack of data
folds = 5
kf = KFold(folds, shuffle = True) 
val_accuracy = train_accuracy = []

for train_index, val_index in kf.split(train_imgs):
    #K-1 used for training, last K fold used for testing/validation
    data_train, data_val = train_imgs[train_index], train_imgs[val_index]
    labels_train, labels_val = train_labels[train_index], train_labels[val_index]
    
    model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.01), metrics=["accuracy"])
    history = model.fit(data_train,labels_train,epochs=5,verbose=2,validation_data=(data_val,labels_val))
    train_pred = model.predict(data_train)
    val_pred = model.predict(data_val)
    val_accuracy.append((val_pred == labels_val).mean())
    train_accuracy.append((train_pred == labels_train).mean())