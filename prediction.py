# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:18:11 2019

@author: ryan4
"""

from IAINet import IAINet
from keras.optimizers import SGD

def prediction(test_imgs,test_labels,weights,n_dim):
    """
    Run IAINet on test_imgs.
    
    Input:
        test_imgs --> numpy array containing all test images
        weights --> str containing filename of weights to be loaded and tested
        n_dim --> height/width of test_imgs   
    
    Output:
        preds --> numpy array containing the predictions of each pixel on test_imgs
    """    
    
    print("Loading model...")
    model = IAINet((n_dim,n_dim,3))
    print(model.summary())
    
    print("Loading weights...")
    model.load_weights(weights)
    model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.1,momentum=0.9,decay=1e-6,nesterov=True), metrics=["accuracy"])    
    
    print('Predicting...')
    preds = model.predict(test_imgs)
    
    print('Evaluating...')
    score = model.evaluate(test_imgs,test_labels,verbose=1)
    
    #Accuracy may not be best metric because many more 0s than 1s (~50mill more)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    return preds