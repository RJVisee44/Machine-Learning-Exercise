# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:36:19 2019

@author: ryan4
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose,Flatten,Dense
from keras.utils import plot_model


def IAINet(input_shape):
    """
    input shape -> int: (width,height,3)
        Input shape of the image passed to the network    
    """
    
    #Create model
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),input_shape=input_shape,activation="relu"))
    model.add(Conv2D(16,(5,5),activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(3,3),activation="relu"))
    model.add(Conv2DTranspose(32,(2,2)))
    model.add(Conv2D(16,(3,3)))
    
    #Need fully connected layer for output 
    model.add(Flatten())
    model.add(Dense(input_shape[0]*input_shape[0], input_shape=(input_shape[0], ), activation='sigmoid'))
    
    #plot_model(model, to_file='model.png')

    return model
