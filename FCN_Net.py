# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:36:19 2019

@author: ryan4
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose,Reshape

def FCN_Net(input_shape):
    """
    input shape -> int: (width,height,3)
        Input shape of the image passed to the network    
    """
    
    #Create model
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),padding='same',input_shape=input_shape,activation="relu"))
    model.add(Conv2D(16,(5,5),padding='same',activation="relu"))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32,(3,3),padding='same',activation="relu"))
    model.add(Conv2DTranspose(32,(2,2), strides =(2,2)))
    
    #Sigmoid activation for output
    model.add(Conv2D(16,(3,3),padding='same',activation='sigmoid'))
    model.add(Reshape((input_shape[0]*input_shape[0], -1)))
    #Note: padding can get input equal to output
    
    return model

