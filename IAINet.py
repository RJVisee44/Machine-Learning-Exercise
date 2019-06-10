# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:36:19 2019

@author: ryan4
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose
from keras.utils import plot_model

def IAINet(input_shape):
    #Create model
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),activation="relu",input_shape=input_shape))
    model.add(Conv2D(16,(5,5),activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(3,3),activation="relu"))
    model.add(Conv2DTranspose(32,(2,2)))
    model.add(Conv2D(16,(3,3)))

    plot_model(model, to_file='model.png')

    return model
