import os
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

def pixel_shuffle(scale):
    '''
    This function implements pixel shuffling.
    ATTENTION: the scale should be bigger than 2, otherwise just returns the input.
    '''
    if scale > 1:
        return lambda x: tf.nn.depth_to_space(x, scale)
    else:
        return lambda x:x

def add_down_block(x_inp, filters, kernel_size=(3, 3), padding="same", strides=1,r=False):
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x_inp)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.BatchNormalization()(x)
    if r:
        # if r=True then we import an (1X1) Conv2D after input layer 
        # in order the dimensions of 2 tensors coincide.
        x_inp = K.layers.Conv2D(filters,(1,1), padding=padding, strides=strides)(x_inp)
    x = K.layers.Add()([x,x_inp])
    return x

def add_up_block(x_inp,skip,filters, kernel_size=(3, 3), padding="same", strides=1,upscale_factor=2):
    x = pixel_shuffle(scale=upscale_factor)(x_inp)
    x = K.layers.Concatenate()([x, skip])
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Activation('relu')(x)
    return x

def add_bottleneck(x_inp,filters, kernel_size=(3, 3), padding="same", strides=1):
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x_inp)
    x = K.layers.Activation('relu')(x)
    return x



def RUNet():
    """
      Implementing with Keras the Robust UNet Architecture as proposed by
      Xiaodan Hu, Mohamed A. Naiel, Alexander Wong, Mark Lamm, Paul Fieguth
      in "RUNet: A Robust UNet Architecture for Image Super-Resolution"
    """
    inputs = K.layers.Input((64,256, 3))
    
    
    down_1 = K.layers.Conv2D(64,(7,7), padding="same", strides=1)(inputs)
    down_1 = K.layers.BatchNormalization()(down_1)
    down_1 = K.layers.Activation('relu')(down_1)
    
    down_2 = K.layers.MaxPool2D(pool_size=(2,2))(down_1)
    down_2 = add_down_block(down_2,64)
    down_2 = add_down_block(down_2,64)
    down_2 = add_down_block(down_2,64)
    down_2 = add_down_block(down_2,128,r=True)
    down_2 = K.layers.Dropout(0.5)(down_2)
    
    down_3 = K.layers.MaxPool2D(pool_size=(2, 2),strides=2)(down_2)
    down_3 = add_down_block(down_3,128)
    down_3 = add_down_block(down_3,128)
    down_3 = add_down_block(down_3,128)
    down_3 = add_down_block(down_3,256,r=True)
    down_3 = K.layers.Dropout(0.5)(down_3)
    
    down_4 = K.layers.MaxPool2D(pool_size=(2, 2))(down_3)
    down_4 = add_down_block(down_4,256)
    down_4 = add_down_block(down_4,256)
    down_4 = add_down_block(down_4,256)
    down_4 = add_down_block(down_4,256)
    down_4 = add_down_block(down_4,256)
    down_4 = add_down_block(down_4,512,r=True) 
    down_4 = K.layers.Dropout(0.5)(down_4)
    
    down_5 = K.layers.MaxPool2D(pool_size=(2, 2))(down_4)
    down_5 = add_down_block(down_5,512)
    down_5 = add_down_block(down_5,512)
    down_5 = K.layers.BatchNormalization()(down_5)
    down_5 = K.layers.Activation('relu')(down_5)
    
    
    bn_1 = add_bottleneck(down_5, 1024)
    bn_2 = add_bottleneck(bn_1, 512)
    bn_2 = K.layers.Dropout(0.5)(bn_2)
    
    up_1 = add_up_block(bn_2,down_5, 512,upscale_factor=1)
    up_2 = add_up_block(up_1,down_4, 384,upscale_factor=2)
    up_3 = add_up_block(up_2,down_3, 256,upscale_factor=2)
    up_4 = add_up_block(up_3,down_2, 96,upscale_factor=2) 
    up_4 = K.layers.Dropout(0.5)(up_4)
    
    up_5 = pixel_shuffle(scale=2)(up_4)
    up_5 = K.layers.Concatenate()([up_5,down_1])
    up_5 = K.layers.Conv2D(99,(3,3), padding="same", strides=1)(up_5)
    up_5 = K.layers.Activation('relu')(up_5)
    up_5 = K.layers.Conv2D(99,(3,3), padding="same", strides=1)(up_5)
    up_5 = K.layers.Activation('relu')(up_5)
   
    outputs = K.layers.Conv2D(3,(1,1), padding="same")(up_5)
    model = K.models.Model(inputs, outputs)
    return model
