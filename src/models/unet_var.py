#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from keras import models, layers

def unet_var(input_shape=(512, 512, 4), num_class=2, num_filt_init=32):
    """
    A shallow variant of the U-net proposed by Ronneberger et al (2015), one major
    difference being that 2D convolutions don't reduce the size of the images.
    num_filt_init - the initial number of filters of the 2D conv 
    - uses Keras' functional API
    - doubles the number of filter units with each convolutional layer
    To Do:
    - generalize even more, by making the number of layers an input arg
    """
    # a few parameters that may be made input parameters
    layers.Dropout_rate = 0.5
    ki = 'he_normal'
    
    # initialization
    level = 0
    inputs = layers.Input(shape=input_shape)
    
    # --- DECODING PART
    # level 1: convolutions, maxpool and 'horizontal' copy
    level += 1
    conv1_1 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(inputs)
    conv1_2 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(conv1_1)
    copy1 = layers.Dropout(layers.Dropout_rate)(conv1_2)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
    # level 2: convolutions, maxpool and horizontal transfer
    level += 1
    conv2_1 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(pool1)
    conv2_2 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(conv2_1)
    copy2 = layers.Dropout(layers.Dropout_rate)(conv2_2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
    # level 3: convolutions (deepest layer)
    level += 1
    conv3_1 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(pool2)
    conv3_2 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same', kernel_initializer=ki)(conv3_1)

    
    # --- ENCODING PART
    # level 2: upsampling, convolution, concatenation, convolutions
    level -= 1
    upsample2 = layers.UpSampling2D((2, 2))(conv3_2)
    upconv2_1 = layers.Conv2D(num_filt_init * 2**(level-1), (2,2), activation='relu', padding='same')(upsample2)
    cat2 = layers.Concatenate(axis=3)([copy2, upconv2_1])
    upconv2_2 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same')(cat2)
    upconv2_3 = layers.Conv2D(num_filt_init * 2**(level-1), (3, 3), activation='relu', padding='same')(upconv2_2)
    
    # level 1: upsampling, convolution, concatenation, convolutions
    level -= 1
    upsample1 = layers.UpSampling2D((2, 2))(upconv2_3)
    upconv1_1 = layers.Conv2D(num_filt_init * 2**(level-1), (2,2), activation='relu', padding='same')(upsample1)
    cat1 = layers.Concatenate(axis=3)([copy1, upconv1_1])
    upconv1_2 = layers.Conv2D(num_filt_init * 2**(level-1), (3,3), activation='relu', padding='same')(cat1)
    upconv1_3 = layers.Conv2D(num_filt_init * 2**(level-1), (3, 3), activation='relu', padding='same')(upconv1_2)
    
    # output stage
    upconv1_4 = layers.Conv2D(2, (3, 3), activation='relu', padding='same')(upconv1_3)
    upconv1_5 = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(upconv1_4)
    
    
    model = models.Model(inputs=inputs, outputs=upconv1_5)
    return model