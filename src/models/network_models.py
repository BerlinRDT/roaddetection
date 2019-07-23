#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of network models.
"""

from keras import models, layers

def unet_var(input_shape=(512, 512, 4), num_class=2, num_filt_init=32):
    """
    A shallow variant of the U-net proposed by Ronneberger et al (2015). A major
    difference to the original model is that 2D convolutions don't reduce the 
    size of the images.
    To Do:
    - generalize even more, by making the number of levels an input arg
    
    Input
    -----
        input_shape: shape of the expected input
        num_class: number of classes to be discriminated, default 2 (currently
          without effect)
        num_filt_init: initial number of filters of the 2D conv (will be 
          doubled at each level of the decoding part)
    Output
    -----
        model: the Keras network model

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


def unet(input_size=(512, 512, 4)):
    """
    Original U-net as used for the RDT portfolio project.
    
    Input
    -----
        input_size: shape of the expected input
    
    Output
    -----
        model: the Keras network model
    """
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(input=inputs, output=conv10)

    return model


def segnet(input_size=(512, 512, 4)):
    """
    Segnet as used for the RDT portfolio project. Outdated Syntax 
    (ZeroPadding2D) is left for reproducibility.
    
    Input
    -----
        input_size: shape of the expected input
    
    Output
    -----
        model: the Keras network model
    """
    
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    inputs = layers.Input(input_size)

    # encoder

    padding1 = layers.ZeroPadding2D(padding=(pad, pad))(inputs)
    conv1 = layers.Conv2D(filter_size, kernel, activation='relu', border_mode='valid')(padding1)
    norm1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(norm1)

    padding2 = layers.ZeroPadding2D(padding=(pad, pad))(pool1)
    conv2 = layers.Conv2D(256, kernel, activation='relu', border_mode='valid')(padding2)
    norm2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(norm2)

    padding3 = layers.ZeroPadding2D(padding=(pad, pad))(pool2)
    conv3 = layers.Conv2D(512, kernel, activation='relu', border_mode='valid')(padding3)
    norm3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(norm3)

    # decoder

    padding4 = layers.ZeroPadding2D(padding=(pad, pad))(pool3)
    conv4 = layers.Conv2D(512, kernel, border_mode='valid')(padding4)
    norm4 = layers.BatchNormalization()(conv4)

    upsampling5 = layers.UpSampling2D(size=(pool_size, pool_size))(norm4)
    padding5 = layers.ZeroPadding2D(padding=(pad, pad))(upsampling5)
    conv5 = layers.Conv2D(256, kernel, border_mode='valid')(padding5)
    norm5 = layers.BatchNormalization()(conv5)

    upsampling6 = layers.UpSampling2D(size=(pool_size, pool_size))(norm5)
    padding6 = layers.ZeroPadding2D(padding=(pad, pad))(upsampling6)
    conv6 = layers.Conv2D(128, kernel, border_mode='valid')(padding6)
    norm6 = layers.BatchNormalization()(conv6)

    upsampling7 = layers.UpSampling2D(size=(pool_size, pool_size))(norm6)
    padding7 = layers.ZeroPadding2D(padding=(pad, pad))(upsampling7)
    conv7 = layers.Conv2D(filter_size, kernel, border_mode='valid')(padding7)
    norm7 = layers.BatchNormalization()(conv7)


    conv8 = layers.Conv2D(1, 1, activation='sigmoid')(norm7)

    model = models.Model(input=inputs, output=conv8)

    return model
