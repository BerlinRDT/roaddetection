#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of network models.
"""

from keras import models, layers

def unet_flex(input_shape=(512, 512, 4), num_class=2, num_level=4, num_filt_init=32):
    """
    U-net-like model. Highly configurable.
    Note that 2D convolutions don't reduce the size of the images, in contrast
    to the original description by Ronneberger et al (2015).
    
    Input
    -----
        input_shape: shape of the expected input (height, width, depth)
        num_class: number of classes to be discriminated, default 2 (currently
          only allowed value)
        num_level: number of depth levels of network, default 4
        num_filt_init: initial number of filters of the 2D conv (will be 
          doubled at each level of the decoding part), default 32
    Output
    -----
        model: the Keras network model
    """
    assert(num_class == 2), "number of classes different from 2 not yet implemented"
    # disallow more than 6 levels (models get too large, likely with little benefit)
    assert(num_level in range(1, 7)), "number of levels must be between 1 and 6"
    # make sure that size of the images vibes with number of levels
    assert(max(input_shape[:2]) >= 2**num_level), "number of levels too high for images of given size"
    
    # a few parameters that may be made input parameters
    dropout_rate = 0.5
    ki = 'he_normal'

    # building block for the decoding part: convolutions, maxpool and 'horizontal' copy
    def block_decode(input):
        conv1 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same', kernel_initializer=ki)(input)
        conv2 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same', kernel_initializer=ki)(conv1)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        copy = layers.Dropout(dropout_rate)(conv2)
        return pool, copy

    def bottleneck(input):
        conv1 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same', kernel_initializer=ki)(input)
        conv2 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same', kernel_initializer=ki)(conv1)
        return conv2

    # building block for the encoding part: inverse convolution, concatenation, convolutions
    def block_encode(conv_input, copy_input):
        upconv = layers.Conv2DTranspose(num_filt_init * 2**level, (2,2), strides=(2,2), activation='relu', padding='same')(conv_input)
        cat = layers.Concatenate(axis=3)([copy_input, upconv])
        conv1 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same')(cat)
        conv2 = layers.Conv2D(num_filt_init * 2**level, (3,3), activation='relu', padding='same')(conv1)
        return conv2

    # --- initialization
    # store horizontal connections in dict with level as key
    horz_dict = {}
    inputs = layers.Input(shape=input_shape)
    block_layer = inputs
    # --- DECODING PART
    for level in range(num_level):
        block_layer, horz_dict[level] = block_decode(block_layer)
    # --- BOTTLENECK (in terms of num_level, count as an additional level)
    level = num_level
    block_layer = bottleneck(block_layer)
    level += 1
    # --- ENCODING PART
    for level in range(num_level-1, -1, -1):
        block_layer = block_encode(block_layer, horz_dict[level])
    # --- last step: 1x1 convolution & sigmoid activation
    block_layer = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(block_layer)
    # plug togeher
    model = models.Model(inputs=inputs, outputs=block_layer)
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
