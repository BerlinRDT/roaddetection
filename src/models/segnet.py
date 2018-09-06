from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, MaxPooling2D, UpSampling2D, Reshape
from keras.models import *


def segnet(input_size=(512, 512, 4)):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    inputs = Input(input_size)

    # encoder

    padding1 = ZeroPadding2D(padding=(pad, pad))(inputs)
    conv1 = Conv2D(filter_size, kernel, activation='relu', border_mode='valid')(padding1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm1)

    padding2 = ZeroPadding2D(padding=(pad, pad))(pool1)
    conv2 = Conv2D(256, kernel, activation='relu', border_mode='valid')(padding2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm2)

    padding3 = ZeroPadding2D(padding=(pad, pad))(pool2)
    conv3 = Conv2D(512, kernel, activation='relu', border_mode='valid')(padding3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm3)

    # decoder

    padding4 = ZeroPadding2D(padding=(pad, pad))(pool3)
    conv4 = Conv2D(512, kernel, border_mode='valid')(padding4)
    norm4 = BatchNormalization()(conv4)

    upsampling5 = UpSampling2D(size=(pool_size, pool_size))(norm4)
    padding5 = ZeroPadding2D(padding=(pad, pad))(upsampling5)
    conv5 = Conv2D(256, kernel, border_mode='valid')(padding5)
    norm5 = BatchNormalization()(conv5)

    upsampling6 = UpSampling2D(size=(pool_size, pool_size))(norm5)
    padding6 = ZeroPadding2D(padding=(pad, pad))(upsampling6)
    conv6 = Conv2D(128, kernel, border_mode='valid')(padding6)
    norm6 = BatchNormalization()(conv6)

    upsampling7 = UpSampling2D(size=(pool_size, pool_size))(norm6)
    padding7 = ZeroPadding2D(padding=(pad, pad))(upsampling7)
    conv7 = Conv2D(filter_size, kernel, border_mode='valid')(padding7)
    norm7 = BatchNormalization()(conv7)


    conv8 = Conv2D(1, 1, activation='sigmoid')(norm7)

    model = Model(input=inputs, output=conv8)

    return model
