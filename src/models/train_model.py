#!/usr/bin/env python
# coding: utf-8

"""
 Run network model.
 This is essentially code ported from notebooks/networks/run_network.ipynb,
 the notebook that had been used for training models (U-nets). Modifications
 are made mostly to the metric, which had proven to be problematic.
 Very obviously this will only be a transient stage, to be followed by
 implementing config files allowing users to specify settings.
 """

from src.models.data import trainGenerator
from src.models.network_models import unet_flex
from src.models.metrics_img import IoU_binary, precision, recall, f1_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.optimizers import Adam
import os
import logging


def get_logger():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    formatter = logging.Formatter(log_fmt)
    fh = logging.FileHandler('logs/unet-2.log')
    fh.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(fh)
    return logger


def plot_history(history):
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    # plt.close()

    plt.plot(history["f1_score"], label="f1_score")
    plt.plot(history["val_f1_score"], label="val_f1_score")
    plt.legend()
    plt.show()
    # plt.close()


def main():
    # Define directories
    dirs = []
    data_dir = "data"
    model_dir = "models/UNet"

    train_dir = os.path.join(data_dir, "train")
    dirs.append(train_dir)

    train_partial_dir = os.path.join(data_dir, "train_partial")
    dirs.append(train_partial_dir)

    validation_dir = os.path.join(data_dir, "validate")
    dirs.append(validation_dir)

    test_dir = os.path.join(data_dir, "test")
    dirs.append(test_dir)


    # User settings
    # ------------- image characteristics and augmentation -----------------------------
    # size of tiles
    target_size = (512, 512)
    # input arguments to Keras' ImageDataGenerator
    data_gen_args = dict(
        data_format="channels_last",
        horizontal_flip=True,
        vertical_flip=True)
    # directory into which to place *training* images from ImageDataGenerator for inspection;
    # default should be None because this slows things down
    imgdatagen_dir = None
    #imgdatagen_dir = data_dir + '/imgdatagenerator'

    #--------------- network weights ----------------------------------------------------
    # path to & filename of pre-trained model to use - set to None if you want to start from scratch
    pretrained_model_fn = model_dir + '/models_unet_borneo_and_harz_05_09_16_22.hdf5'
    pretrained_model_fn = model_dir + '/unet_test.hdf5'
    #pretrained_model_fn = None

    # path to & filename of model to save
    trained_model_fn = model_dir + '/unet_test_full.hdf5'

    #--------------- training details / hyperparameters -----------------------------------
    # batch size
    batch_size = 4
    # steps per epoch, should correspond to [number of training images] / batch size
    steps_per_epoch = 600 // batch_size
    # number of epochs
    epochs = 50
    # number of steps on validation set
    validation_steps = 60
    # self-explanatory variables:
    optimizer = Adam(lr=1e-4)
    loss = 'binary_crossentropy'
    loss_weights = None
    metrics = [IoU_binary, precision, recall, f1_score]

    # Count image tiles in train/validation/test directories
    for directory in dirs:
        for file_type in ["sat", "map", "sat_rgb"]:
            target = os.path.join(directory, file_type)
            print(target, ":", len(os.listdir(target)))


    # Set up ImageDataGenerators for training and validation sets
    train_gen = trainGenerator(batch_size, data_dir + '/train','sat','map',
                               data_gen_args, save_to_dir = imgdatagen_dir, image_color_mode="rgba", target_size=target_size)

    validation_gen = trainGenerator(batch_size, data_dir + '/validate','sat','map',
                                    data_gen_args, save_to_dir = None, image_color_mode="rgba", target_size=target_size)


    # Define model
    model = unet_flex(num_filt_init=64, num_level=3)

    # compile
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    # show summary
    model.summary()

    # possibly load weights
    if pretrained_model_fn:
        model.load_weights(pretrained_model_fn)

    # define callbacks (including checkpoints)
    model_checkpoint = ModelCheckpoint(trained_model_fn, monitor='loss',verbose=1, save_best_only=True)
    # - stop training if loss doesn't improve for 5 consecutive epochs
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None)
    # - logging
    logger = get_logger()
    logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: logger.info({'epoch': epoch, 'logs': logs})
    )

    # Run training
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  callbacks=[model_checkpoint, early_stop, logging_callback],
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps
                                 )

    plot_history(history.history)


if __name__ == '__main__':
    main()