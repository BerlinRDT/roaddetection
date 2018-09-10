from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from scipy import signal

from pathlib import Path

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
            new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                             new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
                new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        # img = (img - np.mean(img)) / np.std(img)

        mask = mask / 255
        # mask = (mask - np.mean(mask)) / np.std(mask)

        mask[mask > 0.3] = 1
        mask[mask <= 0.3] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for file in Path(test_path).iterdir():
        img = io.imread(file, as_gray=as_gray)
        img = img / 255
        # img = trans.resize(img, target_size)
        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield (img, file.name)


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, name, flag_multi_class=False, num_class=2):
    # print(npyfile)
    for i, item in enumerate(npyfile):
        io.imsave(os.path.join(save_path, name), item.reshape((512, 512)))


# -------------- below here, functions for feature engineering ----------------
def conv_img(x, conv_matrix, collapse_bands=False):
    """
    Performs 2D convolution on each band of input matrix x, representing an 
    [height, width, band] image, preserving its shape.
    If collapse_bands is True, a single-band average across bands will be returned.
    Each band of the resulting output is standardized (divided by the sd of its elements)
    """
    x_conv = np.empty(x.shape, dtype = np.float32)
    for band_ix in range(x.shape[2]):
        x_conv[:,:,band_ix] = signal.convolve2d(x[:,:,band_ix].astype(np.float32), conv_matrix, boundary='symm', mode='same')
        # subtract band-wise grand average
        x_conv[:,:,band_ix] -= np.mean(x_conv[:,:,band_ix])
        if not collapse_bands:
            # divide by std
            x_conv[:,:,band_ix] /= np.std(x_conv[:,:,band_ix])
    if collapse_bands:
        x_conv = np.mean(x_conv, axis=2)
        x_conv /= np.std(x_conv)
        
    return x_conv

def feature_eng_conv(x, conv_fun, **kwargs):
    """
    Feature engineering on input matrix x, representing an [height, width, band] image
    """
    xf = None
    num_feature = len(conv_fun)
    if num_feature:
        # preallocate
        xf = np.empty(x.shape[:2] + (num_feature,))
        for i, f in enumerate(conv_fun):
            xf[:,:,i] = conv_img(x, f(), collapse_bands=True)
    return xf
            

# matrices to be used for convolution
def conv_matrix_inhibsurround(n=7):
    """
    n by n, positive center, negative surround
    Elements sum to zero
    """
    assert(n>=5)
    m = np.ones((n, n), dtype=np.float32) / -(n**2 - 3**2)
    m[n//2-1:n//2+1, n//2-1:n//2+1] = 1.0/3**2
    return m


def conv_matrix_horizontalbar(n=7):
    """
    n by n, positive center row, negative surround
    Elements sum to zero
    """
    m = np.ones((n, n), dtype=np.float32) / -(n**2 - n)
    m[n//2, :] = 1.0/n
    return m

def conv_matrix_verticalbar(n=7):
    """
    n by n, positive center column, negative surround
    Elements sum to zero
    """
    m = np.ones((n, n), dtype=np.float32) / -(n**2 - n)
    m[:, n//2] = 1.0/n
    return m

def conv_matrix_diag_ullr():
    """
    3 by 3, positive diagonal, negative surround
    Elements sum to zero
    """
    m = np.eye(3, dtype=np.float32) / 3.0
    m[m<=0.0] = -1.0/6.0
    return m

def conv_matrix_diag_llur():
    """
    3 by 3, positive diagonal, negative surround
    Elements sum to zero
    """
    m = np.eye(3, dtype=np.float32) / 3.0
    m[m<=0.0] = -1.0/6.0
    m = np.fliplr(m)
    return m
            
            
            





