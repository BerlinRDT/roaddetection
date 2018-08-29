"""
Collection of metrics to be used for evaluating accuracy of road detection.
y_true and y_pred are tensors of dimensions (<batch size>, 512, 512).
Output must be a tensor.
Ergo, for reasons of efficiency, all operations had best be done with tensorflow
methods so that we don't have to convert tensors to numpy arrays back and forth.
"""

import numpy as np
import keras.backend as K
import tensorflow as tf

def IoU_binary(y_true, y_pred):
    """
    Returns the Intersection over Union (IoU) of binary classifier
    """
    # for now, value of threshold is set arbitrarily here!
    threshold = tf.constant(0.25, dtype=tf.float32)
    zero_int = tf.constant(0, dtype=tf.int64)
    nometric_val = tf.constant(-1.0, dtype=tf.float32)
    # tensor of booleans 
    y_pred_label = tf.greater_equal(y_pred, threshold)
    # convert y_true to booleans
    y_true = tf.greater_equal(y_true, threshold)
    # intersection
    inters = tf.logical_and(y_pred_label, y_true)
    # union
    union = tf.logical_or(y_pred_label, y_true)
    # count True instances in both
    inters_sum = tf.count_nonzero(inters)
    union_sum = tf.count_nonzero(union)
    # if the union is zero we have no metric, return -1, their ratio otherwise
    return tf.cond(tf.equal(union_sum, zero_int),
                   lambda: tf.multiply(nometric_val, 1),
                   lambda: tf.cast(tf.divide(inters_sum, union_sum), dtype=tf.float32))
    
def test_IoU_binary():
    """Run a few simple tests on IoU_binary"""
    # set up a set of simple arrays with the same principal shape and data type 
    # as our image arrays: one 3 by 3 image each
    y_true = np.zeros([1, 3, 3], dtype=np.float32)
    y_pred = np.zeros([1, 3, 3], dtype=np.float32)
    # all zeros: should return -1
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(res.eval())
    assert(abs(res.eval()+1.0) < 1e-6)
    # after the two lines below, y_true and y_pred have one intersecting element,
    # and their union is five, so the expected value is 1/5
    y_true[0, :, 0] = 1.0
    y_pred[0, 0, :] = 0.7
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(res.eval())
    assert(abs(res.eval()-0.2) < 1e-6)
    # two images
    y_true = np.ones([2, 3, 3], dtype=np.float32)
    y_pred = np.zeros([2, 3, 3], dtype=np.float32)
    y_pred[0,:,:] = 1.0
    # should return 0.5
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(res.eval())
    assert(abs(res.eval()-0.5) < 1e-6)
    # should return 1.0
    y_pred = np.ones([2, 3, 3], dtype=np.float32)
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(res.eval())
    assert(abs(res.eval()-1.0) < 1e-6)
    
    

def dummy_metric(y_true, y_pred):
    # this is the place to try out stuff
    # return K.shape(K.flatten(y_pred))
    whatever = 9
    return tf.convert_to_tensor(whatever)
    # return K.sum(K.flatten(y_true))
    