"""
Collection of metrics to be used for evaluating accuracy of road detection.
y_true and y_pred are tensors of dimensions (<batch size>, 512, 512).
Output must be a tensor.
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
import functools

# -----------------------------------------------------------------------------
# This is the central function posted by Christian Skoldt which allows us to 
# wrap any tf.metrics or tf.streaming.metrics method to Keras
# https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras/50527423#50527423
# -----------------------------------------------------------------------------
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def auc_roc(y_true, y_pred, summation_method='careful_interpolation', num_thresholds=400, curve='ROC'):
    # note that y_true and y_pred are flattened - this should not be necessary, 
    # but makes sure that there will be no undocumented computation of auc_roc
    # per image (see note above on the size of these variables)
    return tf.metrics.auc(K.flatten(y_true), K.flatten(y_pred), summation_method=summation_method, num_thresholds=num_thresholds, curve=curve)
    
@as_keras_metric
def auc_pr(y_true, y_pred, summation_method='careful_interpolation', num_thresholds=400, curve='PR'):
    return tf.metrics.auc(K.flatten(y_true), K.flatten(y_pred), summation_method=summation_method, num_thresholds=num_thresholds, curve=curve)

@as_keras_metric
def auc_pr_multiclass(y_true, y_pred, summation_method='careful_interpolation', num_thresholds=400, curve='PR'):
    # set up a weight tensor which is all zeros where the y scores for the 
    # no_road values reside
    shape_4thdim = tf.shape(y_pred)[3]
    zeros_shape = tf.add(tf.shape(y_pred), tf.convert_to_tensor([0, 0, 0, 1-shape_4thdim]))
    ones_shape = tf.add(tf.shape(y_pred), tf.convert_to_tensor([0, 0, 0, -1]))
    weights = tf.concat([tf.zeros(zeros_shape, dtype=tf.float32),
                         tf.ones(ones_shape, dtype=tf.float32)], 3)
    return tf.metrics.auc(K.flatten(y_true), 
                          K.flatten(y_pred), 
                          weights = K.flatten(weights),
                          summation_method=summation_method, 
                          num_thresholds=num_thresholds, 
                          curve=curve)

def test_auc_roc():
    """Run a few simple tests on auroc"""
    # set up a set of simple arrays with the same principal shape and data type 
    # as our image arrays
    y_true = np.zeros([100, 3, 3], dtype=np.int32)
    y_true[:, :, 0] = 1
    y_pred = np.float32(np.random.rand(100, 3, 3))
    y_pred[:, :, 0] += 0.5 
    res = auc_roc(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(K.eval(res))
    assert(K.eval(res) > 0.5)


def iou_binary(y_true, y_pred):
    """
    Returns the Intersection over Union (iou) of binary classifier
    """
    # for now, value of threshold is set arbitrarily here!
    threshold = tf.constant(0.03, dtype=tf.float32)
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
    print(K.eval(res))
    assert(abs(K.eval(res)+1.0) < 1e-6)
    # after the two lines below, y_true and y_pred have one intersecting element,
    # and their union is five, so the expected value is 1/5
    y_true[0, :, 0] = 1.0
    y_pred[0, 0, :] = 0.7
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(K.eval(res))
    assert(abs(K.eval(res)-0.2) < 1e-6)
    # two images
    y_true = np.ones([2, 3, 3], dtype=np.float32)
    y_pred = np.zeros([2, 3, 3], dtype=np.float32)
    y_pred[0,:,:] = 1.0
    # should return 0.5
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(K.eval(res))
    assert(abs(K.eval(res)-0.5) < 1e-6)
    # should return 1.0
    y_pred = np.ones([2, 3, 3], dtype=np.float32)
    res = IoU_binary(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
    print(K.eval(res))
    assert(abs(K.eval(res)-1.0) < 1e-6)

def dummy_metric(y_true, y_pred):
    # this is the place to try out stuff
    #return K.shape(K.flatten(y_pred))
    return K.shape(y_pred)[3]
    # whatever = 9
    # return tf.convert_to_tensor(whatever)
    #return K.eval(K.flatten(y_true))
    #return tf.reduce_max(y_pred)
    #return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[0]

    