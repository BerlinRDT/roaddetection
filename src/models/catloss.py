import numpy as np
from keras import backend as K

def categoricalCrossentropy(y_noisy, y_pred, the0, the1, the2, the3):
    '''
    Calculate the class-weighted categorical cross-entropy for the given
    predicted and true sets.

    y_true [in] The truth set to test against. This is a Tensor with a last
                dimension that contains a set of 1-of-N selections.
    y_pred [in] The predicted set to test against. This is a Tensor with a last
                dimension that contains a set of 1-of-N selections.
    returns     A Tensor function that will calculate the weighted categorical
                cross-entropy on the inputs.
    '''
    P = np.array([[1, the0, the0], 
                  [the1,  1, (the0+the1)*.5], 
                  [the1, (the0+the1)*.5, 1]])
    P = np.array([[1, 0, 0], 
                  [0, 1, the0], 
                  [the1, 0, 1]])
    if P is not None:
        # Wrap the loss weights in a tensor object.
        p = np.linalg.inv(P)
        p =  p / (np.sum(p))
        #assert (np.isclose(np.sum(p), 1) == True)
        theWeights =  K.constant(p, shape=p.shape)

        y_true = K.dot(y_noisy,theWeights)

    return  K.categorical_crossentropy(y_true, y_pred)

def noisy_loss(the0, the1, the2, the3):    
    def loss(y_true, y_pred):
        return categoricalCrossentropy(y_true, y_pred,  the0, the1, the2, the3)
    return loss