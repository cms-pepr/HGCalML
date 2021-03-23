
import tensorflow as tf
from tensorflow.python.framework import ops



_ncov = tf.load_op_library('neighbour_covariance.so')

def NeighbourCovariance(coordinates, features, n_idxs):
    '''
    .Input("coordinates: float32")
    .Input("features: float32")
    .Input("n_idxs: int32")
    .Output("covariance: float32") Vout x F x C(C+1)/2
    .Output("means float32"); Vout x F x C

    '''
    
    #coordinates = tf.debugging.check_numerics(coordinates,'coordinates has nans')
    #features = tf.debugging.check_numerics(features,'features have nans')
    
    covariance, means = _ncov.NeighbourCovariance(coordinates=coordinates, 
                                                  features=features, 
                                                  n_idxs=n_idxs)
    
    #covariance = tf.debugging.check_numerics(covariance,'covariance has nans')
    #means = tf.debugging.check_numerics(means,'means have nans')
    return covariance, means


@ops.RegisterGradient("NeighbourCovariance")
def _NeighbourCovarianceGrad(op, covariancegrad, meansgrad):
    
    return None, None, None #no grad for row splits and masking values
    
    
    
    
    
    