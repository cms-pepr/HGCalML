
import tensorflow as tf
from tensorflow.python.framework import ops
from accknn_op import AccumulateKnn

    

def NeighbourCovariance(coordinates, distsq, features, n_idxs):
    '''
    expands to V x F x C**2, but not in the neighbour dimension
    '''
    
    features = tf.nn.relu(features)#sure they're>=0
    #create feature and distance weighted coordinate means
    featcoordinates = tf.expand_dims(coordinates,axis=1) * tf.expand_dims(features,axis=2) # V x F x C
    featcoordinates = tf.reshape(featcoordinates, [-1, features.shape[1] * coordinates.shape[1]]) # V x F*C
    
    fcdsum = AccumulateKnn(distsq,  featcoordinates, n_idxs, mean_and_max=False)[0] * distsq.shape[1] # only sum, V x F*C
    fcdsum = tf.reshape(fcdsum, [-1, features.shape[1], coordinates.shape[1]])
    
    fdsum = AccumulateKnn(distsq,  features, n_idxs, mean_and_max=False)[0] * distsq.shape[1] # V x F
    fdsum += 1e-9
    fdsum = tf.expand_dims(fdsum, axis=2)
    
    fcdmean = tf.math.divide_no_nan(fcdsum, fdsum) # V x F x C
    
    centeredcoords = tf.expand_dims(coordinates,axis=1) - fcdmean # V x F x C
    
    xi = tf.expand_dims(centeredcoords, axis=3) * tf.expand_dims(centeredcoords, axis=2) # V x F x C^T x C
    xi = tf.reshape(xi, [-1, features.shape[1] * coordinates.shape[1]**2])
    
    distweightcov = AccumulateKnn(distsq,  xi, n_idxs, mean_and_max=False)[0] * distsq.shape[1]
    distweightcov = tf.reshape(distweightcov, [-1, features.shape[1], coordinates.shape[1]**2])
    distweightcov /= fdsum
    
    return distweightcov, fcdmean  # V x F x C**2, V x F x C
        
    
    