
import tensorflow as tf
from tensorflow.python.framework import ops
from accknn_op import AccumulateKnn

    

def NeighbourCovariance(coordinates, distsq, features, n_idxs):
    '''
    expands to V x F x C**2, but not in the neighbour dimension
    
    Feed features without activation!
    
    '''
        
    features = tf.nn.sigmoid(features) #make sure they're in a good range
    
    nF = features.shape[1]
    nC = coordinates.shape[1]
    nKf = tf.cast(distsq.shape[1],dtype='float32')
    
    #calc mean of features over all neighbours (1/K factor too much)
    sum_F = AccumulateKnn(distsq,  features, n_idxs, mean_and_max=False)[0] * nKf
    #not gonna work like this
    
    n_fweight = tf.math.divide_no_nan(features, sum_F)#include distance weighting norm
    
    #build feature-weighted coordinates: V x 1 x C * V x F x 1 
    FC = tf.expand_dims(coordinates,axis=1) * tf.expand_dims(n_fweight,axis=2)
    #reshape to V x F*C
    FC = tf.reshape(FC, [-1, nF*nC])
    #sum over neighbours (factor 1/K too much)
    
    sum_FC = AccumulateKnn(distsq,  FC, n_idxs, mean_and_max=False)[0] * nKf
    #reshape back to V x F x C
    mean_C = tf.reshape(sum_FC, [-1, nF, nC])
    
    #now we have centred coordinates: V x F x C
    centered_C = tf.expand_dims(coordinates,axis=1) - mean_C
    
    #build covariance input: V x F x C x 1  *  V x F x 1 x C
    cov = tf.expand_dims(centered_C, axis=3) * tf.expand_dims(centered_C, axis=2)
    # reshape to something useful
    cov = tf.reshape(cov, [-1, nF,nC**2])
    cov *= tf.expand_dims(n_fweight, axis=2) #add feature weights
    cov = tf.reshape(cov, [-1, nF*nC**2])
    #sum over neighbours
    cov = AccumulateKnn(distsq,  cov, n_idxs, mean_and_max=False)[0] * nKf 
    #reshape back
    cov = tf.reshape(cov, [-1, nF, nC**2])
    
    #debug checks
    cov    = tf.debugging.check_numerics(cov, "cov has NaNs\n")
    mean_C    = tf.debugging.check_numerics(mean_C, "mean_C has NaNs\n")
    
    
    return cov, mean_C







    