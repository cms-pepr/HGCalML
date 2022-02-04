
import tensorflow as tf
from tensorflow.python.framework import ops

_bin_by_coordinates = tf.load_op_library('bin_by_coordinates.so')

'''
 .Input("coordinates: float")
    .Input("row_splits: int32")
    .Input("bin_width: float")
    .Input("nbins: int32")//same in all dimensions
    .Output("output: int32"); 
'''

def BinByCoordinates(coordinates, row_splits, bin_width, restrict_nbins=-1, calc_n_per_bin=True):
    '''
    
    bin_width: scalar input (can be tensor)
    
    bin width will be a lower bound, can be larger in some dimensions
    
    nbins will be a singleton. bin width (dim: N_coordinates) will be 
    adjusted such that the number of bins will be the same in all dimensions.
    
    output:
    - bin indices (flat)
    - number of bins (same for all dimensions)
    - bin width (different in different dimensions, adjusted)
    
    '''
    
    #calculate
    dmax_coords = tf.reduce_max(coordinates,axis=0)
    dmin_coords = tf.reduce_min(coordinates,axis=0)
    
    nbins = (dmax_coords - dmin_coords) / bin_width  #this will be max
    
    if restrict_nbins>0:
        nbins = tf.where(nbins>restrict_nbins, restrict_nbins, nbins)
        bin_width = tf.reduce_max((dmax_coords - dmin_coords) / nbins)[...,tf.newaxis]
        nbins = (dmax_coords - dmin_coords) / bin_width
    
    nbins = tf.cast(nbins, dtype='int32')+1 #add one for safety
    
    coordinates -= tf.expand_dims(dmin_coords,axis=0)
    
    binass,flatbinass,nperbin = _bin_by_coordinates.BinByCoordinates(coordinates=coordinates, 
                                                row_splits=row_splits, 
                                                bin_width=bin_width, nbins=nbins,
                                                calc_n_per_bin=calc_n_per_bin)
    #print(nbins)
    #print(binass)
    #sanity checks
    with tf.control_dependencies([tf.assert_less(binass, 
                                                 tf.expand_dims(
                                                     tf.concat([tf.constant([row_splits.shape[0]-1]) ,nbins],axis=0),
                                                     axis=0))]):
        
        if calc_n_per_bin:                                            
            return binass,flatbinass,nperbin,nbins,bin_width
        else:
            return binass,flatbinass,nbins,bin_width

@ops.RegisterGradient("BinByCoordinates")
def _BinByCoordinatesGrad(op, idxout_grad, flatidxgrad,npbingrad):
    
    return None, None, None, None
  
