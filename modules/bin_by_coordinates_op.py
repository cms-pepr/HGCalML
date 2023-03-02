
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

def BinByCoordinates(coordinates, row_splits, bin_width=None, n_bins=None, calc_n_per_bin=True, pre_normalized=False, name=""):
    '''
    
    Assign bins to coordinates

    @type coordinates: tf.Tensor(float32)
    @param coordinates: coordinates per input point

    @type row_splits: tf.Tensor(int)
    @param row_splits: row splits following tf.ragged convention

    @type bin_width: tf.Tensor(float32) / None
    @param bin_width: will be the same for all dimensions (either bin_width or n_bins must be specified)
    
    @type n_bins: tf.Tensor(int) / None
    @param n_bins: this is the maximum number of bins in any dimension (either bin_width or n_bins must be specified)

    @type calc_n_per_bin: bool
    @param calc_n_per_bin: calculates the number of points per bin and returns it
    
    
    
    output:
    - bin indices (dim = [rs] + dim(coordinates)). The first index constitues the row split index
    - bin indices (the above) flattened
    - number of bins used per dimension (dim = dim(coordinates))
    - bin width used (dim = 1)
    - (opt) number of points per bin (dim = 1)
    
    '''
    
    tf.debugging.check_numerics(coordinates,"BinByCoordinates: input coordinates "+name)
    #calculate
    #orig_coordinates = coordinates
    if not pre_normalized:
        min_coords = tf.reduce_min(coordinates,axis=0,keepdims=True)
        #min_coords = tf.where(tf.math.is_finite(min_coords), min_coords, 0.)
        coordinates -= min_coords
    dmax_coords = tf.reduce_max(coordinates,axis=0) 
    dmax_coords = tf.where(tf.reduce_min(coordinates,axis=0) == dmax_coords, dmax_coords+1., dmax_coords)  + 1e-3
    
    dmax_coords = tf.where(tf.math.is_finite(dmax_coords), dmax_coords, 1.)
    
    with tf.control_dependencies([tf.assert_greater(dmax_coords,0.),
                                  tf.debugging.check_numerics(coordinates,"BinByCoordinates: adjusted coordinates "+name)]):
    
        if bin_width is None:
            assert n_bins is not None
            bin_width = (dmax_coords) / tf.cast(n_bins, dtype='float32')
            n_bins = None #re-calc in dimensions
            bin_width = tf.reduce_max(bin_width)[...,tf.newaxis]#just add a '1' dimension
            
        if n_bins is None:
            assert bin_width is not None
            n_bins = (dmax_coords) / bin_width  
            n_bins += 1.
            n_bins = tf.cast(n_bins, dtype='int32')
        
    with tf.control_dependencies([tf.assert_greater(n_bins,0), 
                                  tf.assert_greater(bin_width,0.)]):
        
        binass,flatbinass,nperbin = _bin_by_coordinates.BinByCoordinates(coordinates=coordinates, 
                                                row_splits=row_splits, 
                                                bin_width=bin_width, nbins=n_bins,
                                                calc_n_per_bin=calc_n_per_bin)
    #sanity checks
    #with tf.control_dependencies([tf.assert_less(binass, 
    #                                             tf.expand_dims(
    #                                                 tf.concat([tf.constant([row_splits.shape[0]-1]) ,n_bins],axis=0),
    #                                                 axis=0))]):
        
    if calc_n_per_bin:                                            
        return binass,flatbinass,n_bins,bin_width,nperbin
    else:
        return binass,flatbinass,n_bins,bin_width

@ops.RegisterGradient("BinByCoordinates")
def _BinByCoordinatesGrad(op, idxout_grad, flatidxgrad,npbingrad):
    
    return None, None, None, None
  
