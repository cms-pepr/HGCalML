
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

def BinByCoordinates(coordinates, row_splits, bin_width):
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
    
    #calculate bin width here
    
    dmax_coords = tf.reduce_max(coordinates,axis=0)
    dmin_coords = tf.reduce_min(coordinates,axis=0)
    
    nbins = (dmax_coords - dmin_coords) / bin_width  #this will be max
    nbins = tf.cast(nbins, dtype='int32')+1 #add one for safety
    
    
    return _bin_by_coordinates.BinByCoordinates(coordinates=coordinates, 
                                                row_splits=row_splits, 
                                                bin_width=bin_width, nbins=nbins), nbins, bin_width

@ops.RegisterGradient("BinByCoordinates")
def _BinByCoordinatesGrad(op, idxout_grad):
    
    return None, None, None, None
  
