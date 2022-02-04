
import tensorflow as tf
from tensorflow.python.framework import ops

_binned_select_knn = tf.load_op_library('binned_select_knn.so')

def _BinnedSelectKnn(K : int, coords,  bin_idx, bin_boundaries, n_bins, bin_width , tf_compatible=False):
    '''
    the op wrapper only
    '''
    
    return _binned_select_knn.BinnedSelectKnn(n_neighbours=K, 
                                              coords=coords,
                                              bin_idx=bin_idx,
                                              bin_boundaries=bin_boundaries,
                                              n_bins=n_bins,
                                              bin_width=bin_width,
                                              tf_compatible=tf_compatible
                                              )



def BinnedSelectKnn(K : int, coords, row_splits, bin_width=tf.constant([.5],dtype='float32'), tf_compatible=False):
    from bin_by_coordinates_op import BinByCoordinates
    
    dbinning,binning,nper, nb = BinByCoordinates(coords, row_splits, bin_width)
    
    sorting = tf.argsort(binning)
    
    scoords = tf.gather_nd( coords, sorting[...,tf.newaxis])
    sbinning = tf.gather_nd( binning, sorting[...,tf.newaxis])
    
    bin_boundaries = tf.concat([row_splits[0:1], nper],axis=0)
    # make it row split like
    bin_boundaries = tf.cumsum(bin_boundaries)
    
    idx,dist = _BinnedSelectKnn(K, scoords,  sbinning, bin_boundaries=bin_boundaries, 
                                n_bins=nb, bin_width=bin_width, tf_compatible=tf_compatible )
    
    
    #sort back dist
    #dist = tf.scatter_nd(sorting[...,tf.newaxis], dist, dist.shape)
    
    return idx, dist



_sknn_grad_op = tf.load_op_library('select_knn_grad.so')
@ops.RegisterGradient("BinnedSelectKnn")
def _BinnedSelectKnnGrad(op, idxgrad, dstgrad):
    
    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]

    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)

    
    #FIXME
    return coord_grad,None,None,None,None
  
