
import tensorflow as tf
from tensorflow.python.framework import ops

_binned_select_knn = tf.load_op_library('binned_select_knn.so')

def _BinnedSelectKnn(K : int, coords,  bin_idx, dim_bin_idx, bin_boundaries, n_bins, bin_width , tf_compatible=False):
    '''
    the op wrapper only
    '''
    
    return _binned_select_knn.BinnedSelectKnn(n_neighbours=K, 
                                              coords=coords,
                                              bin_idx=bin_idx,
                                              dim_bin_idx=dim_bin_idx,
                                              bin_boundaries=bin_boundaries,
                                              n_bins=n_bins,
                                              bin_width=bin_width,
                                              tf_compatible=tf_compatible
                                              )


def BinnedSelectKnn(K : int, coords, row_splits, n_bins=None, max_bin_dims=3, tf_compatible=False):
    from bin_by_coordinates_op import BinByCoordinates
    from index_replacer_op import IndexReplacer
    
    
    # the following number of bins seems a good~ish estimate for good performance
    # for homogenous point distributions but should be subject to more tests
    if n_bins is None:
        n_bins = tf.math.pow(tf.cast(row_splits[1],dtype='float32')/(K/32),1/max_bin_dims)
        n_bins = tf.cast(n_bins,dtype='int32')
        n_bins = tf.where(n_bins<5,5,n_bins)
        
    bin_coords = coords
    if bin_coords.shape[-1]>max_bin_dims:
        bin_coords = bin_coords[:,:max_bin_dims]
    
    dbinning,binning, nb, bin_width, nper = BinByCoordinates(bin_coords, row_splits, n_bins=n_bins)
    
    #if this becomes a bottleneck one could play tricks since nper and bin numbers are predefined
    sorting = tf.argsort(binning)
    
    scoords = tf.gather_nd( coords, sorting[...,tf.newaxis])
    sbinning = tf.gather_nd( binning, sorting[...,tf.newaxis])
    sdbinning = tf.gather_nd( dbinning, sorting[...,tf.newaxis])
    
    bin_boundaries = tf.concat([row_splits[0:1], nper],axis=0)
    # make it row split like
    bin_boundaries = tf.cumsum(bin_boundaries)
    
    idx,dist = _BinnedSelectKnn(K, scoords,  sbinning, sdbinning, bin_boundaries=bin_boundaries, 
                                n_bins=nb, bin_width=bin_width, tf_compatible=tf_compatible )
    #sort back 
    idx = IndexReplacer(idx,sorting)
    dist = tf.scatter_nd(sorting[...,tf.newaxis], dist, dist.shape)
    idx = tf.scatter_nd(sorting[...,tf.newaxis], idx, idx.shape)
    
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
  
