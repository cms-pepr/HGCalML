
import tensorflow as tf
from tensorflow.python.framework import ops
import globals as gl
from oc_helper_ops import SelectWithDefault

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


def BinnedSelectKnn(K : int, coords, row_splits, n_bins=None, max_bin_dims=3, tf_compatible=False, max_radius=None):
    '''
    max_radius is a dummy for now to make it a drop-in replacement
    '''
    from bin_by_coordinates_op import BinByCoordinates
    from index_replacer_op import IndexReplacer
    
    # the following number of bins seems a good~ish estimate for good performance
    # for homogenous point distributions but should be subject to more tests
    elems_per_rs = 1
    if row_splits.shape[0] is not None:
        elems_per_rs = row_splits[1]
        #do checks
        tf.assert_equal(row_splits[-1],coords.shape[0])
    
    if n_bins is None:
        n_bins = tf.math.pow(tf.cast(elems_per_rs,dtype='float32')/(K/32),1/max_bin_dims)
        n_bins = tf.cast(n_bins,dtype='int32')
        n_bins = tf.where(n_bins<5,5,n_bins)
        n_bins = tf.where(n_bins>20,20,n_bins)#just a guess
        
    bin_coords = coords
    if bin_coords.shape[-1]>max_bin_dims:
        bin_coords = bin_coords[:,:max_bin_dims]
    
    dbinning,binning, nb, bin_width, nper = BinByCoordinates(bin_coords, row_splits, n_bins=n_bins)
    
    #if this becomes a bottleneck one could play tricks since nper and bin numbers are predefined
    sorting = tf.argsort(binning)
    
    scoords = tf.gather_nd( coords, sorting[...,tf.newaxis])
    sbinning = tf.gather_nd( binning, sorting[...,tf.newaxis])
    sdbinning = tf.gather_nd( dbinning, sorting[...,tf.newaxis])
    
    #add a leading 0
    bin_boundaries = tf.concat([tf.zeros([1],dtype='int32'), nper],axis=0) #row_splits[0:1]
    # make it row split like
    bin_boundaries = tf.cumsum(bin_boundaries)
    
    idx,dist = _BinnedSelectKnn(K, scoords,  sbinning, sdbinning, bin_boundaries=bin_boundaries, 
                                n_bins=nb, bin_width=bin_width, tf_compatible=tf_compatible )
    
    if row_splits.shape[0] is None:
        return idx, dist
    #sort back 
    idx = IndexReplacer(idx,sorting)
    dist = tf.scatter_nd(sorting[...,tf.newaxis], dist, dist.shape)
    idx = tf.scatter_nd(sorting[...,tf.newaxis], idx, idx.shape)
    
    if not gl.knn_ops_use_tf_gradients:
        return idx, dist
        
    ncoords = SelectWithDefault(idx, coords, 0.)
    distsq = (ncoords[:,0:1,:]-ncoords)**2
    distsq = tf.reduce_sum(distsq,axis=2)
    distsq = tf.where(idx<0, 0., distsq)
    return idx, distsq


_sknn_grad_op = tf.load_op_library('select_knn_grad.so')
@ops.RegisterGradient("BinnedSelectKnn")
def _BinnedSelectKnnGrad(op, idxgrad, dstgrad):
    
    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]

    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)

    
    return coord_grad,None,None,None,None,None
  
