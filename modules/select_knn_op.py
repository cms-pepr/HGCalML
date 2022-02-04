
import tensorflow as tf
from tensorflow.python.framework import ops
import globals as gl
from oc_helper_ops import SelectWithDefault

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')

def SelectKnn(K : int, coords,  row_splits, masking_values=None, threshold=0.5, tf_compatible=False, max_radius=-1.,
              mask_mode='none', mask_logic='xor'):
    '''
    returns indices and distances**2 , gradient for distances is implemented!
    
    new: mask (switch):
    masked:
      0) none = no masking
      1) acc  = get to have neighbours
      2) scat = get to be neighbours
      
      10) xor: exclusive (one xor the other) -> exchange between collections, direction given by 1 and 2
      20) and: selected  (one and the other) -> pooling
      
    no gradient for the mask!
    
    '''
    assert mask_mode=='none' or mask_mode=='acc' or  mask_mode=='scat'
    assert mask_mode=='none' or mask_logic=='xor' or mask_logic=='and' 
    
    if masking_values is None:
        assert mask_mode=='none'
        masking_values = tf.zeros_like(coords[:,0:1])
        
   
        
    mask = tf.zeros_like(masking_values, dtype='int32')
    mask = tf.where(masking_values>threshold, mask+1, mask)
    
    #print('mask',mask)
    
    op_mask_mode = 0

        
    if mask_logic=='xor':
        op_mask_mode=10
    elif mask_logic=='and':
        op_mask_mode=20
        
    if mask_mode=='acc':
        op_mask_mode+=1
    elif mask_mode=='scat':
        op_mask_mode+=2
        
        
    '''
      0) none = no masking
      1) acc  = get to have neighbours
      2) scat = get to be neighbours
      
      
      10) xor: exclusive (one xor the other) -> exchange between collections, direction given by 1 and 2
      20) and: selected  (one and the other) -> pooling (scat and acc don't matter)
    '''
    
    
    idx,distsq = _sknn_op.SelectKnn(n_neighbours=K, tf_compatible=tf_compatible, max_radius=max_radius,
                                 coords=coords, row_splits=row_splits, mask=mask, mask_mode=op_mask_mode)
    
    #safe guards
    with tf.control_dependencies([
        tf.assert_equal(tf.range(tf.shape(idx)[0]), idx[:,0]),
        tf.assert_less(idx, row_splits[-1]),
        tf.assert_less(-2, idx)
        ]):
    
    
        if not gl.knn_ops_use_tf_gradients:
            return idx, distsq
        
        ncoords = SelectWithDefault(idx, coords, 0.)
        distsq = (ncoords[:,0:1,:]-ncoords)**2
        distsq = tf.reduce_sum(distsq,axis=2)
        distsq = tf.where(idx<0, 0., distsq)
        return idx, distsq



_sknn_grad_op = tf.load_op_library('select_knn_grad.so')

@ops.RegisterGradient("SelectKnn")
def _SelectKnnGrad(op, gradidx, dstgrad):
    
    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]
    
    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)
    
    return coord_grad, None, None #no grad for row splits and masking values
    
    
    
    
    
    
    
    
    
    
  
