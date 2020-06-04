
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')

def SelectKnn(K : int, coords,  row_splits, tf_compatible=True, max_radius=-1.):
    '''
    returns indices and distances**2 , gradient for distances is implemented!
    '''
    
    return _sknn_op.SelectKnn(n_neighbours=K, tf_compatible=tf_compatible, max_radius=max_radius,
                                 coords=coords, row_splits=row_splits)
    



_sknn_grad_op = tf.load_op_library('select_knn_grad.so')

@ops.RegisterGradient("SelectKnn")
def _SelectKnnGrad(op, gradidx, dstgrad):
    
    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]
    
    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)
    
    return coord_grad, None #no grad for row splits
    
    
    
    
    
    
    
    
    
    
  
