
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')
_ld_op = tf.load_op_library('local_distance.so')

def LocalDistance(coords,  neighbour_idxs):
    
    '''
    .Input("coordinates: float32")
    .Input("neighbour_idxs: int32")
    .Output("distances: float32");
    '''
    return _ld_op.LocalDistance(coordinates=coords,neighbour_idxs=neighbour_idxs)
    



_sknn_grad_op = tf.load_op_library('select_knn_grad.so')

@ops.RegisterGradient("LocalDistance")
def _LocalDistanceGrad(op, dstgrad):
    #
    # uses the selectKnn gradient (which is also a distance gradient)
    
    
    coords = op.inputs[0]
    indices = op.inputs[1]
    distances = op.outputs[0]
    
    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, 
                                             indices=indices, 
                                             distances=distances, 
                                             coordinates=coords)
    
    return coord_grad, None #no grad for row splits and masking values
    
    
    
    
    
    
    
    
    
    
  
