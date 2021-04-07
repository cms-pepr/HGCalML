
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Indices MUST be unique in each row.
Only exception are multiple self-references, that can be used as sort of padding.
Alternatively, the index -1 is skipped (non TF conpatible padding)

'''

_accknn_op = tf.load_op_library('accumulate_knn.so')
_accknn_grad_op = tf.load_op_library('accumulate_knn_grad.so')

def AccumulateKnn(distances,  features, indices, mean_and_max=True):
    '''
    
    .Output("out_features: float32")
    .Output("out_max_idxs: int32");
    
    
    Assumes that neighbour indices can be padded with -1, but not mixed, e.g. [1,4,-1,2] needs to be [1,4,2,-1]
    Other than the padding, the indices must be unique
    
    '''
    return _accknn_op.AccumulateKnn(n_moments=0, mean_and_max=mean_and_max, 
                                    distances=distances,  features=features, indices=indices)


@ops.RegisterGradient("AccumulateKnn")
def _AccumulateKnnGrad(op, grad, gradmaxidxs):
    """
      
    """
    
    
    distances  = op.inputs[0]
    features  = op.inputs[1]
    max_feat_indices = op.outputs[1]
    neigh_indices = op.inputs[2]
    
    coord_grad , feat_grad = _accknn_grad_op.AccumulateKnnGrad(grad_from_out_features=grad,
                                                               distances=distances,
                                                               features=features,
                                                               neigh_indices=neigh_indices,
                                                               max_feat_indices=max_feat_indices)
    
    return [coord_grad , feat_grad, None] #no gradient for indices
  
  

  
  
