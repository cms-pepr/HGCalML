
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Indices MUST be unique in each row.
Only exception are multiple self-references, that can be used as sort of padding.
Alternatively, the index -1 is skipped (non TF conpatible padding)

'''

_accknn_op = tf.load_op_library('accumulate_knn.so')
_accknn_grad_op = tf.load_op_library('accumulate_knn_grad.so')

def AccumulateKnn(distances,  features, indices, n_moments=0):
    '''
    
    .Output("out_features: float32")
    .Output("out_max_idxs: int32");
    '''
    if n_moments > 0:
        raise ValueError("AccumulateKnn: n_moments not implemented")
    return _accknn_op.AccumulateKnn(n_moments=n_moments, distances=distances,  features=features, indices=indices)


@ops.RegisterGradient("AccumulateKnn")
def _AccumulateKnnGrad(op, grad, gradmaxidxs):
  """
    
  """
  
  #maybe revert this back to TF.. this one is super slow
  
  
  
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
  
  

_accknn_nd_op = tf.load_op_library('accumulate_knn_nd.so')
_accknn_nd_grad_op = tf.load_op_library('accumulate_knn_nd_grad.so')

def AccumulateKnnNd(coords,  features, indices, n_moments=0):
    if n_moments > 3 or n_moments<0:
        raise ValueError("AccumulateKnnNd: n_moments must be between 0 and 3 (including)")
    return _accknn_nd_op.AccumulateKnnNd(n_moments=n_moments, coords=coords,  features=features, indices=indices)


@ops.RegisterGradient("AccumulateKnnNd")
def _AccumulateKnnNdGrad(op, grad, gradmaxidxs, gradfeatsum):
  """
    
  """
  
  n_coords = op.inputs[0].shape[1]
  n_features = op.inputs[1].shape[1]
  n_neigh = op.inputs[2].shape[1]
  
  coords  = op.inputs[0]
  features  = op.inputs[1]
  max_feat_indices = op.outputs[1]
  orig_op_out_feat = op.outputs[0]
  feat_sum = op.outputs[2]
  neigh_indices = op.inputs[2]
  
  coord_grad , feat_grad = _accknn_nd_grad_op.AccumulateKnnNdGrad(grad_from_out_features=grad,
                                                                  grad_from_sum_features=gradfeatsum,
                                                             coords=coords,
                                                             features=features,
                                                             neigh_indices=neigh_indices,
                                                             max_feat_indices=max_feat_indices,
                                                             orig_op_out_feat=orig_op_out_feat,
                                                             orig_up_out_feat_sum = feat_sum)

  return [coord_grad , feat_grad, None] #no gradient for indices
  
  
