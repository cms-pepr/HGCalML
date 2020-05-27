
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_accknn_op = tf.load_op_library('accumulate_knn.so')
_accknn_grad_op = tf.load_op_library('accumulate_knn_grad.so')

def AccumulateKnn(coords,  features, indices, n_moments):
    return _accknn_op.AccumulateKnn(n_moments=n_moments, coords=coords,  features=features, indices=indices)


@ops.RegisterGradient("AccumulateKnn")
def _AccumulateKnnGrad(op, grad, gradmaxidxs):
  """
    
  """
  
  #maybe revert this back to TF.. this one is super slow
  
  
  n_coords = op.inputs[0].shape[1]
  n_features = op.inputs[1].shape[1]
  n_neigh = op.inputs[2].shape[1]
  
  coords  = op.inputs[0]
  features  = op.inputs[1]
  max_feat_indices = op.outputs[1]
  neigh_indices = op.inputs[2]

  coord_grad , feat_grad = _accknn_grad_op.AccumulateKnnGrad(grad_from_out_features=grad,
                                                             coords=coords,
                                                             features=features,
                                                             neigh_indices=neigh_indices,
                                                             max_feat_indices=max_feat_indices)

  return [coord_grad , feat_grad, None] #no gradient for indices
  
  

_accknn_nd_op = tf.load_op_library('accumulate_knn_nd.so')
_accknn_nd_grad_op = tf.load_op_library('accumulate_knn_nd_grad.so')

def AccumulateKnnNd(coords,  features, indices, n_moments=0):
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
  neigh_indices = op.inputs[2]
  
  coord_grad , feat_grad = _accknn_nd_grad_op.AccumulateKnnNdGrad(grad_from_out_features=grad,
                                                                  grad_from_sum_features=gradfeatsum,
                                                             coords=coords,
                                                             features=features,
                                                             neigh_indices=neigh_indices,
                                                             max_feat_indices=max_feat_indices,
                                                             orig_op_out_feat=orig_op_out_feat)

  return [coord_grad , feat_grad, None] #no gradient for indices
  
  
