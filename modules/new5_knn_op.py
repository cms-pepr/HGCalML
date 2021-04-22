
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_nknn_op = tf.load_op_library('new5_knn.so')

def New5Knn(K : int, coords, n_vtx_per_bin_cumulative, bin_neighbours, vtx_bin_assoc):
    '''
    todo
    '''

    return _nknn_op.New5Knn(n_neighbours=K, coords=coords, n_vtx_per_bin_cumulative = n_vtx_per_bin_cumulative, bin_neighbours = bin_neighbours, vtx_bin_assoc=vtx_bin_assoc)

#  _sknn_grad_op = tf.load_op_library('select_knn_grad.so')
#
#  @ops.RegisterGradient("SelectKnn")
#  def _SelectKnnGrad(op, gradidx, dstgrad):
#
#      coords = op.inputs[0]
#      indices = op.outputs[0]
#      distances = op.outputs[1]
#
#      coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)
#
#      return coord_grad, None, None #no grad for row splits and masking values
#
