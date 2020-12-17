
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_nknn_op = tf.load_op_library('new3_knn.so')

def New3Knn(K : int, coords,  row_splits, tf_compatible=True, max_radius=-1., n_bins_x=16, n_bins_y=16):
    '''
    todo
    '''

    return _nknn_op.New3Knn(n_neighbours=K, tf_compatible=tf_compatible, max_radius=max_radius,
                                 coords=coords, row_splits=row_splits,
                                 n_bins_x = n_bins_x, n_bins_y = n_bins_y)




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
#      return coord_grad, None #no grad for row splits
