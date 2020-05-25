
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')

def SelectKnn(K : int, coords,  row_splits, tf_compatible=True, max_radius=-1.):
    idx,dst = _sknn_op.SelectKnn(n_neighbours=K, tf_compatible=tf_compatible, max_radius=max_radius,
                                 coords=coords, row_splits=row_splits)
    return idx

