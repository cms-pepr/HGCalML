
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')

def SelectKnn(K : int, coords,  row_splits):
    return _sknn_op.SelectKnn(n_neighbours=K, coords=coords, row_splits=row_splits)


