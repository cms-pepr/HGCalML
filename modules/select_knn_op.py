
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_sknn_op = tf.load_op_library('select_knn.so')

def SelectKnn(K : int, coords,  row_splits, tf_compatible=True, max_radius=-1., self_loop=False):
    nneigh = K 
    if not self_loop:
        nneigh += 1
    idx,dst = _sknn_op.SelectKnn(n_neighbours=nneigh, tf_compatible=tf_compatible, max_radius=max_radius,
                                 coords=coords, row_splits=row_splits)
    if not self_loop:
        return idx[:,1:]
    else:
        return idx

