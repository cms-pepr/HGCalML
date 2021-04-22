
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_nknn_op = tf.load_op_library('pre2_knn.so')

def Pre2Knn(coords, n_bins_x, n_bins_y):
    '''
    todo
    '''

    return _nknn_op.Pre2Knn(coords=coords, n_bins_x = n_bins_x, n_bins_y = n_bins_y)
