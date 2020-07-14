
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Description
'''

_selthresh_op = tf.load_op_library('select_threshold.so')

def SelectThreshold(x, pl, rowsplits, threshold=0.5):
    
    hardness=100
    cutoff = threshold-0.1
    scaler = tf.nn.sigmoid(hardness*(x-threshold))
    plnew = pl * scaler
    
    scatter_idxs, rs  = _selthresh_op.SelectThreshold(th_value=x,
                                          rowsplits=rowsplits,
                                          threshold=threshold)
    
    #print('scatter_idxs',scatter_idxs)
    #print('rs',rs)
    
    gathered = tf.gather_nd(plnew, scatter_idxs)
    
    return tf.reshape(gathered, [-1, plnew.shape[1]]), rs, scatter_idxs


