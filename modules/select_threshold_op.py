
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Description
'''

_selthresh_op = tf.load_op_library('select_threshold.so')

#@tf.function
def SelectThreshold(x, pl, rowsplits, threshold=0.5):
    
    
    
    #print('x',x.shape)
    #print('pl',pl.shape)
    #print('rowsplits',rowsplits.shape)
    #scatter_idxs, rs  = _selthresh_op.SelectThreshold(th_value=x,
    #                                      rowsplits=rowsplits,
    #                                      threshold=threshold)
    
    if x.shape[0] is None: #keras pre-compile
        return pl, rowsplits, tf.zeros_like(rowsplits)
    
    all_idxs = tf.expand_dims(tf.range(x.shape[0]),axis=1)
    
    scatter_idxs = tf.expand_dims(all_idxs[x>threshold],axis=1)
    #print('scatter_idxs',scatter_idxs.shape)
    
    new_rs = [tf.zeros((1,), dtype='int32') for i in tf.range(tf.shape(rowsplits)[0])]
    ntot = tf.zeros((1,), dtype='int32')
    for i in tf.range(tf.shape(rowsplits)[0]-1):
        x_s = x[rowsplits[i]:rowsplits[i+1]][:,0]
        n = tf.shape(x_s[x_s>threshold])[0]
        ntot += n
        new_rs[i+1] = ntot
        
    rs = tf.concat(new_rs, axis=0)

    #print('scatter_idxs',scatter_idxs)
    #print('rs',rs)
    
    #print('rs',rs.shape)
    gathered = tf.gather_nd(pl, scatter_idxs)
    #print('gathered',gathered.shape)
    
    return tf.reshape(gathered, [-1, pl.shape[1]]), rs, scatter_idxs


