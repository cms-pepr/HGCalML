
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Description
'''

#_selthresh_op = tf.load_op_library('select_threshold.so')

#@tf.function
def SelectThreshold(x, pl, rowsplits, hardness = 20., threshold=0.5):
    
    
    
    #print('x',x.shape)
    #print('pl',pl.shape)
    #print('rowsplits',rowsplits.shape)
    #scatter_idxs, rs  = _selthresh_op.SelectThreshold(th_value=x,
    #                                      rowsplits=rowsplits,
    #                                      threshold=threshold)
    
    if x.shape[0] is None: #keras pre-compile
        return pl, rowsplits, tf.zeros_like(rowsplits)
    
    all_idxs = tf.expand_dims(tf.range(x.shape[0]),axis=1)
    
    #make sure at least two survive so that the gradient is not cut off and there is a gradient w.r.t. selection!
    global_max = tf.expand_dims(tf.reduce_max(x),axis=0)
    rest = tf.reduce_max(tf.where(x==global_max,-200.,x))# now the second to max
    global_max = tf.expand_dims( rest,axis=0)
    
    threshold = tf.reduce_min(tf.concat([global_max, tf.zeros_like(global_max)+threshold],axis=0))
    
    tf.print('threshold',threshold)
    offset = -4.6/hardness # approx =  - tf.math.log(1/0.01 - 1.)/hardness
    weight = tf.nn.sigmoid( hardness*(x + offset - threshold) )
    weighted_pl = weight * pl
    
    tf.print('xmean',tf.reduce_mean(x))
    
    scatter_idxs = tf.expand_dims(all_idxs[x>=threshold],axis=1)
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
    gathered = tf.gather_nd(weighted_pl, scatter_idxs)
    #print('gathered',gathered.shape)
    
    return tf.reshape(gathered, [-1, pl.shape[1]]), rs, scatter_idxs


