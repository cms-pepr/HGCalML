
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Description
'''

#_selthresh_op = tf.load_op_library('select_threshold.so')

#@tf.function
def SelectThreshold(x, pl, rowsplits, hardness = 20., threshold=0.5):
    
    
    '''
    
    x >= 0
    
    '''
    
    if x.shape[0] is None: #keras pre-compile
        return pl, rowsplits, tf.zeros_like(rowsplits), x
    
    all_idxs = tf.expand_dims(tf.range(x.shape[0]),axis=1)
    
    #make sure at least two survive so that the gradient is not cut off and there is a gradient w.r.t. selection!
    #do this per RS
    
    tf.print(x.shape, rowsplits.shape)
    xragged = tf.RaggedTensor.from_row_splits(values=x,
              row_splits=rowsplits)
    
    #make sure at least one vertex per RS remains
    max_per_rs = tf.reduce_max(xragged, axis=1)
    global_max = tf.expand_dims(tf.reduce_min(max_per_rs),axis=0)-1e-6
    
    threshold = tf.reduce_min(tf.concat([global_max, tf.zeros_like(global_max)+threshold],axis=0))
    
    tf.print('threshold',threshold)
    offset = -4.6/hardness # approx =  - tf.math.log(1/0.01 - 1.)/hardness
    weight = tf.nn.sigmoid( hardness*(x + offset - threshold) )
    weighted_pl = weight * pl
    
    tf.print('xmean, xmin',tf.reduce_mean(x), tf.reduce_min(x))
    
    scatter_idxs = tf.expand_dims(all_idxs[x>=threshold],axis=1)
    #print('scatter_idxs',scatter_idxs.shape)
    
    new_rs = [tf.zeros((1,), dtype='int32') for i in tf.range(tf.shape(rowsplits)[0])]
    ntot = tf.zeros((1,), dtype='int32')
    for i in tf.range(tf.shape(rowsplits)[0]-1):
        x_s = x[rowsplits[i]:rowsplits[i+1]][:,0]
        n = tf.shape(x_s[x_s>=threshold])[0]
        ntot += n
        new_rs[i+1] = ntot
        
    rs = tf.concat(new_rs, axis=0)

    gathered = tf.gather_nd(weighted_pl, scatter_idxs)
    gathered_threshvals = tf.gather_nd(x, scatter_idxs)
    
    plout, rsnew, scat_idx, thout = tf.reshape(gathered, [-1, pl.shape[1]]), rs, scatter_idxs, gathered_threshvals
    
    
    return plout, rsnew, scat_idx, thout


def SelectThresholdRagged(x, pl, rowsplits, hardness = 20., threshold=0.5):
    
    
    '''
    
    x >= 0
    
    '''
    
    if x.shape[0] is None: #keras pre-compile
        return pl, rowsplits, tf.zeros_like(rowsplits), x
    
    all_idxs = tf.expand_dims(tf.range(x.shape[0]),axis=1)
    
    #make sure at least two survive so that the gradient is not cut off and there is a gradient w.r.t. selection!
    #do this per RS
    
    tf.print("HERE A")
    tf.print(x.shape, rowsplits.shape)
    xragged = tf.RaggedTensor.from_row_splits(values=x,
              row_splits=rowsplits)
    
    tf.print("HERE B")
    #make sure at least one vertex per RS remains
    max_per_rs = tf.reduce_max(xragged, axis=1)
    threshold = tf.reduce_min(tf.concat([max_per_rs, tf.zeros_like(max_per_rs)+threshold],axis=1),axis=1,keepdims=True)
    
    threshold +=xragged*0.#broadcast to right dimensions (hopefully already implemented in TF
    
    tf.print(threshold,summarize=300)
    tf.print('xragged',xragged.shape)
    sel = xragged[xragged>threshold]
    tf.print('sel',sel.shape)
    
    global_max = tf.expand_dims(tf.reduce_min(max_per_rs),axis=0)-1e-2
    
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

    gathered = tf.gather_nd(weighted_pl, scatter_idxs)
    gathered_threshvals = tf.gather_nd(x, scatter_idxs)
    
    plout, rsnew, scat_idx, thout = tf.reshape(gathered, [-1, pl.shape[1]]), rs, scatter_idxs, gathered_threshvals
    
    
    return plout, rsnew, scat_idx, thout


#SelectThreshold = SelectThresholdRagged