import tensorflow as tf

def CreatePseudoRS(asso_idx, data):
    '''
    returns:
     - indices to gather_nd the data back to original sorting
     - pseudo row splits
     - resorted data, according to the pseudo RS
    
    '''
    ids = tf.range(tf.shape(asso_idx)[0],dtype='int32')
    args = tf.argsort(asso_idx, axis=-1)
    sids = tf.expand_dims(tf.gather(ids,args),axis=1)
    
    sasso_idx = tf.gather(asso_idx,args)
    u,belongs_to_prs,c = tf.unique_with_counts(sasso_idx)
    
    
    c = tf.concat([tf.zeros_like(c[0:1], dtype='int32'),c], axis=0)
    
    sdata = tf.gather_nd(data, sids)
    
    return tf.expand_dims(args,axis=1), tf.cumsum(c, axis=0), sdata, tf.expand_dims(belongs_to_prs,axis=1)
   