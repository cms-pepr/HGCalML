import tensorflow as tf


def create_prs_indices(asso_idx):
    '''
    returns
     - indices to gather_nd the data back to original sorting
     - indices to gather_nd the data to prs sorting
     - pseudo row splits
     
    '''
    
    args = tf.argsort(asso_idx, axis=-1)
    
    sids = tf.expand_dims(args,axis=1)
    
    tf.assert_equal(sids[:,0], args, "sids args")
    
    sasso_idx = tf.gather(asso_idx,args)
    _,_,c = tf.unique_with_counts(sasso_idx)
    
    
    prs = tf.concat([tf.zeros_like(c[0:1], dtype='int32'),c], axis=0)
    prs = tf.cumsum(prs, axis=0)
    
    return sids, prs

def revert_prs(data, sids):
    return tf.scatter_nd(sids, data, tf.shape(data) )

def CreatePseudoRS(asso_idx, data):
    raise ValueError("BUG IN HERE")
    '''
    older implementation, clean up at some point 
    
    returns:
     - indices to gather_nd the data back to original sorting
     - pseudo row splits
     - resorted data, according to the pseudo RS
     - index in not resorted space which pseudo RS each point belongs to
    
    '''
    ids = tf.range(tf.shape(asso_idx)[0],dtype='int32')
    args = tf.argsort(asso_idx, axis=-1)
    sids = tf.expand_dims(tf.gather(ids,args),axis=1)
    
    sasso_idx = tf.gather(asso_idx,args)
    u,belongs_to_prs,c = tf.unique_with_counts(sasso_idx)
    
    
    prs = tf.concat([tf.zeros_like(c[0:1], dtype='int32'),c], axis=0)
    
    sdata = tf.gather_nd(data, sids)
    
    return tf.expand_dims(args,axis=1), tf.cumsum(prs, axis=0), sdata, tf.expand_dims(belongs_to_prs,axis=1)
