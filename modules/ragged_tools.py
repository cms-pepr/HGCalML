'''
convenience tools when dealing with ragged tensors where workarounds are still needed
'''
import tensorflow as tf

def unpack_ragged(rt):
    rs = []
    while hasattr(rt, 'values'):
        rs.append(rt.row_splits)
        rt = rt.values
    return rt, rs
    
def pack_ragged(vals, rsl):
    rt = vals
    for i in reversed(range(len(rsl))):
        rt = tf.RaggedTensor.from_row_splits(rt, rsl[i])
    return rt
    
def print_ragged_shape(x):
    rt, rs = unpack_ragged(x)
    print(f'unpacked shape: {rt.shape}, rs: {rs}')   
    
def rwhere(cond, x, y):
    '''
    sometimes tf.where does not work for reasons still unknown to me.
    try explicit unpacking these cases
    '''
    c, rsa = unpack_ragged(cond)
    x, rsb = unpack_ragged(x)
    y, rsc = unpack_ragged(y)
    
    out = tf.where(c, x, y)
    return pack_ragged(out,rsa)

def add_ragged_offset_to_flat(x, rs):
    '''
    for V x F inputs
    rs: [nrs]
    '''
    xr = tf.RaggedTensor.from_row_splits(x, rs) #[e, vr, f]
    xr += tf.expand_dims(rs, axis=1)[...,tf.newaxis]
    return xr.values


def rconcat(l, *args, **kwargs):
    '''
    just gets around the row split dtype issue
    '''
    assert len(l) == 2
    a,b = l
    arsdt = a.row_splits.dtype
    return tf.concat([a, b.with_row_splits_dtype(arsdt)], *args, **kwargs)

def normalise_index(t_idx, rs, add_rs_offset, return_n_per = False):
    '''
    normalises so they are consequtive per row split.
    :param add_rs_offset: makes then consequtive throughout the batch
    :param return_n_per: returns the number of unique indices (excluding noise!) per rs
    '''
    out = []
    nper = [0]
    offset = 0
    t_idx = t_idx[:,0] 
    for i in tf.range(rs.shape[0] - 1):
        #double unique?
        r_t_idx = t_idx[rs[i]:rs[i+1]]
        
        r_idxm = tf.reduce_min(r_t_idx, keepdims=True)#this could be noise or not
        
        r_t_idx = tf.concat([r_idxm, r_t_idx], axis=0) #make sure the smallest is first
        _, uidx = tf.unique(r_t_idx) # unique idx will five the smallest first as "0"
        r_t_idx = uidx[1:] + r_idxm # add-back smallest 0 -> -1 for noise, and remove first again
        if add_rs_offset:
            r_t_idx = tf.where(r_t_idx>=0, r_t_idx + offset, -1)#keep noise at -1
            offset = tf.reduce_max(r_t_idx)+1
        if return_n_per:
            nper.append(offset)
        out.append(r_t_idx)
    
    out = tf.concat(out, axis=0)[:,tf.newaxis]
    if not return_n_per:
        return out
    return out, tf.concat(nper, axis=0)
    
    
    
    
    