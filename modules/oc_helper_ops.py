import tensorflow as tf
from tensorflow.python.framework import ops

'''
Helper ops for object condensation.

--
CreateMidx: creates indices to gather vertices belonging to the truth objects.
Output dimension is: K x V_max_per_object
                     The rest is padded with '-1'
                     
The indices are to be used with SelectWithDefault(indices, tensor, default=0),
where the '-1' indices will be replaced by the default value.

For, for example for the attractive potential:

sel_dxs, m_not = CreateMidx(truth_idxs, True)
padmask = SelectWithDefault(sel_dxs, tf.zeros_like(truth_idxs,dtype='float32')+1., 0) 

distances = SelectWithDefault(sel_dxs, coords) 
distances = (tf.expand_dims(coords, axis=0) - distances)**2
distances *= padmask
...
...
m_not is the usual K x V dimension (for now) and can just be multiplied

'''

_op = tf.load_op_library('oc_helper_m_indices.so')

def CreateMidx(truth_idxs, calc_m_not=False):
    '''
   /*
 * Takes as helper input
 * y, idx, count = tf.unique_with_counts(x)
 *
 * y is the unique tensor
 * idx going to be ignored
 * count -> tf.max(count) -> nmax_per_unique
 *
 */

REGISTER_OP("MIndicesOpFunctor")
    //.Attr("return_mnot")
    .Input("asso_idxs: int32")
    .Input("unique_idxs: int32")
    .Input("nmax_per_unique: int32")
    .Output("sel_idxs: int32")
    .Output("m_not: float32");

    '''
    
    #only consider non-noise
    c_truth_idxs = truth_idxs[truth_idxs>=0]
    unique_idxs, _, cperunique = tf.unique_with_counts(c_truth_idxs)
    
    nmax_per_unique = tf.reduce_max(cperunique)
    #for empty tensors tf.reduce_max returns -lowest 32 bit integer or similar here
    #eager
    if nmax_per_unique.numpy() < 1:
        return None, None, None

    sel_dxs, m_not = _op.MIndices( 
        calc_m_not=calc_m_not,
        asso_idxs = truth_idxs,
        unique_idxs = unique_idxs,
        nmax_per_unique = nmax_per_unique
        )
    
    return sel_dxs, tf.expand_dims(m_not,axis=2), tf.expand_dims(cperunique,axis=1) #just some conventions
    
@ops.RegisterGradient("CreateMidx")
def _CreateMidxGrad(op, sel_dxs, m_not):
    return None


def SelectWithDefault(indices, tensor, default=0):
    
    expidxs = tf.expand_dims(indices,axis=2)
    tfidxs = tf.where(expidxs<0,0,expidxs)
    gtens = tf.gather_nd(tensor,tfidxs)
    out = tf.where(expidxs<0, default, gtens)
    
    #check if the size ends up as we might want
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(tf.shape(out)), tf.shape(tf.shape(indices)) + 1),
        tf.assert_equal(tf.shape(out)[1], tf.shape(indices)[1]),
        tf.assert_equal(tf.shape(tensor)[1], tf.shape(out)[2])]):
        
        return out


def per_rs_segids_to_unique(pred_sid, rs, return_nseg=False, strict_check=True):
    '''
    Input:
    - shower ids per hit: only ids>=0 allowed (but noise can be re-added with +1, <convert>, -1, tf.where(...)
    - row splits
    
    Args:
    - return_nseg: returns number of total ids across all row splits
    - strict_check: checks if the segment IDs have no empty spots (e.g. [0, 3, 2] would be missing '1')
    
    
    Transforms ids per hit (e.g. which condensation point it is associated to) per row split, such as:
    [ 2, 1, 2, 0, 3, |row split| 2, 1, 0 ]
    to unique ids spanning over row splits, such that the above becomes:
    [ 2, 1, 2, 0, 3, 6, 5, 4 ]
    
    If return_nseg is True, it also returns the number of unique ids, in the above case 7.
    
    If return_nseg is True:
        and also case strict_check==True, it is also required that all segment ids are used, e.g.
        [0, 2, 1] -> ok
        [0, 2, 0] -> not ok
        
        If strict checking is off, this can result in 'empty' segments.
    
    '''
    expand=False
    if len(tf.shape(pred_sid)) > 1:
        pred_sid = pred_sid[:,0]
        expand = True
        
    rs_seg_ids = tf.ragged.row_splits_to_segment_ids(rs, out_type=tf.int32)
    
    #get the offsets
    r_pred_maxids = tf.RaggedTensor.from_row_splits(pred_sid, rs)
    r_pred_maxids = tf.reduce_max(r_pred_maxids,axis=1) #returns -abs(max(int)) for empty
    r_pred_maxids = tf.where(r_pred_maxids<0, 0, r_pred_maxids)
    r_pred_maxids = tf.cumsum(r_pred_maxids)
    r_pred_maxids = tf.concat([rs[0:1],r_pred_maxids],axis=0) #add a leading zero
    
    #catch empty here
    
    #add the offsets at the right places
    pred_sid = pred_sid + tf.gather(r_pred_maxids, rs_seg_ids)
    pred_sid = tf.where(pred_sid<0, 0, pred_sid)#if they are empty
    if expand:
        pred_sid = pred_sid[:, tf.newaxis]
        
    if not return_nseg:    
        return pred_sid
    else:
        nseg = tf.reduce_max(pred_sid) + 1
        nseg = tf.where(nseg<0, 0, nseg) #if pred_sid is empty, reduce_max returns -max(int32)
        if strict_check:
            try:
                checksid = pred_sid
                if expand:
                    checksid = pred_sid[:,0]
                uniques = tf.reduce_sum(tf.cast(tf.unique(checksid)[0],dtype='int32') * 0 +  1)
                uniques = tf.where(nseg==0, 0, uniques)
                tf.assert_equal(uniques, nseg)
            except Exception as e:
                print('pred_sid, nseg, uniques, rs:', pred_sid, nseg, uniques, rs)
                raise e
        return pred_sid, nseg
    

