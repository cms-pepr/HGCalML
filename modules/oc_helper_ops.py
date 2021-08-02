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
    return tf.where(expidxs<0, default, gtens)




