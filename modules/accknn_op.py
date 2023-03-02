
import tensorflow as tf
from tensorflow.python.framework import ops
import globals as gl
from oc_helper_ops import SelectWithDefault

'''
Indices MUST be unique in each row.
Only exception are multiple self-references, that can be used as sort of padding.
Alternatively, the index -1 is skipped (non TF conpatible padding)

'''

_accknn_op = tf.load_op_library('accumulate_knn.so')
_accknn_grad_op = tf.load_op_library('accumulate_knn_grad.so')


def AccumulateLinKnn(weights,  features, indices, 
                  mean_and_max=True, force_tf=False):
    '''
    Accumulates neighbour features with linear weights (not exp(-w) as AccumulateKnn)
    '''
    if (not gl.acc_ops_use_tf_gradients) and (not force_tf):
        return _accknn_op.AccumulateKnn(distances=weights,  features=features, indices=indices,
                                    n_moments=0, mean_and_max=mean_and_max)
    
    
    weights = tf.expand_dims(weights,axis=2) #V x K x 1
    nfeat = SelectWithDefault(indices, features, 0.) # V x K x F
    wfeat = weights*nfeat
    fmean = tf.reduce_mean(wfeat,axis=1)# V x F
    fmax = tf.reduce_max(wfeat,axis=1)
    fout = fmean
    if mean_and_max:
        fout = tf.concat([fmean,fmax],axis=1)
    return fout,None


def AccumulateKnn(distances,  features, indices, 
                  mean_and_max=True,force_tf=False):
    '''
    
    .Output("out_features: float32")
    .Output("out_max_idxs: int32");
    
    
    Assumes that neighbour indices can be padded with -1, but not mixed, e.g. [1,4,-1,2] needs to be [1,4,2,-1]
    Other than the padding, the indices must be unique
    
    '''
    #compatibility
    distances = tf.exp(-distances)

    
    if (not gl.acc_ops_use_tf_gradients) and (not force_tf):
        return _accknn_op.AccumulateKnn(distances=distances,  features=features, indices=indices,
                                    n_moments=0, mean_and_max=mean_and_max)
    
    
    distances = tf.expand_dims(distances,axis=2) #V x K x 1
    nfeat = SelectWithDefault(indices, features, 0.) # V x K x F
    wfeat = distances*nfeat
    fmean = tf.reduce_mean(wfeat,axis=1)# V x F
    fmax = tf.reduce_max(wfeat,axis=1)
    fout = fmean
    if mean_and_max:
        fout = tf.concat([fmean,fmax],axis=1)
    return fout,None

#this refers to the OP called AccumulateKnn, not the function below
@ops.RegisterGradient("AccumulateKnn")
def _AccumulateKnnGrad(op, grad, gradmaxidxs):
    """
      
    """
    
    
    distances  = op.inputs[0]
    features  = op.inputs[1]
    max_feat_indices = op.outputs[1]
    neigh_indices = op.inputs[2]
    
    dist_grad , feat_grad = _accknn_grad_op.AccumulateKnnGrad(grad_from_out_features=grad,
                                                               distances=distances,
                                                               features=features,
                                                               neigh_indices=neigh_indices,
                                                               max_feat_indices=max_feat_indices)
    
    return [dist_grad , feat_grad, None] #no gradient for indices

