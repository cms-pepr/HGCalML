

import tensorflow as tf
from tensorflow.python.framework import ops

_push_knn = tf.load_op_library('push_knn.so')
_push_knn_grad = tf.load_op_library('push_knn_grad.so')


def PushKnn(w,f,nidx):
    '''
    Pushes features (summing them) to the neighbours.
    This assumes that if there is a self neighbour index, that is actually means something!
    This op is compatible with '-1' indices as synonym for no neighbour
    '''
    #go through the columns
    
    #add a zero column for the '-1's
    return _push_knn.PushKnn(weights = w,features = f,indices = nidx)

@ops.RegisterGradient("PushKnn")
def _PushKnnGrad(op, grad):
    
    w  = op.inputs[0]
    f  = op.inputs[1]
    nidx = op.inputs[2]
    
    #grad = tf.debugging.check_numerics(grad,"_PushKnnGrad: input gradient")
    #f = tf.debugging.check_numerics(f,"_PushKnnGrad: input features")
    #w = tf.debugging.check_numerics(w,"_PushKnnGrad: input weights")
    
    wgrad, fgrad = _push_knn_grad.PushKnnGrad(grad=grad,weights=w,features=f,indices=nidx)
    
    #fgrad = tf.debugging.check_numerics(fgrad,"_PushKnnGrad: fgrad gradient")
    #wgrad = tf.debugging.check_numerics(wgrad,"_PushKnnGrad: wgrad gradient")
    
    return wgrad, fgrad, None #no gradient for indices

def _tf_push_knn(w,f,nidx):
    '''
    
    This is just for unit tests. Do not use, it is very slow and uses a lot of memory
    
    Pushes features (summing them) to the neighbours.
    This assumes that if there is a self neighbour index, that is actually means something!
    This op is compatible with '-1' indices as synonym for no neighbour
    '''
    #go through the columns
    
    #add a zero column for the '-1's
    if f.shape[0] == None:
        return f
    
    f = tf.concat([ 0.* f[0:1], f ],axis=0)
    w = tf.concat([ 0.* w[0:1], w ],axis=0)
    nidx = tf.concat( [ 0 * nidx[0:1], nidx + 1], axis=0) #zero goes to zero
    
    out = f*0.
    for i_c in tf.range(nidx.shape[1]):
        for i_v in tf.range(nidx.shape[0]): #make sure this is atomic
            sel_nidx = nidx[i_v:i_v+1,i_c:i_c+1]
            sel_w = w[i_v:i_v+1,i_c:i_c+1]
            sel_f = f[i_v:i_v+1]
            out += tf.scatter_nd(sel_nidx, sel_w*sel_f, shape=f.shape)
    return out[1:] #remove 0 again

#PushKnn = _tf_push_knn #DEBUG
