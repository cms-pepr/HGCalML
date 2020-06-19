import tensorflow as tf
from tensorflow.python.framework import ops


_bc_op = tf.load_op_library('build_condensates.so')

#@tf.function
def BuildCondensates(ccoords, betas, features, row_splits, radius=0.8, min_beta=0.1):
    '''
    betas are sum(V) x 1
    '''
    
    #https://github.com/tensorflow/tensorflow/issues/37512
    #get sorted etc here
    #this works in autograph mode while lists won't
    #my_list = tf.zeros([0])
    #for old_vals in OG_values:
    #    tf.autograph.experimental.set_loop_options(
    #        shape_invariants=[(my_list, tf.TensorShape([None]))])
    #    new_vals = tf.math.multiply(old_vals, 2)
    #    my_list = tf.concat([my_list, new_vals], 0)
    #return my_list

    
    summed_features, asso_idx = features, row_splits #for None call, get right types and pass through shapes
    if ccoords.shape[0] is not None:
        beta_sorting=tf.zeros([0],dtype='int32')
        
        for e in tf.range(tf.shape(row_splits)[0]-1):
            sorted = tf.argsort(betas[row_splits[e]:row_splits[e+1],0],axis=0, direction='DESCENDING')
            beta_sorting= tf.concat([beta_sorting,sorted],0)
        
        summed_features, asso_idx = _bc_op.BuildCondensates(ccoords=ccoords, betas=betas, beta_sorting=beta_sorting, 
                                       features=features, row_splits=row_splits, radius=radius, 
                                       min_beta=min_beta)
    return summed_features, asso_idx
    
    
    
_bc_grad_op = tf.load_op_library('build_condensates_grad.so')

@ops.RegisterGradient("BuildCondensates")
def _BuildCondensatesGrad(op, feat_grad, idx_grad):
    
    asso_idx = op.outputs[1]

    feat_grad = _bc_grad_op.BuildCondensatesGrad(sumfeat_grad=feat_grad , asso_idx=asso_idx )

    return [None, None, None, feat_grad, None] 
  
  
