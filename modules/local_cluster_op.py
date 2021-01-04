import tensorflow as tf
from tensorflow.python.framework import ops


_op = tf.load_op_library('local_cluster.so')

def LocalCluster(neighbour_idxs, hierarchy_idxs, row_splits):
    '''
    .Input("neighbour_idxs: int32") //change to distances!!
    .Input("hierarchy_idxs: int32")
    .Input("global_idxs: int32")
    .Input("row_splits: int32")
    .Output("out_row_splits: int32")
    .Output("selection_idxs: int32")
    .Output("backscatter_idxs: int32");

    '''
    global_idxs = tf.range(hierarchy_idxs.shape[0],dtype='int32')
    rs,sel,ggather = _op.LocalCluster(neighbour_idxs=neighbour_idxs, 
                            hierarchy_idxs=hierarchy_idxs, 
                            global_idxs=global_idxs, 
                            row_splits=row_splits)
    
    return rs,sel,ggather
    

@ops.RegisterGradient("LocalCluster")
def _LocalClusterGrad(op, asso_grad, is_cgrad,ncondgrad):
    
    return [None, None, None, None, None]