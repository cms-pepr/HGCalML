import tensorflow as tf
from tensorflow.python.framework import ops


_op = tf.load_op_library('local_group.so')

def LocalGroup(neighbour_idxs, hierarchy_idxs, hierarchy_score, row_splits,
               score_threshold):
    '''
    .Input("neighbour_idxs: int32") // V x K
    .Input("hierarchy_idxs: int32") // V x 1  per group
    .Input("hierarchy_score: float32") // V x 1 score per group
    .Attr("score_threshold: float32")
    .Input("global_idxs: int32") // V x 1
    .Input("row_splits: int32") // RS
    .Output("out_row_splits: int32") //RS'
    .Output("selection_neighbour_idxs: int32") //V' x K'
    .Output("backscatter_idxs: int32"); //V

    '''
    global_idxs = tf.range(hierarchy_idxs.shape[0],dtype='int32')
    rs,sel,ggather, npergroup = _op.LocalGroup(
        neighbour_idxs=neighbour_idxs, 
        hierarchy_idxs=hierarchy_idxs, 
        hierarchy_score = hierarchy_score,
        score_threshold = score_threshold,
        global_idxs=global_idxs, 
        row_splits=row_splits
        )
    
    return rs,sel,ggather, npergroup
    

@ops.RegisterGradient("LocalGroup")
def _LocalGroupGrad(op, g_rs,g_sel,g_ggather, g_npergroup):
    
    return [None, None, None, None, None]