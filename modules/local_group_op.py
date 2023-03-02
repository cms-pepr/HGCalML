import tensorflow as tf
from tensorflow.python.framework import ops


_op = tf.load_op_library('local_group.so')

def LocalGroup(neighbour_idxs, hierarchy_idxs, hierarchy_score, row_splits,
               score_threshold):
    '''
    
    !!!! indices must have the -1s at the end!
    
    
    .Input("neighbour_idxs: int32") // V x K
    .Input("hierarchy_idxs: int32") // V x 1  per group
    .Input("hierarchy_score: float") // V x 1 score per group
    .Attr("score_threshold: float")
    .Input("row_splits: int32") // RS
    .Output("out_row_splits: int32") //RS'
    .Output("directed_neighbour_indices: int32") //V x K
    .Output("selection_indices: int32") //V' x 1
    .Output("backgather_idxs: int32"); //V

    '''
    rs,dirnidx,sel,ggather = _op.LocalGroup(
        neighbour_idxs=neighbour_idxs, 
        hierarchy_idxs=hierarchy_idxs, 
        hierarchy_score = hierarchy_score,
        score_threshold = score_threshold,
        row_splits=row_splits
        )
    #make sure shapes are ok
    dirnidx = tf.reshape(dirnidx, neighbour_idxs.shape)
    sel = tf.reshape(sel, [-1,1])
    ggather = tf.reshape(ggather, [-1,1])
    
    return rs,dirnidx,sel,ggather
    

@ops.RegisterGradient("LocalGroup")
def _LocalGroupGrad(op, g_rs,g_dirnidx,g_sel,g_ggather):
    
    return [None, None, None, None]