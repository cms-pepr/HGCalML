import tensorflow as tf
from tensorflow.python.framework import ops


_bc_op = tf.load_op_library('build_condensates.so')

#@tf.function
def BuildCondensates(ccoords, betas, row_splits, radius=0.8, min_beta=0.1, soft=False):
    '''
    betas are sum(V) x 1
    
      .Attr("radius: float")
    .Attr("min_beta: float")
    .Attr("soft: bool")
    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("is_cpoint: int32");


    '''
    
    return _bc_op.BuildCondensates(ccoords=ccoords, betas=betas, row_splits=row_splits, 
                                   radius=radius, min_beta=min_beta,soft=soft)
    
    

@ops.RegisterGradient("BuildCondensates")
def _BuildCondensatesGrad(op, asso_grad, is_cgrad):
    
    return [None, None, None]
  
  
