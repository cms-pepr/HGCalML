import tensorflow as tf
from tensorflow.python.framework import ops


_bc_op = tf.load_op_library('build_condensates.so')

#@tf.function
def BuildCondensates(ccoords, betas, row_splits, 
                     radius=0.8, min_beta=0.1, 
                     dist=None,
                     tosum=None,
                     soft=False):
    '''
    betas are sum(V) x 1
    
    .Attr("radius: float")
    .Attr("min_beta: float")
    .Attr("soft: bool")
    .Attr("sum: bool")
    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("dist: float32")
    .Input("tosum: float32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("is_cpoint: int32")
    .Output("n_condensates: int32")
    .Output("summed: float32");


    '''
    dosum = True
    if dist is None:
        dist = tf.ones_like(betas)
    else:
        tf.assert_equal(tf.shape(betas),tf.shape(dist))
    if tosum is None:
        tosum = tf.constant([[]],dtype='float32')
        dosum=False
    
    asso,isc,ncond,summed = _bc_op.BuildCondensates(radius=radius, min_beta=min_beta,soft=soft,sum=dosum,
        ccoords=ccoords, betas=betas, dist=dist,tosum=tosum, row_splits=row_splits)
    if dosum:
        return asso,isc,ncond,summed
    return asso,isc,ncond
    

@ops.RegisterGradient("BuildCondensates")
def _BuildCondensatesGrad(op, asso_grad, is_cgrad,ncondgrad,sumgrad):
    
    return [None, None, None, None,None]
  
  
