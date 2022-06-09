import tensorflow as tf
from tensorflow.python.framework import ops

_bc_op = tf.load_op_library('assign_to_condensates.so')


# @tf.function
def AssignToCondensates(ccoords,
                        c_point_idx,
                        row_splits,
                        radius=0.8,
                        dist=None):
    '''

    REGISTER_OP("AssignToCondensates")
    .Attr("radius: float")
    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("c_point_idx: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32");


    '''
    if dist is None:
        dist = tf.ones_like(ccoords[:, 0:1])
    else:
        tf.assert_equal(tf.shape(ccoords[:, 0:1]), tf.shape(dist))

    return _bc_op.AssignToCondensates(ccoords=ccoords,
                                      dist=dist,
                                      c_point_idx=c_point_idx,
                                      row_splits=row_splits,
                                      radius=radius)


@ops.RegisterGradient("AssignToCondensates")
def _AssignToCondensatesGrad(op, asso_grad):
    return [None, None, None, None]


#### convenient helpers, not the OP itself


from condensate_op import BuildCondensates


def BuildAndAssignCondensates(ccoords, betas, row_splits,
                              radius=0.8, min_beta=0.1,
                              dist=None,
                              soft=False,
                              assign_radius=None):
    if assign_radius is None:
        assign_radius = radius

    asso, iscond, ncond = BuildCondensates(ccoords, betas, row_splits,
                                           radius=radius, min_beta=min_beta,
                                           dist=dist,
                                           soft=soft)

    c_point_idx, _ = tf.unique(asso)
    asso_idx = AssignToCondensates(ccoords,
                                   c_point_idx,
                                   row_splits,
                                   radius=assign_radius,
                                   dist=dist)

    return asso_idx, iscond, ncond



