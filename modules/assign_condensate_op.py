import time

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.framework import ops

from bin_by_coordinates_op import BinByCoordinates
import numpy as np_

_bc_op_binned = tf.load_op_library('assign_to_condensates_binned.so')


def AssignToCondensatesBinned(ccoords,
                        betas,
                        row_splits,
                        beta_threshold,
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

    row_splits = tf.constant(row_splits)
    num_rows = row_splits.shape[0]-1

    n_bins_sc = 30
    min_coords = tf.reduce_min(ccoords,axis=0,keepdims=True)
    ccoords -= min_coords
    _, bins_flat, n_bins, bin_width, _ = BinByCoordinates(ccoords, row_splits, n_bins=n_bins_sc)
    # _, bins_flat, n_bins, bin_width = find_bins(ccoords, n_bins_sc, True)

    sorting_indices = tf.argsort(bins_flat)
    sorting_indices_back = tf.argsort(sorting_indices)
    ccoords = tf.gather(ccoords, sorting_indices)
    betas = tf.gather(betas, sorting_indices)
    dist = tf.gather(dist, sorting_indices)
    bins_flat = tf.gather(bins_flat, sorting_indices)
    orig_indices = tf.gather(tf.range(betas.shape[0]), sorting_indices)
    bin_splits = tf.ragged.segment_ids_to_row_splits(bins_flat,num_segments=tf.reduce_prod(n_bins)*num_rows)

    # _h is for high beta vertices, filtered ones for faster performance
    high_beta_indices = (betas>beta_threshold)[..., 0]
    row_splits_h = tf.ragged.segment_ids_to_row_splits(tf.ragged.row_splits_to_segment_ids(row_splits)[high_beta_indices])

    print(row_splits)
    print(row_splits_h)

    betas_h = betas[high_beta_indices]
    ccoords_h = ccoords[high_beta_indices]
    dist_h = dist[high_beta_indices]
    orig_indices_h = orig_indices[high_beta_indices]
    # row_splits_h = tf.constant([0, len(ccoords_h)])
    n_bins_sc_2 = 10
    # bins_h, bins_flat_h, n_bins_h, bin_width_h = find_bins(ccoords_h, n_bins_sc_2, True)
    _, bins_flat_h, n_bins_h, bin_width_h, _ = BinByCoordinates(ccoords_h, row_splits_h, n_bins=n_bins_sc)
    # x, bins_flat, n_bins, bin_width, _ = BinByCoordinates(ccoords, row_splits, n_bins=n_bins_sc)


    # _h tensors aren't sorted by bins because they are not currently being used. Maybe they'd be useful for the CUDA
    # op.
    # sorting_indices_h = tf.argsort(bins_flat_h)
    # ccoords_h = tf.gather(ccoords_h, sorting_indices_h)
    # betas_h = tf.gather(betas_h, sorting_indices_h)
    # dist_h = tf.gather(dist_h, sorting_indices_h)
    # bins_flat_h = tf.gather(bins_flat_h, sorting_indices_h)
    bin_splits_h = tf.ragged.segment_ids_to_row_splits(bins_flat_h,num_segments=tf.reduce_prod(n_bins_h)*num_rows)

    # Would set default to -1 instead of 0 (where no scattering is done)
    indices_to_filtered = tf.scatter_nd(tf.where(high_beta_indices),tf.range(betas_h.shape[0])+1, [betas.shape[0]])-1

    with tf.device('/cpu'):
        condensates_assigned,assignment =  _bc_op_binned.AssignToCondensatesBinned(
                beta_threshold=beta_threshold,
                ccoords=ccoords,
                dist=dist,
                beta=betas,
                bins_flat=bins_flat,
                bin_splits=bin_splits,
                n_bins=n_bins,
                bin_widths=bin_width,
                indices_to_filtered = indices_to_filtered,
                ccoords_h=ccoords_h,
                dist_h=dist_h,
                beta_h=betas_h,
                bins_flat_h=bins_flat_h,
                bin_splits_h=bin_splits_h,
                n_bins_h=n_bins_h,
                row_splits=row_splits,
                row_splits_h=row_splits_h,
                bin_widths_h=bin_width_h,)

    assignment = tf.gather(assignment, sorting_indices_back)
    pred_shower_alpha_idx = orig_indices_h[condensates_assigned>0]
    is_cond = tf.scatter_nd(pred_shower_alpha_idx[:, tf.newaxis], tf.ones(pred_shower_alpha_idx.shape[0]), [betas.shape[0]])

    return is_cond, assignment


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



def BuildAndAssignCondensatesBinned(ccoords, betas, row_splits,
                              radius=0.8, min_beta=0.1,
                              dist=None,
                              soft=False):

    dist = dist*radius
    t1 = time.time()
    asso_idx = AssignToCondensatesBinned(ccoords=ccoords,
                                   betas=betas,
                                   row_splits=row_splits,
                                   beta_threshold=min_beta,
                                   dist=dist)
    print("Took", time.time()-t1, "seconds.")

    return asso_idx



