import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from bin_by_coordinates_op import BinByCoordinates
import numpy as np_

_bc_op_binned = tf.load_op_library('binned_assign_to_condensates.so')


def BuildAndAssignCondensatesBinned(ccoords,
                        betas,
                        dist,
                        row_splits,
                        beta_threshold,
                        assign_by_max_beta=True,
                        nbin_dims=3):
    """
    :param ccoords: Clustering coordinates of shape [nvert,ndim]
    :param betas: Betas of shape [nvert, 1]
    :param dist: Local distance thresholds of shape [nvert, 1]
    :param row_splits: Row splits of shape [nvert+1]
    :param beta_threshold: Minimum beta threshold
    :param nbin_dims <= ccoords.shape[1]. Binning to be done in these many number of coordinates
    :return: 5 elements: (in order)
    1. assignment: Assignment in ascending order from 0,1,2,3...N. Resents at every ragged segment.
    2. association: elements correspond to condensate index associated to each vertex  (self indexing). -1 for noise
    3. alpha_idx: alpha indices of the condensate points, this corresponds to `assignment`. n_condensates return can be used
                    as row split with it to make a ragged tensor
    4. is_cond: [nvert,1], 1 if the vertex is condensate otherwise 0
    5. n_condensates: a row splits like structure representing number of showrs in each segment
    """

    assert 1 <= nbin_dims <= 6
    row_splits = tf.constant(row_splits)
    num_rows = row_splits.shape[0]-1


    nvertmax = int(tf.reduce_sum(row_splits[1:] - row_splits[0:-1]))
    n_bins_sc = max(4,min(int(math.ceil(math.pow((nvertmax)/5, 1/3))), 25))

    min_coords = tf.reduce_min(ccoords,axis=0,keepdims=True)
    ccoords -= min_coords

    ccoords_2 = ccoords[:, 0:min(ccoords.shape[1], nbin_dims)]
    with tf.device('/cpu'):
        _, bins_flat, n_bins, bin_width = BinByCoordinates(ccoords_2, row_splits, n_bins=n_bins_sc, calc_n_per_bin=False, pre_normalized=True)

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
    row_splits_h = tf.ragged.segment_ids_to_row_splits(tf.ragged.row_splits_to_segment_ids(row_splits)[high_beta_indices], num_segments=num_rows)
    betas_h = betas[high_beta_indices]
    ccoords_h = ccoords[high_beta_indices]
    dist_h = dist[high_beta_indices]
    orig_indices_h = orig_indices[high_beta_indices]

    # Would set default to -1 instead of 0 (where no scattering is done)
    indices_to_filtered = tf.scatter_nd(tf.where(high_beta_indices),tf.range(betas_h.shape[0])+1, [betas.shape[0]])-1


    with tf.device('/cpu'):
        condensates_assigned_h,assignment, alpha_indices,asso, n_condensates =  _bc_op_binned.BinnedAssignToCondensates(
                beta_threshold=beta_threshold,
                assign_by_max_beta=assign_by_max_beta,
                ccoords=ccoords,
                dist=dist,
                beta=betas,
                bins_flat=bins_flat,
                bin_splits=bin_splits,
                n_bins=n_bins,
                bin_widths=bin_width,
                indices_to_filtered = indices_to_filtered,
                original_indices=orig_indices_h,
                ccoords_h=ccoords_h,
                dist_h=dist_h,
                beta_h=betas_h,
                row_splits=row_splits,
                row_splits_h=row_splits_h)

    assignment = tf.gather(assignment, sorting_indices_back)
    asso = tf.gather(asso, sorting_indices_back)
    pred_shower_alpha_idx = alpha_indices[alpha_indices!=-1]
    is_cond = tf.cast(tf.scatter_nd(pred_shower_alpha_idx[:, tf.newaxis], tf.ones(pred_shower_alpha_idx.shape[0]), [betas.shape[0]]), tf.bool)

    return assignment[:, tf.newaxis], asso, pred_shower_alpha_idx, is_cond[:, np.newaxis], n_condensates



# @ops.RegisterGradient("AssignToCondensates")
# def _AssignToCondensatesGrad(op, asso_grad):
#     return [None, None, None, None]


#### convenient helpers, not the OP itself





