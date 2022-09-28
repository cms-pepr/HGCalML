import math

import numpy as np
import tensorflow as tf
from bin_by_coordinates_op import BinByCoordinates

_bc_op_binned = tf.load_op_library('binned_assign_to_condensates.so')


def calc_ragged_shower_indices(assignment, row_splits, gather_noise=True, return_reverse=False):
    """

    :param assignment: [nvert, 1] Values should be consecutive i.e. -1,0,1,2,3,...N. Same is true for the every segment
                        i.e. for the second segment it should restart from -1,0,1,2,3...N. This is the first return
                        value of BuildAndAssignCondensatesBinned.
    :param row_splits: [nvert+1], row splits
    :param gather_noise: boolean, whether to gather noise or not. If set to True, the first element in the shower
                         dimension will always correspond to the noise even if there is no noise vertex present.
    :param return_reverse: returns the reverse operation to gather back. Example (x is some point feature vector [V x F]):
                           idx, revidx = calc_ragged_shower_indices(assignement, row_splits, return_reverse=True)
                           xc = tf.gather_nd(x, idx)
                           xc = tf.reduce_mean(xc, axis=2) #mean of all associated hits
                           xg = tf.gather_nd(xc, revidx) #non-ragged representation [V x F], with the original row splits
    :return: a double ragged tensor of indices, first ragged dimension iterates endcaps/samples and the second, showers
             in that endcap. This can be used in gather_nd as follows:
                ragged_indices = calc_ragged_shower_indices(assignment, row_splits, gather_noise=False)
                x = tf.gather_nd(assignment, ragged_indices)
    :return: (opt) an index tensor to reverse the operation (TBI)

    """
    if return_reverse:
        assert  gather_noise #reverse cannot work when noise is discarded
    
    orig_assignement = assignment
    if gather_noise:
        assignment = assignment[:, 0] + 1
    else:
        assignment = assignment[:, 0]
        old_segment_ids = tf.ragged.row_splits_to_segment_ids(row_splits)
        sel = assignment >= 0
        new_segment_ids = old_segment_ids[sel]
        back_indices = tf.range(len(assignment))[sel]
        assignment = assignment[sel]

        row_splits = tf.ragged.segment_ids_to_row_splits(new_segment_ids, num_segments=len(row_splits)-1)

    n_condensates = tf.math.segment_max(assignment, tf.ragged.row_splits_to_segment_ids(row_splits)) + 1
    n_condensates_n = tf.concat(([0], tf.cumsum(n_condensates)), axis=0)

    assignment_u = assignment+tf.gather(n_condensates_n, tf.ragged.row_splits_to_segment_ids(row_splits))
    rsx = tf.math.unsorted_segment_sum(tf.ones_like(assignment_u), assignment_u, n_condensates_n[-1])

    sorting_indices_showers_ragged = tf.argsort(assignment_u)
    if not gather_noise:
        sorting_indices_showers_ragged = tf.gather(back_indices, sorting_indices_showers_ragged)

    sorting_indices_showers_ragged = tf.RaggedTensor.from_row_lengths(sorting_indices_showers_ragged[..., tf.newaxis], rsx)
    sorting_indices_showers_ragged = tf.RaggedTensor.from_row_splits(sorting_indices_showers_ragged, n_condensates_n)

    if return_reverse:
        revidx = tf.RaggedTensor.from_row_splits(orig_assignement+1, row_splits)
        nrs = tf.shape(sorting_indices_showers_ragged.row_splits)[0]-1
        addrs = tf.range(nrs)[...,tf.newaxis][...,tf.newaxis]
        revidx = tf.concat([0 * revidx + addrs,revidx],axis=-1)
        return sorting_indices_showers_ragged, revidx.values
    else:
        return sorting_indices_showers_ragged

def BuildAndAssignCondensatesBinned(ccoords,
                        betas,
                        dist,
                        row_splits,
                        beta_threshold,
                        distance_threshold = 1.,
                        assign_by_max_beta=True,
                        nbin_dims=3):
    """
    :param ccoords: [nvert,ndim] Clustering coordinates of shape
    :param betas: [nvert, 1] Betas of shape
    :param dist: [nvert, 1] Local distance thresholds of shape
    :param row_splits: Row splits of shape [nvert+1]
    :param beta_threshold: Minimum beta threshold (scalar)
    :param nbin_dims <= ccoords.shape[1]. Binning to be done in these many number of coordinates (scalar)
    :param assign_by_max_beta: Boolean. If set to True, the higher beta vertex eats up all the vertices in its radius. If set to
             False, the assignment for a vertex is done to the condensate it is the closest to. True behaves like
             anti-kt jet clustering algorighm while False behaves like XCone.
    :return: 5 elements: (in order)
    1. assignment: [nvert,1] Assignment in ascending order from 0,1,2,3...N. Resets at every ragged segment.
    2. association: elements correspond to condensate index associated to each vertex  (self indexing). -1 for noise
    3. alpha_idx: alpha indices of the condensate points, this corresponds to `assignment`. n_condensates return can be used
                    as row split with it to make a ragged tensor
    4. is_cond: [nvert,1], 1 if the vertex is condensate otherwise 0
    5. n_condensates: a row splits like structure representing number of showers in each segment
    """

    assert 1 <= nbin_dims <= 6
    assert 0. <= beta_threshold <= 1.
    assert distance_threshold > 0.
    #row_splits = tf.constant(row_splits)
    num_rows = row_splits.shape[0]-1

    dist =  dist / distance_threshold

    nvertmax = int(tf.reduce_sum(row_splits[1:] - row_splits[0:-1]))
    n_bins_sc = max(4,min(int(math.ceil(math.pow((nvertmax)/5, 1/3))), 25))

    min_coords = tf.reduce_min(ccoords,axis=0,keepdims=True)
    ccoords -= min_coords

    ccoords_2 = ccoords[:, 0:min(ccoords.shape[1], nbin_dims)]
    #with tf.device('/cpu'):
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
        condensates_assigned_h,assignment, alpha_indices,asso,n_condensates =  _bc_op_binned.BinnedAssignToCondensates(
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





