import time

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.framework import ops

from bin_by_coordinates_op import BinByCoordinates
_bc_op = tf.load_op_library('assign_to_condensates.so')
_bc_op_binned = tf.load_op_library('assign_to_condensates_binned.so')
import numpy as np

def find_bins(ccoords, n_bins_sc, check=False):
    dmax_coords = tf.reduce_max(ccoords,axis=0)
    bin_width = (dmax_coords) / tf.cast(n_bins_sc, dtype='float32')
    n_bins = None  # re-calc in dimensions
    bin_width = tf.reduce_max(bin_width)[..., tf.newaxis]  # just add a '1' dimension
    n_bins = tf.cast(dmax_coords/bin_width, tf.int32)+1

    edges = tf.linspace(0.0, tf.reduce_max(dmax_coords), tf.reduce_max(n_bins), n_bins_sc)
    bin_width = edges[1]-edges[0]

    print(bin_width, edges, n_bins, n_bins_sc)
    # 0/0

    bins = tf.cast(tfp.stats.find_bins(ccoords, edges), tf.int32)
    bins_flat = bins[:, 0]*n_bins[1]*n_bins[2] + bins[:, 1] * n_bins[2] + bins[:, 2]


    return bins, bins_flat, n_bins, tf.reduce_max(bin_width)

def AssignToCondensatesBinned(ccoords,
                        betas,
                        row_splits,
                        beta_threshold,
                        dist=None):

    assert len(row_splits) == 2

    '''

    REGISTER_OP("AssignToCondensates")
    .Attr("radius: float")
    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("c_point_idx: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32");


    '''

    n_bins_sc = 30
    min_coords = tf.reduce_min(ccoords,axis=0,keepdims=True)
    ccoords -= min_coords
    # x, bins_flat, n_bins, bin_width, _ = BinByCoordinates(ccoords, row_splits, n_bins=n_bins_sc)
    _, bins_flat, n_bins, bin_width = find_bins(ccoords, n_bins_sc, True)


    flat_bin_finding_vector = np.concatenate((np.flip(np.cumprod(np.flip(n_bins.numpy())))[1:], [1]))
    print("Bin finding vector", n_bins.numpy(),flat_bin_finding_vector)

    sorting_indices = tf.argsort(bins_flat)
    ccoords = tf.gather(ccoords, sorting_indices)
    betas = tf.gather(betas, sorting_indices)
    dist = tf.gather(dist, sorting_indices)
    bins_flat = tf.gather(bins_flat, sorting_indices)
    bin_splits = tf.ragged.segment_ids_to_row_splits(bins_flat,num_segments=tf.reduce_prod(n_bins))

    # print(bin_width)
    # 0/0

    # min_coords = tf.reduce_min(ccoords_h,axis=0,keepdims=True)
    # ccoords_h -= min_coords

    # _, bins_flat_h, n_bins_h, bin_width_h, _ = BinByCoordinates(ccoords_h, row_splits, n_bins=n_bins_sc_2, pre_normalized=True)

    # _h is for high beta vertices, filtered ones for faster performance
    high_beta_indices = (betas>beta_threshold)[..., 0]
    betas_h = betas[high_beta_indices]
    ccoords_h = ccoords[high_beta_indices]
    dist_h = dist[high_beta_indices]
    row_splits_h = tf.constant([0, len(ccoords_h)])
    n_bins_sc_2 = 10
    bins_h, bins_flat_h, n_bins_h, bin_width_h = find_bins(ccoords_h, n_bins_sc_2, True)

    sorting_indices_h = tf.argsort(bins_flat_h)
    # ccoords_h = tf.gather(ccoords_h, sorting_indices_h)
    # betas_h = tf.gather(betas_h, sorting_indices_h)
    # dist_h = tf.gather(dist_h, sorting_indices_h)
    # bins_flat_h = tf.gather(bins_flat_h, sorting_indices_h)
    bin_splits_h = tf.ragged.segment_ids_to_row_splits(bins_flat_h,num_segments=tf.reduce_prod(n_bins_h))

    beta_filtered_indices_full = np.zeros(len(betas), np.int32)-1
    beta_filtered_indices_full[np.argwhere(betas > beta_threshold)[:, 0]] = np.arange(len(betas_h))

    indices_to_filtered = np.zeros(len(betas), np.int32) -1
    indices_to_filtered[high_beta_indices] = np.arange(len(betas_h))

    # print(np.argwhere(indices_to_filtered>0))
    # 0/0

    # tf.scatter_nd(tf.scatter(tf.range(len(betas))), xyz_h)




    print("Before", bin_width_h, bin_width, n_bins, n_bins_h)


    with tf.device('/cpu'):
        x,y =  _bc_op_binned.AssignToCondensatesBinned(
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
                bin_widths_h=bin_width_h,)

    print(x.numpy().tolist()[0:100])
    print("The sum is", np.sum(x), np.sum(y))
    print("U", np.unique(y), np.sum(y!=-1))
    # print("X", y[y!=-1])
    # print(np.argwhere(y.numpy()!=-1))

    return x


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



