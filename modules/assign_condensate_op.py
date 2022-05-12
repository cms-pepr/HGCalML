import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from bin_by_coordinates_op import BinByCoordinates
from numba import njit

_bc_op = tf.load_op_library('assign_to_condensates.so')

#@tf.function
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
        dist = tf.ones_like(ccoords[:,0:1])
    else:
        tf.assert_equal(tf.shape(ccoords[:,0:1]),tf.shape(dist))
    
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

@njit
def collect_func_native_gen(low_bin_indices,high_bin_indices, n_bins,flat_bin_finding_vector, assignment, coords, alpha_coords, alpha_radius,
                        shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full):

    range_bin_indices = high_bin_indices - low_bin_indices
    ndim = len(high_bin_indices)

    bin_finding_div_vector = np.ones_like(low_bin_indices)
    for i in range(ndim - 1):
        bin_finding_div_vector[ndim - i - 2] = bin_finding_div_vector[ndim - i - 1] * range_bin_indices[ndim - i - 1]

    total_iterations = np.prod(range_bin_indices)

    for iteration in range(total_iterations):
        bin_vector = low_bin_indices + (iteration // bin_finding_div_vector) % range_bin_indices
        b_flat = np.sum(flat_bin_finding_vector * bin_vector)

        if np.any(bin_vector >= n_bins):
            continue

        start_index = row_splits[b_flat]
        end_index = row_splits[b_flat + 1]
        for l in range(start_index, end_index):
            if assignment[l] == -1:
                if np.sum((coords[l] - alpha_coords) ** 2) < alpha_radius ** 2:
                    assignment[l] = shower_idx
                    beta[l] = 0
                    if beta_filtered_indices_full[l] != -1:
                        beta_filtered[beta_filtered_indices_full[l]] = 0


def build_condensates_cpu(coords, beta, pred_dist, beta_threshold=0.3, dist_threshold=1.5, ):
    beta = beta[:, 0]
    coords = coords - np.min(coords, axis=0, keepdims=True)
    pred_dist = pred_dist[:, 0] * dist_threshold

    _, bins_flat, n_bins, bin_width, _ = BinByCoordinates(coords, [0, len(coords)], n_bins=30)

    bins_flat = bins_flat.numpy()
    n_bins = n_bins.numpy()
    bin_width = float(bin_width[0])

    sorting_indices = np.argsort(bins_flat)
    beta = beta[sorting_indices]
    coords = coords[sorting_indices]
    bins_flat = bins_flat[sorting_indices]
    pred_dist = pred_dist[sorting_indices]

    row_splits = tf.ragged.segment_ids_to_row_splits(bins_flat,num_segments=np.prod(n_bins)).numpy()

    flat_bin_finding_vector = np.concatenate((np.flip(np.cumprod(np.flip(n_bins)))[1:], [1]))

    assignment = np.zeros_like(beta, dtype=np.int32) -1

    shower_idx = 0
    beta_filtered_indices = np.argwhere(beta > beta_threshold)[:, 0]
    beta_filtered_indices_full = (assignment * 0 -1).astype(np.int32)
    beta_filtered_indices_full[beta_filtered_indices] = np.arange(len(beta_filtered_indices))
    beta_filtered = np.array(beta[beta_filtered_indices])

    bin_width = np.array([bin_width] * int(coords.shape[1]))
    pred_shower_alpha_idx = assignment * 0

    while True:
        alpha_idx = beta_filtered_indices[np.argmax(beta_filtered)]
        max_beta = beta[alpha_idx]
        # print("max beta", max_beta)
        if max_beta < beta_threshold:
            break
        alpha_coords = coords[alpha_idx]
        alpha_radius = pred_dist[alpha_idx]

        low_bin_indices = np.floor((alpha_coords - alpha_radius)/bin_width).astype(np.int32)
        high_bin_indices = np.ceil((alpha_coords + alpha_radius)/bin_width).astype(np.int32)
        collect_func_native_gen(low_bin_indices, high_bin_indices, n_bins, flat_bin_finding_vector, assignment, coords, alpha_coords,
                                alpha_radius, shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full)


        beta[alpha_idx] = 0
        pred_shower_alpha_idx[shower_idx] = alpha_idx
        shower_idx += 1

    assignment_2 = assignment * 0
    assignment_2[sorting_indices] = assignment

    return assignment_2, pred_shower_alpha_idx[0:shower_idx]


def BinnedBuildAndAssignCondenates(ccoords, betas, row_splits,
                              radius=0.8, min_beta=0.1,
                              dist=None,
                              soft=False,
                              assign_radius=None):
    if soft:
        raise NotImplementedError('Adjust code for soft and try again')
    if row_splits[1] != len(ccoords):
        raise NotImplementedError('Adjust for row splits and try again')

    if dist is None:
        dist = np.ones_like(betas)

    assignment, pred_shower_alpha_idx = build_condensates_cpu(ccoords, betas, dist, beta_threshold=min_beta, dist_threshold=radius)

    return assignment, pred_shower_alpha_idx

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
    
    c_point_idx,_ = tf.unique(asso)
    asso_idx = AssignToCondensates(ccoords,
                     c_point_idx,
                     row_splits, 
                     radius=assign_radius, 
                     dist=dist)
    
    return asso_idx, iscond, ncond



  
