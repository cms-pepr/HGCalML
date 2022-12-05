import math

import numpy as np
import tensorflow as tf
from bin_by_coordinates_op import BinByCoordinates

_bc_op_binned = tf.load_op_library('binned_assign_to_condensates.so')

def cumsum_ragged(tensor, exclusive=False):
    v = tf.RaggedTensor.from_row_splits(tf.cumsum(tensor.values, exclusive=exclusive), tensor.row_splits)
    w = tf.concat(([[0]], tf.reduce_max(v, axis=1)[0:-1]), axis=0)[:, tf.newaxis]
    v = v - w
    return v


def merge_noise_as_indiv_cp(assignment, asso_idx, alpha_idx, is_cond, n_condensates, row_splits):
    #assignment first
    r_orig_assignment = tf.RaggedTensor.from_row_splits(assignment, row_splits)
    assignment = r_orig_assignment
    noise_mask = assignment < 0
    x = cumsum_ragged(tf.cast(noise_mask, tf.int32))
    x = tf.where(noise_mask, x, 0)
    assignment = tf.where(noise_mask, x-1, assignment+tf.reduce_max(x, axis=1, keepdims=True))
    
    assignment = assignment.values #back to flat
    
    #-1 has been added in the front of every row split
    
    allrange = tf.range(tf.shape(asso_idx)[0])
    
    rallrange = tf.RaggedTensor.from_row_splits(allrange, row_splits)
    rallrange = tf.ragged.boolean_mask(rallrange, r_orig_assignment[:,:,0]<0)
    
    alpha_idx = tf.concat([rallrange, tf.RaggedTensor.from_row_splits(alpha_idx, n_condensates)],axis=1)
    alpha_idx = alpha_idx.values
    
    #print('>>>',asso_idx,allrange, is_cond)
    is_cond = tf.where(asso_idx<0, True, is_cond[:,0])
    asso_idx = tf.where(asso_idx<0, allrange, asso_idx)
    
    r_is_cond = tf.RaggedTensor.from_row_splits(is_cond, row_splits)
    n_condensates = tf.concat([n_condensates[0:1], tf.cumsum(tf.reduce_sum(tf.cast(r_is_cond, dtype='int32'), axis=1),axis=0)],axis=0)
    
    return assignment, asso_idx, alpha_idx, is_cond[...,tf.newaxis], n_condensates
    

def calc_ragged_shower_indices(assignment, row_splits,
                               gather_noise=True):
    """

    :param assignment: [nvert, 1] Values should be consecutive i.e. -1,0,1,2,3,...N. Same is true for the every segment
                        i.e. for the second segment it should restart from -1,0,1,2,3...N. This is the first return
                        value of BuildAndAssignCondensatesBinned.
    :param row_splits: [nvert+1], row splits
    :param gather_noise: boolean, whether to gather noise or not. If set to True, the first element in the shower
                         dimension will always correspond to the noise even if there is no noise vertex present.

    :return: a double ragged tensor of indices, first ragged dimension iterates endcaps/samples and the second, showers
             in that endcap. This can be used in gather_nd as follows:
                ragged_indices = calc_ragged_shower_indices(assignment, row_splits, gather_noise=False)
                x = tf.gather_nd(assignment, ragged_indices)

    """

    if gather_noise:
        # move up one if there is any noise a row split
        assignment = tf.RaggedTensor.from_row_splits(assignment, row_splits)
        assignment = assignment - tf.reduce_min(assignment, axis=1, keepdims=True)

        assignment = assignment.values
        assignment = assignment[:, 0]
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


    return sorting_indices_showers_ragged
    
# FIXME, there seems to be some sort of squeeze happening for '1' ragged dimensions?    
def calc_ragged_cond_indices(assignment, alpha_idx, n_condensates, row_splits, 
                           gather_noise=True, return_reverse=True):
    '''
    :param assignment: assignment indices from BuildAndAssignCondensatesBinned
    :param alpha_idx: indices of condensation points (output of BuildAndAssignCondensatesBinned)
           important: is must be guaranteed that tf.gather_nd(assignment, alpha_idx[...,tf.newaxis]) is 
           [0,1,2,3,4...., (row split), 0,1,2,3,4...]
    :param n_condensates: number of condensates per row split (output of BuildAndAssignCondensatesBinned)
                          noise is not counted here
    
    :param row_splits: full row_splits
    
    :param return_reverse: returns the reverse operation to gather back. Example (x is some point feature vector [V x F]):
                           cidx, revidx = calc_ragged_cond_indices...
                           xc = tf.gather_nd(x, cidx)
                           xg = tf.gather_nd(xc, revidx) # non-ragged representation [V x F], 
                                                         # with the original row splits
                           
                           
    :param return_flat_reverse: also returns flat reverse indices in addition. Only effective if
                                return_reverse==True.
                                Example:
                                cidx, revidx, flat_revidx = calc_ragged_cond_indices...
                                
                                xc = tf.gather_nd(x, idx) #ragged [B, ragged, x.shape[-1]]
                                xc = xc.values        #not ragged [B' , x.shape[-1]]
                                xg = tf.gather_nd(xc, flat_revidx) #reverse operation [B , x.shape[-1]]
                                
    :return: indices to gather condensation point features into a ragged tensor of structure: 
             [event, condensation point, F]
             The first entry for each event will be a dummy entry for noise to have parallel structures
             w.r.t. calc_ragged_shower_indices
    '''
    
    alpha_exp = alpha_idx[...,tf.newaxis]
    
    adapted_assignment = tf.RaggedTensor.from_row_splits(assignment, row_splits)
    n_nonoise_condensates =  tf.reduce_max(adapted_assignment, axis=1)[:,0] + 1
    n_nonoise_condensates = tf.concat([n_nonoise_condensates[0:1]*0, 
                                       tf.cumsum(n_nonoise_condensates, axis=0)],axis=0)
    
    # fails with 
    # tf.Tensor([   0 1437 2453 3846 4948], shape=(5,), dtype=int32) tf.Tensor([0 2 2 2 2], shape=(5,), dtype=int32)
    #print(n_condensates, n_nonoise_condensates)
    #print(f'{n_nonoise_condensates}, \n {alpha_exp}, {alpha_exp.shape},\n{assignment}, {row_splits}')
    rc_ass = tf.RaggedTensor.from_row_splits(alpha_exp, n_nonoise_condensates)
    
    #if row_splits.shape[0] is None: #no point to look for noise in nothing
    #    if return_reverse:
    #        return rc_ass, adapted_assignment.values,\
    #             tf.reduce_sum(adapted_assignment, axis=-1, keepdims=True).values
    #    return rc_ass
    
    if gather_noise:
        
        noise_idx = tf.RaggedTensor.from_row_splits(assignment, row_splits)
        #this is a workaround because argmin/max does not work on ragged
        selidx = tf.RaggedTensor.from_row_splits(tf.range(tf.shape(assignment)[0]), row_splits)
        selidx = tf.expand_dims(selidx, axis=-1)
        selidx = tf.where(noise_idx < 0, selidx, -2)
        noise_idx = tf.reduce_max(selidx, axis=1)
        noise_idx = tf.RaggedTensor.from_row_splits(noise_idx, tf.range(tf.shape(noise_idx)[0]+1))#one per rs
        sel = noise_idx[...,0] >= 0
        noise_idx = tf.ragged.boolean_mask(noise_idx, sel)[...,0]
        #end workaround
        noise_idx = tf.expand_dims(noise_idx, axis=-1)
        rc_ass = tf.concat([noise_idx, rc_ass],axis=1)
        
    if return_reverse:
        
        adapted_assignment = adapted_assignment - tf.reduce_min(adapted_assignment, axis=1, keepdims=True)
        n_condensates = tf.reduce_max(adapted_assignment, axis=1)[:,0] + 1
        n_condensates = tf.concat([n_condensates[0:1]*0, tf.cumsum(n_condensates, axis=0)],axis=0)
    
        #this needs to be done ragged; FIXME
        revidx = adapted_assignment
        nrs = tf.shape(row_splits)[0]-1
        addrs = tf.range(nrs)[...,tf.newaxis][...,tf.newaxis]
        revidx = tf.concat([0 * revidx + addrs,revidx],axis=-1)
        
        add = n_condensates[:-1]
        add = add[...,tf.newaxis][...,tf.newaxis]
        mul = tf.concat([0*add, 0*add + 1],axis=-1)
        add = tf.concat([add, 0*add],axis=-1)
        maskedrevidx = revidx * mul
        flat_rev = tf.reduce_sum(maskedrevidx+add, axis=-1, keepdims=True)
        flat_rev = flat_rev.values #remove ragged
        return rc_ass, revidx.values, flat_rev
        
    return rc_ass
    

def collapse_ragged_noise(ch_input, c_assignement_index):
    '''
    Requires that, if there is noise, it is the first entry in axis 1 
    (guaranteed by all other ops here)
    
    :param ch_input: ragged input in format [e, ragged condensation points, ragged hits, F]
    :param c_assignement_index: condensation point assignment index (-1 for noise)
                                [e, ragged condensation points, 1]
                                
    :return: same as input, but noise reduced to one hit in hit dimension, and that hit is zeroed out
    '''
    
    sel = c_assignement_index < 0
    noisepart = tf.ragged.boolean_mask(ch_input, sel)[:,:,:1]
    nonoisepart = tf.ragged.boolean_mask(ch_input, tf.logical_not(sel))
    return tf.concat([noisepart, nonoisepart], axis=1)
    


def BuildAndAssignCondensatesBinned(ccoords,
                        betas,
                        dist,
                        row_splits,
                        beta_threshold,
                        no_condensation_mask = None,
                        distance_threshold = 1.,
                        assign_by_max_beta=False,
                        nbin_dims=3,
                        keep_noise=False,
                        ):
    """
    :param ccoords: [nvert,ndim] Clustering coordinates of shape
    :param betas: [nvert, 1] Betas of shape
    :param dist: [nvert, 1] Local distance thresholds of shape
    :param row_splits: Row splits of shape [nvert+1]
    :param beta_threshold: Minimum beta threshold (scalar)
    :param no_condensation_mask: 1 for each input that should become its own condensation point, 
                                 right now, this uses way more resources than it needs to
    :param nbin_dims <= ccoords.shape[1]. Binning to be done in these many number of coordinates (scalar)
    :param assign_by_max_beta: Boolean. If set to True, the higher beta vertex eats up all the vertices in its radius. If set to
             False, the assignment for a vertex is done to the condensate it is the closest to. True behaves like
             anti-kt jet clustering algorighm while False behaves like XCone.
    :param keep_noise: If True, "noise" hits that remain after clustering will become one condensation point each. If False,
                       they will be merged into the first 'shower' entry in each ragged index tensor
                        
    :return: 5 elements: (in order)
    1. assignment: [nvert,1] Assignment in ascending order from 0,1,2,3...N. Resets at every ragged segment.
                   tf.gather_nd(assignment, alpha_idx[...,tf.newaxis]) is guaranteed to be [0,1,2,3,4...., (row split), 0,1,2,3,4...]
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

    if no_condensation_mask is None:
        no_condensation_mask = tf.zeros_like(betas, tf.int32)

    # if no_condensation_mask is not None:
    #     assert len(no_condensation_mask.shape) == 2
    #     betas = tf.where(no_condensation_mask>0, 1., betas)
    #     maxcoords = tf.reduce_max(ccoords, axis=0, keepdims=True) # 1 x C , just to know where to start
    #     fidx = tf.cast(tf.range(tf.shape(ccoords)[0]) + 2, dtype='float32')
    #     replcoords = ccoords + tf.expand_dims(fidx,axis=1) * maxcoords * 1.05 * tf.reduce_max(dist)
    #     ccoords = tf.where( no_condensation_mask>0, replcoords, ccoords )


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
    no_condensation_mask = tf.gather(no_condensation_mask, sorting_indices)
    bins_flat = tf.gather(bins_flat, sorting_indices)
    orig_indices = tf.gather(tf.range(betas.shape[0]), sorting_indices)
    bin_splits = tf.ragged.segment_ids_to_row_splits(bins_flat,num_segments=tf.reduce_prod(n_bins)*num_rows)

    # _h is for high beta vertices, filtered ones for faster performance
    high_beta_indices = (tf.logical_or(betas>beta_threshold, no_condensation_mask>0))[..., 0]
    row_splits_h = tf.ragged.segment_ids_to_row_splits(tf.ragged.row_splits_to_segment_ids(row_splits)[high_beta_indices], num_segments=num_rows)
    betas_h = betas[high_beta_indices]
    ccoords_h = ccoords[high_beta_indices]
    dist_h = dist[high_beta_indices]
    no_condensation_mask_h = no_condensation_mask[high_beta_indices]
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
                no_condensation_mask=no_condensation_mask,
                bins_flat=bins_flat,
                bin_splits=bin_splits,
                n_bins=n_bins,
                bin_widths=bin_width,
                indices_to_filtered = indices_to_filtered,
                original_indices=orig_indices_h,
                ccoords_h=ccoords_h,
                dist_h=dist_h,
                beta_h=betas_h,
                no_condensation_mask_h=no_condensation_mask_h,
                row_splits=row_splits,
                row_splits_h=row_splits_h)

    assignment = tf.gather(assignment, sorting_indices_back)
    asso = tf.gather(asso, sorting_indices_back)

    pred_shower_alpha_idx = alpha_indices[alpha_indices!=-1]
    is_cond = tf.cast(tf.scatter_nd(pred_shower_alpha_idx[:, tf.newaxis], tf.ones(pred_shower_alpha_idx.shape[0]), [betas.shape[0]]), tf.bool)
    
    #switch to standard format
    assignment, asso_idx, alpha_idx, is_cond, n_condensates = assignment[:, tf.newaxis], asso, pred_shower_alpha_idx, is_cond[:, np.newaxis], n_condensates
    
    if keep_noise:
        o = merge_noise_as_indiv_cp(assignment, asso_idx, alpha_idx, is_cond, n_condensates, row_splits)
        assignment, asso_idx, alpha_idx, is_cond, n_condensates = o


    #sanity check
    try:
        adapted_assignment = tf.RaggedTensor.from_row_splits(assignment, row_splits)
        n_nonoise_condensates =  tf.reduce_max(adapted_assignment, axis=1)[:,0] + 1
        n_nonoise_condensates = tf.concat([n_nonoise_condensates[0:1]*0, 
                                           tf.cumsum(n_nonoise_condensates, axis=0)],axis=0)
    
        # fails with 
        # tf.Tensor([   0 1437 2453 3846 4948], shape=(5,), dtype=int32) tf.Tensor([0 2 2 2 2], shape=(5,), dtype=int32)
        tf.assert_equal(n_condensates, n_nonoise_condensates)
    
    except Exception as e:
        import pickle
        with open('BuildAndAssignCondensatesBinned_error.pkl','wb') as f:
            pickle.dump(
                {'assignment': assignment.numpy(),
                 'asso_idx': asso_idx.numpy(),
                 'alpha_idx': alpha_idx.numpy(),
                 'is_cond': is_cond.numpy(),
                 'n_condensates': n_condensates.numpy(),
                 'from_assignment_n_condensates': n_nonoise_condensates.numpy(),
                 'row_splits': row_splits.numpy(),
                 #input
                 'ccoords' : ccoords.numpy(),
                 'betas' : betas.numpy(),
                 'dist':dist.numpy(),
                 'beta_threshold':beta_threshold,
                 
                 'assign_by_max_beta':assign_by_max_beta,
                 'no_condensation_mask':no_condensation_mask,
                 'keep_noise': keep_noise,
                 
                 'bins_flat':bins_flat, 
                 'n_bins':n_bins,
                 'bin_width':bin_width         
                    },f)
        raise e


    return assignment, asso_idx, alpha_idx, is_cond, n_condensates



# @ops.RegisterGradient("AssignToCondensates")
# def _AssignToCondensatesGrad(op, asso_grad):
#     return [None, None, None, None]


#### convenient helpers, not the OP itself





