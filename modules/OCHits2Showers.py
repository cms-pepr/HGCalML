import os.path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from numba import njit
from bin_by_coordinates_op import BinByCoordinates

def reconstruct_showers_cond_op(cc, beta, beta_threshold=0.5, dist_threshold=0.5, limit=500, return_alpha_indices=False,
                                pred_dist=None, max_hits_per_shower=-1,
                                soft=False):
    from condensate_op import BuildCondensates

    asso, iscond, ncond = BuildCondensates(
        tf.convert_to_tensor(tf.convert_to_tensor(cc)),
        tf.convert_to_tensor(tf.convert_to_tensor(beta[..., np.newaxis])),
        row_splits=tf.convert_to_tensor(np.array([0, len(cc)], np.int32)),
        dist=tf.convert_to_tensor(pred_dist[..., np.newaxis]) if pred_dist is not None else None,
        min_beta=beta_threshold,
        radius=dist_threshold,
        soft=soft)

    asso = asso.numpy().tolist()
    iscond = iscond.numpy()

    pred_shower_alpha_idx = np.argwhere(iscond==1)[:, 0].tolist()

    map_fn = {x:i for i,x in enumerate(pred_shower_alpha_idx)}
    map_fn[-1] = -1
    pred_sid = [map_fn[x] for x in asso]
    return pred_sid, pred_shower_alpha_idx


def reconstruct_showers(cc, beta, beta_threshold=0.5, dist_threshold=0.5, pred_dist=None):
    from assign_condensate_op import BuildAndAssignCondensates

    asso, iscond, _ = BuildAndAssignCondensates(
        tf.convert_to_tensor(tf.convert_to_tensor(cc)),
        tf.convert_to_tensor(tf.convert_to_tensor(beta)),
        row_splits=tf.convert_to_tensor(np.array([0, len(cc)], np.int32)),
        dist=tf.convert_to_tensor(pred_dist) if pred_dist is not None else None,
        min_beta=beta_threshold,
        radius=dist_threshold)


    asso = asso.numpy().tolist()
    iscond = iscond.numpy()

    pred_shower_alpha_idx = np.argwhere(iscond == 1)[:, 0].tolist()

    map_fn = {x: i for i, x in enumerate(pred_shower_alpha_idx)}
    map_fn[-1] = -1
    pred_sid = [map_fn[x] for x in asso]

    return np.array(pred_sid)[:, np.newaxis], pred_shower_alpha_idx

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

        # print(bin_vector)

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

@njit
def collect_func_native(ix_l, ix_h, iy_l, iy_h, iz_l, iz_h, nbins, assignment, coords, alpha_coords, alpha_radius,
                        shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full):
    x = 0
    for i in range(ix_l, ix_h):
        if i < 0 or i >= nbins:
            continue
        for j in range(iy_l, iy_h):
            if j < 0 or j >= nbins:
                continue
            for k in range(iz_l, iz_h):
                if k < 0 or k >= nbins:
                    continue
                b_flat = i * nbins ** 2 + j * nbins + k
                start_index = row_splits[b_flat]
                end_index = row_splits[b_flat + 1]
                for l in range(start_index, end_index):
                    if assignment[l] == -1:
                        if np.sum((coords[l] - alpha_coords) ** 2) < alpha_radius ** 2:
                            assignment[l] = shower_idx
                            beta[l] = 0
                            x += 1
                            if beta_filtered_indices_full[l] != -1:
                                beta_filtered[beta_filtered_indices_full[l]] = 0
    return x



def reconstruct_showers_binned(coords, beta, beta_threshold=0.3, dist_threshold=1.5, pred_dist=None):
    beta = beta[:, 0]
    coords = coords - np.min(coords, axis=0, keepdims=True)

    if pred_dist is None:
        pred_dist = 0 * beta + 1.0

    pred_dist = pred_dist[:, 0] * dist_threshold

    _, bins_flat, n_bins, bin_width, _ = BinByCoordinates(coords, [0, len(coords)], n_bins=30)

    bins_flat = bins_flat.numpy()
    n_bins = n_bins.numpy()
    bin_width = float(bin_width[0])

    bin_width_x = bin_width_y = bin_width_z = bin_width

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

    bin_width = np.array([bin_width_x, bin_width_y, bin_width_z])
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
    pred_shower_alpha_idx = pred_shower_alpha_idx[0:shower_idx]
    pred_shower_alpha_idx = np.array([sorting_indices[i] for i in pred_shower_alpha_idx])

    return assignment_2[..., np.newaxis], pred_shower_alpha_idx[0:shower_idx]



def reconstruct_showers_no_op(cc, beta, beta_threshold=0.5, dist_threshold=0.5, pred_dist=None, max_hits_per_shower=-1):
    beta = beta[:, 0]
    pred_dist = None if pred_dist is None else pred_dist[:, 0]
    beta_filtered_indices = np.argwhere(beta > beta_threshold)
    beta_filtered = np.array(beta[beta_filtered_indices])
    beta_filtered_remaining = beta_filtered.copy()
    cc_beta_filtered = np.array(cc[beta_filtered_indices])
    pred_sid = beta * 0 - 1
    pred_sid = pred_sid.astype(np.int32)

    max_index = 0
    alpha_indices = []

    while np.sum(beta_filtered_remaining) > 0:
        alpha_index = beta_filtered_indices[np.argmax(beta_filtered_remaining)]
        cc_alpha = cc[alpha_index]
        # print(cc[alpha_index].shape, cc.shape)
        dists = np.sqrt(np.sum((cc - cc_alpha) ** 2, axis=-1))

        this_threshold = dist_threshold * (1 if pred_dist is None else pred_dist[alpha_index])

        # if pred_dist is None:

        if max_hits_per_shower != -1:
            filtered = dists.copy()
            filtered[dists > dist_threshold] = 1000000.
            filtered[pred_sid != -1] = 1000000.
            filtered[filtered.argsort()[max_hits_per_shower:len(filtered)]] = 1000000.
            picked = filtered < 1000000.
            pred_sid[picked] = max_index
        else:
            pred_sid[np.logical_and(dists < this_threshold, pred_sid == -1)] = max_index

        max_index += 1

        dists_filtered = np.sqrt(np.sum((cc_alpha - cc_beta_filtered) ** 2, axis=-1))
        beta_filtered_remaining[dists_filtered < this_threshold] = 0
        alpha_indices.append(alpha_index[0])


    return pred_sid[:, np.newaxis], alpha_indices


class OCHits2Showers():
    def __init__(self, beta_threshold, distance_threshold, is_soft, with_local_distance_scaling, reco_method='binned'):
        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.is_soft = is_soft
        self.with_local_distance_scaling = with_local_distance_scaling
        self.op = reco_method
        if type(self.op) is bool:
            self.op = 'condensate_op'

    def set_beta_threshold(self, beta_threshold):
        self.beta_threshold = beta_threshold

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def call(self, features_dict, pred_dict):
        # with tf.device('/cpu'):
        return self.priv_call(features_dict, pred_dict)

    def priv_call(self, features_dict, pred_dict):
        if self.op == 'condensate_op':
            pred_sid, pred_shower_alpha_idx = reconstruct_showers(pred_dict['pred_ccoords'],
                                                                  pred_dict['pred_beta'],
                                                                  self.beta_threshold,
                                                                  self.distance_threshold,
                                                                  pred_dist=pred_dict['pred_dist'] if self.with_local_distance_scaling else None)
        elif self.op =='numpy':
            pred_sid, pred_shower_alpha_idx = reconstruct_showers_no_op(pred_dict['pred_ccoords'],
                                                                  pred_dict['pred_beta'],
                                                                  self.beta_threshold,
                                                                  self.distance_threshold,
                                                                  pred_dist=pred_dict['pred_dist'] if self.with_local_distance_scaling else None)
        elif self.op=='binned':
            pred_sid, pred_shower_alpha_idx = reconstruct_showers_binned(pred_dict['pred_ccoords'],
                                                                        pred_dict['pred_beta'],
                                                                        self.beta_threshold,
                                                                        self.distance_threshold,
                                                                        pred_dist=pred_dict[
                                                                            'pred_dist'] if self.with_local_distance_scaling else None)
        else:
            raise KeyError('%s reco method not recognized'%self.op)

        processed_pred_dict = dict()
        processed_pred_dict['pred_sid'] = pred_sid
        processed_pred_dict['pred_energy'] = np.zeros_like(processed_pred_dict['pred_sid'], np.float)


        # This uses correction factor based on alpha idx
        # pred_shower_energy = (tf.math.unsorted_segment_sum(features_dict['recHitEnergy'][:, 0], pred_sid[:, 0], num_segments=len(pred_shower_alpha_idx)) * tf.gather(pred_dict['pred_energy_corr_factor'][:, 0], pred_shower_alpha_idx)).numpy()
        # This uses correction factor independent for different hits
        pred_shower_energy = (tf.math.unsorted_segment_sum(features_dict['recHitEnergy'][:, 0] * pred_dict['pred_energy_corr_factor'][:, 0], pred_sid[:, 0], num_segments=len(pred_shower_alpha_idx))).numpy()
        pred_shower_energy_ = np.concatenate(([0], pred_shower_energy), axis=0)
        pred_energy = tf.gather(pred_shower_energy_, pred_sid[:,0] + 1)

        processed_pred_dict['pred_energy'] = pred_energy[..., np.newaxis]
        # pred_shower_energy_2_ = [0]
        # for idx in pred_shower_alpha_idx:
        #     filter = (processed_pred_dict['pred_sid']==pred_sid[idx])[:,0]
        #     this_energy = np.sum(pred_dict['pred_energy_corr_factor'][filter] * features_dict['recHitEnergy'][filter])
        #     processed_pred_dict['pred_energy'][filter] \
        #         = this_energy
        #
        #     pred_shower_energy_2_ = pred_shower_energy_2_ + [this_energy]
        #
        #     # print("Checking x", np.sum(pred_energy[filter]), this_energy)
        #
        # print("V", pred_shower_energy_, np.array(pred_shower_energy_2_))
        # print("Checking", np.sum(np.abs(processed_pred_dict['pred_energy'][:, 0]-pred_energy)))

        if 'pred_energy_unc' in processed_pred_dict:
            processed_pred_dict['pred_energy_unc'] \
                = 0.5*(pred_dict['pred_energy_high_quantile']-pred_dict['pred_energy_low_quantile'])

        processed_pred_dict.update(pred_dict)
        processed_pred_dict.pop('pred_beta')
        processed_pred_dict['pred_id'] = np.argmax(processed_pred_dict['pred_id'], axis=1)[:, np.newaxis]

        return processed_pred_dict, pred_shower_alpha_idx
