import numpy as np
import pandas as pd
import tensorflow as tf


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
    def __init__(self, beta_threshold, distance_threshold, is_soft, with_local_distance_scaling, op):
        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.is_soft = is_soft
        self.with_local_distance_scaling = with_local_distance_scaling
        self.op = op

    def set_beta_threshold(self, beta_threshold):
        self.beta_threshold = beta_threshold

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def call(self, features_dict, pred_dict):
        """

        :param event_data: (features_dict, truth_dict, predictions_dict) coming from HGCal predictor for instance
        :return: Pred sid
        """

        if self.op:
            pred_sid, pred_shower_alpha_idx = reconstruct_showers(pred_dict['pred_ccoords'],
                                                                  pred_dict['pred_beta'],
                                                                  self.beta_threshold,
                                                                  self.distance_threshold,
                                                                  pred_dist=pred_dict['pred_dist'] if self.with_local_distance_scaling else None)
        else:
            pred_sid, pred_shower_alpha_idx = reconstruct_showers_no_op(pred_dict['pred_ccoords'],
                                                                  pred_dict['pred_beta'],
                                                                  self.beta_threshold,
                                                                  self.distance_threshold,
                                                                  pred_dist=pred_dict['pred_dist'] if self.with_local_distance_scaling else None)

        processed_pred_dict = dict()
        processed_pred_dict['pred_sid'] = pred_sid
        processed_pred_dict['pred_energy'] = np.zeros_like(processed_pred_dict['pred_sid'], np.float)

        for idx in pred_shower_alpha_idx:
            filter = (processed_pred_dict['pred_sid']==pred_sid[idx])[:,0]
            processed_pred_dict['pred_energy'][filter] \
                = np.sum(pred_dict['pred_energy_corr_factor'][filter] * features_dict['recHitEnergy'][filter])

        processed_pred_dict.update(pred_dict)
        processed_pred_dict.pop('pred_beta')
        processed_pred_dict['pred_id'] = np.argmax(processed_pred_dict['pred_id'], axis=1)[:, np.newaxis]

        return processed_pred_dict, pred_shower_alpha_idx


