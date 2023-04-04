import time

import numpy as np
import tensorflow as tf

class OCHits2Showers():
    pass


class OCGatherEnergyCorrFac(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac, self).__init__(**kwargs)

    def call(self, pred_sid,pred_corr_factor, rechit_energy, row_splits=None):

        if row_splits is None:
            row_splits = tf.constant([0, rechit_energy.shape[0]])

        pred_sid_p1 = pred_sid +1 # For the -1 label for noise
        pred_corr_factor = tf.where(pred_sid==-1, tf.zeros_like(pred_corr_factor, tf.float32), pred_corr_factor)

        # Second term is a factor to make all the pred_sids unique
        unique_segments = pred_sid_p1[:, 0] + (tf.reduce_max(pred_sid_p1)+1)*tf.ragged.row_splits_to_segment_ids(row_splits, out_type=tf.int32)
        pred_shower_energy = (tf.math.unsorted_segment_sum(rechit_energy[:, 0] * pred_corr_factor[:, 0], unique_segments, num_segments=(tf.reduce_max(unique_segments)+1)))

        pred_energy = tf.gather(pred_shower_energy, unique_segments)

        return pred_energy[:, tf.newaxis]

class OCHits2ShowersLayer(tf.keras.layers.Layer):
    def __init__(self, beta_threshold, distance_threshold, use_local_distance_thresholding=True, assign_by_max_beta=True, nbinning_dims=3,**kwargs):
        super(OCHits2ShowersLayer, self).__init__(**kwargs)

        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.assign_by_max_beta = assign_by_max_beta
        self.use_local_distance_thresholding = use_local_distance_thresholding
        self.nbinning_dims=nbinning_dims

    def set_beta_threshold(self, b):
        self.beta_threshold = b

    def set_distance_threshold(self, d):
        self.distance_threshold = d

    def get_config(self):
        base_config = super(self, self).get_config()
        return dict(list(base_config.items()) + list({'beta_threshold': self.beta_threshold ,
                                                      'distance_threshold': self.distance_threshold,
                                                      'use_local_distance_thresholding': self.use_local_distance_thresholding,
                                                      'assign_by_max_beta': self.assign_by_max_beta,
                                                      }.items()))


    def call(self, pred_ccoords, pred_beta, pred_dist, no_condensation_mask=None, row_splits=None):
        if row_splits is None:
            row_splits = tf.constant([0,pred_dist.shape[0]], tf.int32)

        from assign_condensate_op import BuildAndAssignCondensatesBinned

        if not self.use_local_distance_thresholding:
            pred_dist = np.ones_like(pred_dist)

        # t1 = time.time()
        x = BuildAndAssignCondensatesBinned(
            pred_ccoords,
            pred_beta,
            no_condensation_mask=no_condensation_mask,
            row_splits=row_splits,
            dist=pred_dist*self.distance_threshold,
            assign_by_max_beta=self.assign_by_max_beta,
            beta_threshold=self.beta_threshold,
            nbin_dims=self.nbinning_dims)
        # print("New one took", time.time()-t1,"seconds")

        return x



def process_endcap(hits2showers_layer, energy_gather_layer, features_dict, predictions_dict):
    pred_sid, _, alpha_idx, _, ncond = hits2showers_layer(
            predictions_dict['pred_ccoords'],
            predictions_dict['pred_beta'],
            predictions_dict['pred_dist'])
    pred_sid = pred_sid.numpy()
    alpha_idx = alpha_idx.numpy()

    processed_pred_dict = dict()
    processed_pred_dict['pred_sid'] = pred_sid

    pred_energy = energy_gather_layer(
            pred_sid, 
            predictions_dict['pred_energy_corr_factor'],  
            features_dict['recHitEnergy'])
    processed_pred_dict['pred_energy'] = pred_energy.numpy()

    if 'pred_energy_high_quantile' in predictions_dict.keys():
        processed_pred_dict['pred_energy_unc'] \
            = 0.5 * (predictions_dict['pred_energy_high_quantile'] - predictions_dict['pred_energy_low_quantile'])
        processed_pred_dict.update(predictions_dict)
        processed_pred_dict.pop('pred_beta')
        processed_pred_dict['pred_id'] = np.argmax(
                processed_pred_dict['pred_id'], 
                axis=1)[:, np.newaxis]

    return processed_pred_dict, alpha_idx

