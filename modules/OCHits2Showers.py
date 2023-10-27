import pdb
import sys
import time

import numpy as np
import tensorflow as tf
import hdbscan

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


class OCGatherEnergyCorrFac3(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac2, self).__init__(**kwargs)

    def call(self, pred_sid,pred_corr_factor, rechit_energy, no_noise_idx,
            pred_beta, is_track=None, row_splits=None,
            return_tracks_where_possible=False, return_tracks=False, raw=False):
        """
        Same as `OCGatherEnergyCorrFac` with the addition that one can chose if the energy
        reconstructed with the tracks or the enery reconstructed by the calorimeter should be used.
        Shapes:
            * rechit_energy, is_track:      [N_orig, 1]     (before noise filter)
            * pred_sid, pred_corr_factor:   [N_filtered, 1] (after noise filter)
            * no_noise_idx:                 [N_filtered, 1] (with indices up to N_orig as entries)
        Returns:
            Will always return only one value:
                return_tracks_where_possible overrides `return_tracks`
            By default returns hit energy
        """

        is_track = tf.cast(is_track, tf.int32)
        is_track = tf.reshape(tf.gather(is_track, no_noise_idx), (-1,1))            # Shape [N_filtered, 1]
        rechit_energy = tf.reshape(tf.gather(rechit_energy, no_noise_idx), (-1,1))  # Shape [N_filtered, 1]

        if row_splits is None:
            row_splits = tf.constant([0, rechit_energy.shape[0]])
        else:
            print("WARNING OCGatherEnergyCorrFac2: \nI am not sure if this works with explicit row splits")


        pred_sid_p1 = pred_sid +1 # For the -1 label for noise
        pred_corr_factor = tf.where(pred_sid==-1, tf.zeros_like(pred_corr_factor, tf.float32), pred_corr_factor)

        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy))
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy))

        # Second term is a factor to make all the pred_sids unique
        # In case of only one sample (i.e. rowsplits [0, N_filtered]) this is identical to `pred_sid_p1`
        # Shape: [N_filtered,] ! Different from other shapes
        unique_segments = pred_sid_p1[:, 0] +\
                (tf.reduce_max(pred_sid_p1)+1) * tf.ragged.row_splits_to_segment_ids(row_splits, out_type=tf.int32)

        track_ids = pred_sid[is_track == 1]
        track_uniques, track_ids, track_counts = tf.unique_with_counts(track_ids)
        track_duplicates = track_uniques[track_counts > 1]
        # track_duplicates now is a list that inclused the `pred_sid` where two tracks
        # are assigned to the same `pred_sid`
        duplicate_tensor = tf.logical_and(
                tf.equal(tf.expand_dims(pred_sid, axis=-1), track_duplicates),
                tf.expand_dims(is_track==1, axis=-1)
                )
        beta_exp = tf.reshape(
                tf.repeat(pred_beta, repeats=track_duplicates.shape[0]),
                (-1, 1, track_duplicates.shape[0])
                )
        beta_check = tf.where(duplicate_tensor, beta_exp, tf.zeros_like(beta_exp))
        # beta_check: shape (Nbatch, 1, n_duplicates)
        # all values are zero except for
        track_select = tf.argmax(beta_check, axis=0)
        # track_select now includes the information of which index to chose for the pred_sid
        # that are listed in `track_duplicates`
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        pred_shower_tracks = (
                tf.math.unsorted_segment_sum(
                    e_track[:, 0] * pred_corr_factor[:, 0],
                    unique_segments,
                    num_segments=(tf.reduce_max(unique_segments)+1))
                )
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)
        # This returns the summed up energy with the and has length n_showers
        # we need to replace the entries that have duplicate tracks with the chosen
        # track's energy (multiplied by the corresponding correction factor)
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)
        # This is the updating

        pred_shower_hits = (
                tf.math.unsorted_segment_sum(
                    e_hit[:, 0] * pred_corr_factor[:, 0],
                    unique_segments,
                    num_segments=(tf.reduce_max(unique_segments)+1))
                )

        pred_energy_hits = tf.gather(pred_shower_hits, unique_segments)
        pred_energy_tracks = tf.gather(pred_shower_tracks, unique_segments)

        if return_tracks_where_possible:
            pred_energy = tf.where(
                    pred_energy_tracks!=0.,
                    pred_energy_tracks,
                    pred_energy_hits)
            return pred_energy[:, tf.newaxis]
        elif return_tracks:
            return pred_energy_tracks[:, tf.newaxis]
        else:
            return pred_energy_hits[:, tf.newaxis]



class OCGatherEnergyCorrFac2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac2, self).__init__(**kwargs)

    def call(self, pred_sid,pred_corr_factor, rechit_energy, no_noise_idx,
            pred_beta, is_track=None, row_splits=None,
            return_tracks_where_possible=True, return_tracks=False, raw=False):
        """
        Same as `OCGatherEnergyCorrFac` with the addition that one can chose if the energy
        reconstructed with the tracks or the enery reconstructed by the calorimeter should be used.
        Shapes:
            * rechit_energy, is_track:      [N_orig, 1]     (before noise filter)
            * pred_sid, pred_corr_factor:   [N_filtered, 1] (after noise filter)
            * no_noise_idx:                 [N_filtered, 1] (with indices up to N_orig as entries)
        Returns:
            Will always return only one value:
                return_tracks_where_possible overrides `return_tracks`
            By default returns hit energy
        """

        is_track = tf.cast(is_track, tf.int32)
        # is_track = tf.reshape(tf.gather(is_track, no_noise_idx), (-1,1)) # Shape [N_filtered, 1]
        # rechit_energy = tf.reshape(tf.gather(rechit_energy, no_noise_idx), (-1,1))  # Shape [N_filtered, 1]

        if row_splits is None:
            row_splits = tf.constant([0, rechit_energy.shape[0]])
        else:
            print("WARNING OCGatherEnergyCorrFac2: \nI am not sure if this works with explicit row splits")


        pred_sid_p1 = pred_sid +1 # For the -1 label for noise
        pred_corr_factor = tf.where(pred_sid==-1, tf.zeros_like(pred_corr_factor, tf.float32), pred_corr_factor)

        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy))
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy))

        # Second term is a factor to make all the pred_sids unique
        # In case of only one sample (i.e. rowsplits [0, N_filtered]) this is identical to `pred_sid_p1`
        # Shape: [N_filtered,] ! Different from other shapes
        unique_segments = pred_sid_p1[:, 0] +\
                (tf.reduce_max(pred_sid_p1)+1) * tf.ragged.row_splits_to_segment_ids(row_splits, out_type=tf.int32)

        track_ids = pred_sid[is_track == 1]
        track_uniques, track_ids, track_counts = tf.unique_with_counts(track_ids)
        track_duplicates = track_uniques[track_counts > 1]
        # track_duplicates now is a list that inclused the `pred_sid` where two tracks
        # are assigned to the same `pred_sid`
        duplicate_tensor = tf.logical_and(
                tf.equal(tf.expand_dims(pred_sid, axis=-1), track_duplicates),
                tf.expand_dims(is_track==1, axis=-1)
                )
        beta_exp = tf.reshape(
                tf.repeat(pred_beta, repeats=track_duplicates.shape[0]),
                (-1, 1, track_duplicates.shape[0])
                )
        beta_check = tf.where(duplicate_tensor, beta_exp, tf.zeros_like(beta_exp))
        # beta_check: shape (Nbatch, 1, n_duplicates)
        # all values are zero except for
        track_select = tf.argmax(beta_check, axis=0)
        # track_select now includes the information of which index to chose for the pred_sid
        # that are listed in `track_duplicates`
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        if raw:
            pred_shower_tracks = (
                    tf.math.unsorted_segment_sum(
                        e_track[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        else:
            pred_shower_tracks = (
                    tf.math.unsorted_segment_sum(
                        e_track[:, 0] * pred_corr_factor[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)
        # This returns the summed up energy with the and has length n_showers
        # we need to replace the entries that have duplicate tracks with the chosen
        # track's energy (multiplied by the corresponding correction factor)
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)

        if raw:
            pred_shower_hits = (
                    tf.math.unsorted_segment_sum(
                        e_hit[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        else:
            pred_shower_hits = (
                    tf.math.unsorted_segment_sum(
                        e_hit[:, 0] * pred_corr_factor[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )

        pred_energy_hits = tf.gather(pred_shower_hits, unique_segments)
        pred_energy_tracks = tf.gather(pred_shower_tracks, unique_segments)

        if return_tracks_where_possible:
            pred_energy = tf.where(
                    pred_energy_tracks!=0.,
                    pred_energy_tracks,
                    pred_energy_hits)
            return pred_energy[:, tf.newaxis]
        elif return_tracks:
            return pred_energy_tracks[:, tf.newaxis]
        else:
            return pred_energy_hits[:, tf.newaxis]



class OCGatherEnergyHitsOrTracks(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac2, self).__init__(**kwargs)


    def call(self,
            pred_sid,
            pred_corr_factor,
            rechit_energy,
            no_noise_idx,
            pred_beta,
            is_track=None,
            row_splits=None,
            return_tracks_where_possible=True,
            return_tracks=False,
            raw=False):
        """
        Collects energy of hits belonging to showers.
        Default: 
            1. Check if a track is part of the shower.
                1.1 If exactly one track is part of the shower
                    -> Return (corrected)track energy
                1.2 If more than one track assigned to shower
                    -> Return (corrected) track energy of highes beta track
                1.3 If option is set to only use detector hits
                    -> Ignore tracks (and their correction factor)
            2. Sum over all hits in the shower
            3. Multiply hits by correction factor of hit with highes beta value
        
        Shapes:
            * rechit_energy, is_track:      [N_orig, 1]     (before noise filter)
            * pred_sid, pred_corr_factor:   [N_filtered, 1] (after noise filter)
            * no_noise_idx:                 [N_filtered, 1] (with indices up to N_orig as entries)
        Returns:
            Will always return only one value:
                return_tracks_where_possible overrides `return_tracks`
            By default returns hit energy
        """

        is_track = tf.cast(is_track, tf.int32)

        if row_splits is None:
            row_splits = tf.constant([0, rechit_energy.shape[0]])
        else:
            print("WARNING OCGatherEnergyCorrFac2: \nI am not sure if this works with explicit row splits")
            raise NotImplementedError

        pred_sid_p1 = pred_sid +1 # For the -1 label for noise
        pred_corr_factor = tf.where(
                pred_sid==-1,   # Correct noise with a factor of zero
                tf.zeros_like(pred_corr_factor, tf.float32),
                pred_corr_factor)

        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy))
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy))

        # Second term is a factor to make all the pred_sids unique
        # In case of only one sample (i.e. rowsplits [0, N_filtered]) this is identical to `pred_sid_p1`
        # Shape: [N_filtered,] ! Different from other shapes
        pdb.set_trace()
        unique_segments = pred_sid_p1[:, 0] +\
                (tf.reduce_max(pred_sid_p1)+1) * tf.ragged.row_splits_to_segment_ids(row_splits, out_type=tf.int32)

        track_ids = pred_sid[is_track == 1]
        track_uniques, track_ids, track_counts = tf.unique_with_counts(track_ids)
        track_duplicates = track_uniques[track_counts > 1]
        # track_duplicates now is a list that includes the `pred_sid` where two+ tracks
        # are assigned to the same `pred_sid`
        duplicate_tensor = tf.logical_and(
                tf.equal(tf.expand_dims(pred_sid, axis=-1), track_duplicates),
                tf.expand_dims(is_track==1, axis=-1)
                )
        beta_exp = tf.reshape(
                tf.repeat(pred_beta, repeats=track_duplicates.shape[0]),
                (-1, 1, track_duplicates.shape[0])
                )
        beta_check = tf.where(duplicate_tensor, beta_exp, tf.zeros_like(beta_exp))
        # beta_check: shape (Nbatch, 1, n_duplicates)
        # all values are zero except for
        track_select = tf.argmax(beta_check, axis=0)
        # track_select now includes the information of which index to chose for the pred_sid
        # that are listed in `track_duplicates`
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        if raw:
            pred_shower_tracks = (
                    tf.math.unsorted_segment_sum(
                        e_track[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        else:
            pred_shower_tracks = (
                    tf.math.unsorted_segment_sum(
                        e_track[:, 0] * pred_corr_factor[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)
        # This returns the summed up energy with the and has length n_showers
        # we need to replace the entries that have duplicate tracks with the chosen
        # track's energy (multiplied by the corresponding correction factor)
        indices = tf.expand_dims(track_duplicates+1, axis=1)
        updates = tf.gather_nd(e_track[:, 0], tf.expand_dims(track_select[0], axis=1))
        pred_shower_tracks = tf.tensor_scatter_nd_update(
                pred_shower_tracks, indices, updates)

        if raw:
            pred_shower_hits = (
                    tf.math.unsorted_segment_sum(
                        e_hit[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )
        else:
            pred_shower_hits = (
                    tf.math.unsorted_segment_sum(
                        e_hit[:, 0] * pred_corr_factor[:, 0],
                        unique_segments,
                        num_segments=(tf.reduce_max(unique_segments)+1))
                    )

        pred_energy_hits = tf.gather(pred_shower_hits, unique_segments)
        pred_energy_tracks = tf.gather(pred_shower_tracks, unique_segments)

        if return_tracks_where_possible:
            pred_energy = tf.where(
                    pred_energy_tracks!=0.,
                    pred_energy_tracks,
                    pred_energy_hits)
            return pred_energy[:, tf.newaxis]
        elif return_tracks:
            return pred_energy_tracks[:, tf.newaxis]
        else:
            return pred_energy_hits[:, tf.newaxis]



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


def OCHits2ShowersLayer_HDBSCAN(pred_ccoords, pred_beta, pred_dist, min_cluster_size=5, min_samples=None,
        mask_center=None, mask_radius=None):
    """
    Functions that fills the same role as the class `OCHits2ShowersLayer` but uses HDBSCAN
    instead of the simple clustering algorithm.
    Inputs and outputs are supposed to be compatible with the inputs and outputs of the
    `OCHits2ShowersLayer` class.
    As I don't know all of the  outputs of the `OCHits2ShowersLayer` I will make sure that
    only the outputs that are actually used are returned.
    Inputs:
        - predictions_dict['pred_ccoords']
        - predictions_dict['pred_beta']
        - predictions_dict['pred_dist']
        - min_cluster_size                  to be passed to HDBSCAN algorithm
        - min_samples                       to be passed to HDBSCAN algorithm
        - mask_radius                       to run HDBSCAN only on a subset of the clustering space
    Outputs:
        - pred_sid                      shower id
        - _                             don't know what it is
        - alpha_idx                     location of condensation points
        - _                             don't know what it is
        - _                             number of condensates?
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples = min_samples,
        gen_min_span_tree=True)

    index = np.arange(pred_beta.shape[0])

    if mask_radius is not None:
        mask_radius = float(mask_radius)
        # mask = pred_sid == 0
        # shower_indices = index[mask]                # shape (n_shower_0,)
        # alpha = shower_indices[np.argmax(beta)]     # maps back to orig index
        # center = pred_ccoords[alpha]
        center_mask = np.linalg.norm(pred_ccoords - mask_center, axis=1) < mask_radius
        
        pred_ccoords = pred_ccoords[center_mask]
        pred_beta = pred_beta[center_mask]
        index = index[center_mask]

    clusterer.fit(pred_ccoords)
    pred_sid = clusterer.labels_
    alpha_idx = []
    for sid in np.unique(pred_sid):
        mask = pred_sid == sid
        if sid == -1:
            continue
        beta = pred_beta[mask]                              # shape (n_shower,)
        shower_indices = index[mask]                        # shape (n_shower,)
        alpha_idx.append(shower_indices[np.argmax(beta)])
    alpha_idx = np.array(alpha_idx)
    # turn pred_sid to int32
    pred_sid = pred_sid.astype(np.int32)
    if mask_radius is not None:
        tmp = np.full_like(pred_dist, -1)
        tmp = tmp.reshape((-1))
        tmp[center_mask] = pred_sid
        pred_sid = tf.cast(tmp, tf.int32)

    return pred_sid[:, np.newaxis], None, alpha_idx, None, None


def process_endcap2(hits2showers_layer, energy_gather_layer, features_dict,
        predictions_dict, energy_mode='comb', raw=False,
        hdbscan=False, min_cluster_size=None, min_samples=None, mask_center=None, mask_radius=None):
    """
    Almost identical to `process_endcap`.
    Difference is that this takes into account the existence of tracks when
    summing over the energies.
    As we don't have the `is_track` variable included in the features or
    predictions we currently identify tracks over their z-position (Z==315)

    If there is no `no_noise_sel` in the predictions dict, it will be assumed
    that no noise filtering has been done/is necessary.
    """
    if not 'no_noise_sel' in predictions_dict.keys():
        N_pred = len(predictions_dict['pred_beta'])
        predictions_dict['no_noise_sel'] = np.arange(N_pred).reshape((N_pred,1)).astype(int)
    is_track = np.abs(features_dict['recHitZ']) == 315

    if not hdbscan:
        # Assume the we use the old clustering algorithm
        pred_sid, _, alpha_idx, _, ncond = hits2showers_layer(
                predictions_dict['pred_ccoords'],
                predictions_dict['pred_beta'],
                predictions_dict['pred_dist'])
    else:
        # Assume that we use the new clustering algorithm hdbscan
        pred_sid, _, alpha_idx, _, ncond = hits2showers_layer(
                predictions_dict['pred_ccoords'],
                predictions_dict['pred_beta'],
                predictions_dict['pred_dist'],
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                mask_center=mask_center,
                mask_radius=mask_radius)
    if not isinstance(pred_sid, np.ndarray):
        pred_sid = pred_sid.numpy()
    if not isinstance(alpha_idx, np.ndarray):
        alpha_idx = alpha_idx.numpy()

    processed_pred_dict = dict()
    processed_pred_dict['pred_sid'] = pred_sid


    pred_energy_hits = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=False,
            return_tracks=False,
            raw=False,
            )

    pred_energy_tracks = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=False,
            return_tracks=True,
            raw=False,
            )

    pred_energy_comb = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=True,
            return_tracks=False,
            raw=False,
            )

    pred_energy_hits_raw = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=False,
            return_tracks=False,
            raw=True,
            )

    pred_energy_tracks_raw = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=False,
            return_tracks=True,
            raw=True,
            )

    pred_energy_comb_raw = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            return_tracks_where_possible=True,
            return_tracks=False,
            raw=raw,
            )

    if energy_mode == 'hits':
        if raw:
            processed_pred_dict['pred_energy'] = pred_energy_hits_raw.numpy()
        else:
            processed_pred_dict['pred_energy'] = pred_energy_hits.numpy()
    elif energy_mode == 'tracks':
        if raw:
            processed_pred_dict['pred_energy'] = pred_energy_tracks_raw.numpy()
        else:
            processed_pred_dict['pred_energy'] = pred_energy_tracks.numpy()
    elif energy_mode == 'combined':
        if raw:
            processed_pred_dict['pred_energy'] = pred_energy_comb_raw.numpy()
        else:
            processed_pred_dict['pred_energy'] = pred_energy_comb.numpy()
    else:
        print("Unrecognized energy_mode")
        sys.exit(1)

    processed_pred_dict['pred_energy_hits'] = pred_energy_hits.numpy()
    processed_pred_dict['pred_energy_tracks'] = pred_energy_tracks.numpy()
    processed_pred_dict['pred_energy_comb'] = pred_energy_comb.numpy()
    processed_pred_dict['pred_energy_hits_raw'] = pred_energy_hits_raw.numpy()
    processed_pred_dict['pred_energy_tracks_raw'] = pred_energy_tracks_raw.numpy()
    processed_pred_dict['pred_energy_comb_raw'] = pred_energy_comb_raw.numpy()

    if 'pred_energy_high_quantile' in predictions_dict.keys():
        processed_pred_dict['pred_energy_unc'] \
            = 0.5 * (predictions_dict['pred_energy_high_quantile'] - predictions_dict['pred_energy_low_quantile'])
        processed_pred_dict.update(predictions_dict)
        processed_pred_dict.pop('pred_beta')
        processed_pred_dict['pred_id'] = np.argmax(processed_pred_dict['pred_id'], axis=1)

    return processed_pred_dict, alpha_idx


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

