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
            source='hits', correction='alpha'):
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

        is_track = tf.cast(is_track, tf.int32) # [N_hit,]
        # is_track = tf.reshape(tf.gather(is_track, no_noise_idx), (-1,1)) # Shape [N_filtered, 1]
        # rechit_energy = tf.reshape(tf.gather(rechit_energy, no_noise_idx), (-1,1))  # Shape [N_filtered, 1]

        if row_splits is None:
            row_splits = tf.constant([0, rechit_energy.shape[0]])
        else:
            print("WARNING OCGatherEnergyCorrFac2: \nI am not sure if this works with explicit row splits")


        # Add one as we have -1 for noise
        pred_sid_p1 = pred_sid +1 # [N_hits, 1]
        shower_ids = np.arange(np.max(pred_sid_p1 + 1)) # [N_showers + 1, ] showeres + noise
        pred_corr_factor = tf.where(pred_sid==-1, tf.zeros_like(pred_corr_factor, tf.float32), pred_corr_factor)

        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy)) # [N_hit, 1]
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy)) # [N_hit, 1]

        shower_matrix = tf.cast(pred_sid_p1 == shower_ids, tf.float32) # [N_hit, N_shower+1]
        shower_e_hit_raw = e_hit * shower_matrix # [N_hit, N_shower+1]
        shower_e_track_raw = e_hit * shower_matrix # [N_hit, N_shower+1]
        shower_beta = pred_beta * shower_matrix # [N_hit, N_shower+1] 
        shower_beta_hit = tf.cast(np.logical_not(is_track), tf.float32) * shower_beta
        shower_beta_track = tf.cast(is_track, tf.float32) * shower_beta
        has_hit = tf.reduce_max(shower_beta_hit, axis=0) > 0.0  # [N_showers+1]
        has_track = tf.reduce_max(shower_beta_track, axis=0) > 0.0 # [N_showers+1]
        beta_alpha_hits = tf.math.argmax(shower_beta_hit, axis=0)
        beta_alpha_tracks = tf.math.argmax(shower_beta_track, axis=0)
        correction_alpha_hits = tf.gather_nd(pred_corr_factor[:,0], beta_alpha_hits[...,tf.newaxis]) # [N_showers+1,]
        correction_alpha_hits = tf.cast(has_hit, tf.float32) * correction_alpha_hits
        correction_alpha_tracks = tf.gather_nd(pred_corr_factor[:,0], beta_alpha_tracks[...,tf.newaxis]) # [N_showers+1,]
        correction_alpha_tracks = tf.cast(has_track, tf.float32) * correction_alpha_tracks
        energy_hits_raw = tf.reduce_sum(shower_e_hit_raw, axis=0)
        energy_tracks_raw = tf.reduce_sum(shower_e_track_raw, axis=0)
        energy_hits_alpha_corrected = correction_alpha_hits * energy_hits_raw
        energy_tracks_alpha_corrected = correction_alpha_tracks * energy_tracks_raw
        energy_hits_individual_corrected = tf.reduce_sum(pred_corr_factor * shower_e_hit_raw, axis=0)
        energy_tracks_individual_corrected = tf.reduce_sum(pred_corr_factor * shower_e_track_raw, axis=0)

        e_tracks_raw =  tf.reshape(
                np.sum(shower_matrix * energy_tracks_raw, axis=1),
                shape=(-1,1))
        e_tracks_alpha =  tf.reshape(
                np.sum(shower_matrix * energy_tracks_alpha_corrected, axis=1),
                shape=(-1,1))
        e_tracks_individual =  tf.reshape(
                np.sum(shower_matrix * energy_tracks_individual_corrected, axis=1),
                shape=(-1,1))
        e_hits_raw =  tf.reshape(
                np.sum(shower_matrix * energy_hits_raw, axis=1),
                shape=(-1,1))
        e_hits_alpha =  tf.reshape(
                np.sum(shower_matrix * energy_hits_alpha_corrected, axis=1),
                shape=(-1,1))
        e_hits_individual = tf.reshape(
                np.sum(shower_matrix * energy_hits_individual_corrected, axis=1),
                shape=(-1,1))
        if source == 'tracks':
            if correction == 'raw':
                return e_tracks_raw
            elif correction == 'alpha':
                return e_tracks_alpha
            elif correction == 'individual':
                return e_tracks_individual
        elif source == 'hits':
            if correction == 'raw':
                return e_hits_raw
            elif correction == 'alpha':
                return e_hits_alpha
            elif correction == 'individual':
                return e_hits_individual
        elif source == 'all':
            data = {
                'tracks_raw': e_tracks_raw,
                'tracks_alpha': e_tracks_alpha,
                'tracks_individual': e_tracks_individual,
                'hits_raw': e_hits_raw,
                'hits_alpha': e_hits_alpha,
                'hits_individual': e_hits_individual,
                }
            return data


        # unique_segments = pred_sid_p1[:, 0]

        """
        track_ids = tf.reshape(pred_sid[is_track == 1], shape=(-1,))
        pdb.set_trace()
        track_uniques, track_ids, track_counts = tf.unique_with_counts(track_ids)
        track_duplicates = track_uniques[track_counts > 1]
        # track_duplicates now is a list that inclused the `pred_sid` where two tracks
        # are assigned to the same `pred_sid`
        duplicate_tensor = tf.logical_and(
                # tf.equal(tf.expand_dims(pred_sid, axis=-1), track_duplicates),
                tf.equal(pred_sid, track_duplicates),
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
        """



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


class OCGatherEnergyCorrFac_new(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac_new, self).__init__(**kwargs)

    def call(self,
             pred_sid,
             pred_corr_factor,
             rechit_energy,
             no_noise_idx,
             pred_beta,
             is_track=None,
             return_tracks_where_possible=True,
             return_tracks=False,
             raw=False,
             alpha_idx_tracks=None,
             alpha_idx_hits=None,
             t_is_minbias=None):
        """
        Same as `OCGatherEnergyCorrFac` with the addition that one can chose if the energy
        reconstructed with the tracks or the enery reconstructed by the calorimeter should be used.
        Shapes:
            * rechit_energy, is_track:      [N_orig, 1]     (before noise filter)
            * pred_sid, pred_corr_factor:   [N_filtered, 1] (after noise filter)
            * pred_beta:                    [N_filtered, 1] (after noise filter)
            * no_noise_idx:                 [N_filtered, 1] (with indices up to N_orig as entries)
        Returns:
            Will always return only one value:
                return_tracks_where_possible overrides `return_tracks`
            By default returns hit energy
        """

        if alpha_idx_tracks is None or alpha_idx_hits is None:
            print("Please proved both `alpha_idx_tracks` and `..._hits` to OCGatherEnergyCorfFac_new")
            raise ValueError

        is_track = tf.cast(is_track, tf.int32)

        pred_sid_p1 = pred_sid +1 # For the -1 label for noise
        pred_corr_factor = tf.where(
                pred_sid==-1,
                tf.zeros_like(pred_corr_factor, tf.float32),
                pred_corr_factor)
        rechit_energy = tf.where(
                pred_sid==-1,
                tf.zeros_like(rechit_energy, tf.float32),
                rechit_energy)

        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy))
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy))
        beta_hit = tf.where(is_track==0, pred_beta, tf.zeros_like(pred_beta))
        beta_track = tf.where(is_track==1, pred_beta, tf.zeros_like(pred_beta))
        e_hit_shower = tf.math.unsorted_segment_sum(
            e_hit[:,0], pred_sid_p1[:,0], num_segments=(tf.reduce_max(pred_sid_p1)+1)
        )
        if t_is_minbias is not None:
            t_is_minbias = tf.reshape(t_is_minbias, shape=(-1,1))
            e_no_minbias = tf.where(t_is_minbias==1, tf.zeros_like(e_hit), e_hit)
            e_no_minbias_shower = tf.math.unsorted_segment_sum(
                e_no_minbias[:,0], pred_sid_p1[:,0], num_segments=(tf.reduce_max(pred_sid_p1)+1)
            )
        e_track_shower = tf.math.unsorted_segment_sum(
            e_track[:,0], pred_sid_p1[:,0], num_segments=(tf.reduce_max(pred_sid_p1)+1)
        )
        zero_appendix = tf.constant([[0]], dtype=pred_corr_factor.dtype)
        pred_corr_factor = tf.concat(
                [pred_corr_factor, zero_appendix], axis=0)
        alpha_idx_hits = tf.where(
                tf.math.is_nan(tf.cast(alpha_idx_hits, dtype=tf.float32)),
                -1.0,
                alpha_idx_hits)
        alpha_idx_tracks = tf.where(
                tf.math.is_nan(tf.cast(alpha_idx_tracks, dtype=tf.float32)),
                -1.0,
                alpha_idx_tracks)
        # TODO: The next line doesn't work, so we have to do another way to select the hits
        # correction_hits = pred_corr_factor[tf.cast(alpha_idx_hits, dtype=tf.int32)]
        correction_hits = tf.gather_nd(
                pred_corr_factor,
                tf.cast(alpha_idx_hits, tf.int32)[:,tf.newaxis])
        correction_hits = tf.concat([ [[0.]], correction_hits], axis=0)
        correction_tracks = tf.gather_nd(
                pred_corr_factor,
                tf.cast(alpha_idx_tracks, tf.int32)[:,tf.newaxis])
        correction_tracks = tf.concat([ [[0.]], correction_tracks], axis=0)
        # correction_tracks = pred_corr_factor[tf.cast(alpha_idx_tracks, dtype=tf.int32)]
        e_hit_shower_corrected = e_hit_shower * correction_hits[:,0]
        e_track_shower_corrected = e_track_shower * correction_tracks[:,0]
        e_hit_corrected = tf.reshape(
                tf.gather(e_hit_shower_corrected, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_hit_raw = tf.reshape(
                tf.gather(e_hit_shower, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_track_corrected = tf.reshape(
                tf.gather(e_track_shower_corrected, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_track_raw = tf.reshape(
                tf.gather(e_track_shower, pred_sid_p1[:,0]),
                shape=(-1,1))


        data = {
            'tracks_raw': e_track_raw,
            'tracks_corrected': e_track_corrected,
            'hits_raw': e_hit_raw,
            'hits_corrected': e_hit_corrected,
            # 'no_minbias_fraction': no_minbias_fraction,
            }
        if t_is_minbias is not None:
            e_no_minbias_out = tf.reshape(
                    tf.gather(e_no_minbias_shower, pred_sid_p1[:,0]),
                    shape=(-1,1))
            no_minbias_fraction = tf.math.divide_no_nan(e_no_minbias_out, e_hit_raw)
            data['no_minbias_fraction'] = no_minbias_fraction

        return data


class OCHits2ShowersLayer(tf.keras.layers.Layer):
    def __init__(self, 
            beta_threshold,
            distance_threshold,
            use_local_distance_thresholding=True,
            assign_by_max_beta=True,
            nbinning_dims=3,
            **kwargs):

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


def OCHits2ShowersLayer_HDBSCAN(
        pred_ccoords, pred_beta, pred_dist, min_cluster_size=5, min_samples=None,
        mask_center=None, mask_radius=None, is_track=None):
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
    is_track = tf.cast(is_track, tf.bool)
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
    alpha_idx_track = []
    alpha_idx_hit = []
    for sid in np.unique(pred_sid):
        mask = pred_sid == sid
        mask_track = tf.logical_and(mask, is_track[:,0])
        mask_hit = tf.logical_and(mask, ~is_track[:,0])
        if sid == -1:
            continue
        beta = pred_beta[mask]                              # shape (n_shower,)
        shower_indices = index[mask]                        # shape (n_shower,)
        beta_hit = pred_beta[mask_hit]
        beta_track = pred_beta[mask_track]
        shower_indices_hit = index[mask_hit]
        shower_indices_track = index[mask_track]
        if len(beta_hit) == 0:
            alpha_idx_hit.append(np.nan)
        else:
            alpha_idx_hit.append(shower_indices_hit[np.argmax(beta_hit)])
        if len(beta_track) == 0:
            alpha_idx_track.append(np.nan)
        else:
            alpha_idx_track.append(shower_indices_track[np.argmax(beta_track)])
        alpha_idx.append(shower_indices[np.argmax(beta)])

    alpha_idx = np.array(alpha_idx)
    alpha_idx = np.reshape(alpha_idx, newshape=(-1, 1))
    # turn pred_sid to int32
    pred_sid = pred_sid.astype(np.int32)
    if mask_radius is not None:
        tmp = np.full_like(pred_dist, -1)
        tmp = tmp.reshape((-1))
        tmp[center_mask] = pred_sid
        pred_sid = tf.cast(tmp, tf.int32)

    return pred_sid[:, np.newaxis], None, alpha_idx, alpha_idx_track, alpha_idx_hit



def process_endcap2(hits2showers_layer, energy_gather_layer, features_dict,
        predictions_dict, energy_mode='comb', raw=False,
        hdbscan=False, min_cluster_size=None, min_samples=None,
        mask_center=None, mask_radius=None,
        is_minbias=None):
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

    print("Make showers")
    if not hdbscan:
        # Assume the we use the old clustering algorithm
        pred_sid, _, alpha_idx, _, ncond = hits2showers_layer(
                predictions_dict['pred_ccoords'],
                predictions_dict['pred_beta'],
                predictions_dict['pred_dist'])
        alpha_idx_hits = tf.cast(alpha_idx, tf.float32)
        alpha_idx_tracks = tf.cast(alpha_idx, tf.float32)
    else:
        # Assume that we use the new clustering algorithm hdbscan
        pred_sid, _, alpha_idx, alpha_idx_tracks, alpha_idx_hits = hits2showers_layer(
                predictions_dict['pred_ccoords'],
                predictions_dict['pred_beta'],
                predictions_dict['pred_dist'],
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                mask_center=mask_center,
                mask_radius=mask_radius,
                is_track=is_track)
    if not isinstance(pred_sid, np.ndarray):
        pred_sid = pred_sid.numpy()
    if not isinstance(alpha_idx, np.ndarray):
        alpha_idx = alpha_idx.numpy()
    alpha_idx = np.reshape(alpha_idx, newshape=(-1,))
    print("Made showers")

    processed_pred_dict = dict()
    processed_pred_dict['pred_sid'] = pred_sid

    print("Get energy")
    energy_data = energy_gather_layer(
            pred_sid,
            predictions_dict['pred_energy_corr_factor'],
            features_dict['recHitEnergy'],
            predictions_dict['no_noise_sel'],
            predictions_dict['pred_beta'],
            is_track = is_track,
            alpha_idx_hits = alpha_idx_hits,
            alpha_idx_tracks = alpha_idx_tracks,
            t_is_minbias = is_minbias,
            )
    print("Got energy")


    try:
        processed_pred_dict['pred_energy'] = energy_data['hits_alpha'].numpy()
        processed_pred_dict['pred_energy_hits_alpha'] = energy_data['hits_alpha'].numpy()
        processed_pred_dict['pred_energy_tracks_alpha'] = energy_data['tracks_alpha'].numpy()
        processed_pred_dict['pred_energy_hits_individual'] = energy_data['hits_individual'].numpy()
        processed_pred_dict['pred_energy_tracks_individual'] = energy_data['tracks_individual'].numpy()
        processed_pred_dict['pred_energy_hits_raw'] = energy_data['hits_raw'].numpy()
        processed_pred_dict['pred_energy_tracks_raw'] = energy_data['tracks_raw'].numpy()
    except KeyError:
        pass
    try:
        processed_pred_dict['pred_no_minbias_fraction'] = energy_data['no_minbias_fraction'].numpy()
    except KeyError:
        print("no_minbias_fraction not found")
        pass
    try:
        processed_pred_dict['pred_energy'] = energy_data['hits_raw'].numpy()
        processed_pred_dict['pred_energy_hits_raw'] = energy_data['hits_raw'].numpy()
        processed_pred_dict['pred_energy_hits_corrected'] = energy_data['hits_corrected'].numpy()
        processed_pred_dict['pred_energy_tracks_raw'] = energy_data['tracks_raw'].numpy()
        processed_pred_dict['pred_energy_tracks_corrected'] = energy_data['tracks_corrected'].numpy()
    except KeyError:
        pass


    """
    processed_pred_dict['pred_energy_hits'] = pred_energy_hits.numpy()
    processed_pred_dict['pred_energy_tracks'] = pred_energy_tracks.numpy()
    processed_pred_dict['pred_energy_comb'] = pred_energy_comb.numpy()
    processed_pred_dict['pred_energy_hits_raw'] = pred_energy_hits_raw.numpy()
    processed_pred_dict['pred_energy_tracks_raw'] = pred_energy_tracks_raw.numpy()
    processed_pred_dict['pred_energy_comb_raw'] = pred_energy_comb_raw.numpy()
    """

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

