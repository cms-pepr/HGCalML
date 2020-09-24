#!/usr/bin/env python3

from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy

numpy.set_printoptions(threshold=500)
import numpy as np

np.set_printoptions(threshold=500)
numpy.set_printoptions(threshold=500)
from LayersRagged import RaggedConstructTensor
import os
import argparse
import matplotlib.pyplot as plt
import gzip
import pickle
from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot, createRandomizedColors
from DeepJetCore.training.gpuTools import DJCSetGPUs
from obc_data import build_dataset_analysis_dict, build_window_analysis_dict, append_window_dict_to_dataset_dict, \
    build_window_visualization_dict
import copy
import networkx as nx
import index_dicts
import time

from ragged_plotting_tools import make_plots_from_object_condensation_clustering_analysis

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply

from numba import jit

# tf.compat.v1.disable_eager_execution()


# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


ragged_constructor = RaggedConstructTensor()


def find_uniques_from_betas(betas, coords, dist_threshold, beta_threshold, soft):
    print('find_uniques_from_betas')

    from condensate_op import BuildCondensates
    '''

    .Output("asso_idx: int32")
    .Output("is_cpoint: int32")
    .Output("n_condensates: int32");
    '''
    row_splits = np.array([0, len(betas)], dtype='int32')
    asso_idx, is_cpoint, _ = BuildCondensates(coords, betas, row_splits,
                                              radius=dist_threshold, min_beta=beta_threshold, soft=soft)
    allidxs = np.arange(len(betas))
    representative_indices = allidxs[is_cpoint > 0]
    # this should be fixed downstream. and index to an array of indicies to get a vertex is not a good way to code this!
    labels = asso_idx.numpy().copy()
    for i in range(len(representative_indices)):
        ridx = representative_indices[i]
        labels[asso_idx == ridx] = i

    return labels, representative_indices, asso_idx.numpy()



beta_threshold = 0.1
distance_threshold = 0.5
iou_threshold = 0.1

def calculate_all_iou_tf_3(truth_idx,
                           pred_idx,
                           hit_weight,
                           iou_threshold):
    truth_idx = tf.cast(tf.convert_to_tensor(truth_idx), tf.int32)
    pred_idx = tf.cast(tf.convert_to_tensor(pred_idx), tf.int32)
    hit_weight = tf.cast(tf.convert_to_tensor(hit_weight), tf.float32)

    pred_shower_idx = tf.convert_to_tensor(np.unique(pred_idx))
    truth_shower_idx = tf.convert_to_tensor(np.unique(truth_idx))
    len_pred_showers = len(pred_shower_idx)
    len_truth_showers = len(truth_shower_idx)

    truth_idx_2 = tf.zeros_like(truth_idx)
    pred_idx_2 = tf.zeros_like(pred_idx)
    hit_weight_2 = tf.zeros_like(hit_weight)

    for i in range(len(pred_shower_idx)):
        pred_idx_2 = tf.where(pred_idx == pred_shower_idx[i], i, pred_idx_2)

    for i in range(len(truth_shower_idx)):
        truth_idx_2 = tf.where(truth_idx == truth_shower_idx[i], i, truth_idx_2)

    one_hot_pred = tf.one_hot(pred_idx_2, depth=len_pred_showers)
    one_hot_truth = tf.one_hot(truth_idx_2, depth=len_truth_showers)

    intersection_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)
    union_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], tf.ones_like(one_hot_truth),
                                    transpose_a=True) + tf.linalg.matmul(
        tf.ones_like(one_hot_pred) * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True) - intersection_matrix

    overlap_matrix = (intersection_matrix / union_matrix).numpy()
    pred_shower_idx = pred_shower_idx.numpy()
    truth_shower_idx = truth_shower_idx.numpy()

    # print(overlap_matrix.shape)
    # 0/0

    all_iou = []
    for i in range(len_pred_showers):
        for j in range(len_truth_showers):
            # # if does_intersect_matrix[i,j] == 0:
            # #     continue
            # intersection = tf.cast(tf.math.logical_and((truth_idx == truth_shower_idx[j]), (pred_idx == pred_shower_idx[i])), tf.float32)
            #
            # union = tf.reduce_sum(tf.cast(tf.math.logical_or ((truth_idx == truth_shower_idx[j]), (pred_idx == pred_shower_idx[i])), tf.float32) * hit_weight)
            # # print(tf.reduce_sum(intersection), intersection_matrix[i,j], tf.reduce_sum(union), union_matrix[i,j])
            # if union > 0:
            #     print(tf.reduce_sum(union), union_matrix[i, j])
            #
            # overlap = tf.reduce_sum(hit_weight * intersection) / tf.reduce_sum(hit_weight * union)
            #
            # total_sum += (overlap - overlap_matrix[i,j])**2

            overlap = overlap_matrix[i, j]

            if overlap > iou_threshold:
                if pred_shower_idx[i] == -1 or truth_shower_idx[j] == -1:
                    continue
                all_iou.append((pred_shower_idx[i], truth_shower_idx[j], overlap))
                # G.add_edge('p%d'%index_shower_prediction, 't%d'%unique_showers_this_segment[i], weight=overlap)
    return all_iou

def match_to_truth_2(truth_sid, pred_sid, pred_shower_sid, hit_weight):
    print('match_to_truth')

    global iou_threshold
    truth_shower_sid_to_pred_shower_sid = {}
    truth_shower_sid = np.unique(truth_sid)
    for x in truth_shower_sid:
        truth_shower_sid_to_pred_shower_sid[x] = -1

    pred_shower_sid_to_truth_shower_sid = {}
    for x in pred_shower_sid:
        pred_shower_sid_to_truth_shower_sid[x] = -1

    G = nx.Graph()
    all_iou = calculate_all_iou_tf_3(truth_sid, pred_sid, hit_weight, iou_threshold)
    for iou in all_iou:
        G.add_edge('p%d' % iou[0], 't%d' % iou[1], weight=iou[2])

    X = nx.algorithms.max_weight_matching(G)
    for x, y in X:
        if x[0] == 'p':
            prediction_index = int(x[1:])
            truth_index = int(y[1:])
        else:
            truth_index = int(x[1:])
            prediction_index = int(y[1:])

        # print(x,y, truth_index, prediction_index)

        truth_shower_sid_to_pred_shower_sid[truth_index] = prediction_index
        pred_shower_sid_to_truth_shower_sid[prediction_index] = truth_index

    new_indicing = np.max(truth_shower_sid) + 1
    pred_sid_2 = np.zeros_like(pred_sid, np.int32) - 1

    pred_shower_sid_2 = []

    for k in pred_shower_sid:
        v = pred_shower_sid_to_truth_shower_sid[k]
        if v != -1:
            pred_sid_2[pred_sid == k] = v
            pred_shower_sid_2.append(v)
        else:
            pred_sid_2[pred_sid == k] = new_indicing
            pred_shower_sid_2.append(new_indicing)
            new_indicing += 1

    return pred_sid_2, pred_shower_sid_2


window_id = 0


class WindowAnalyser:
    def __init__(self, truth_sid, features, truth, prediction, beta_threshold,
                 distance_threshold, should_return_visualization_data=False, soft=False):
        features, prediction = index_dicts.split_feat_pred(prediction)
        pred_and_truth_dict = index_dicts.create_index_dict(truth, prediction)
        feature_dict = index_dicts.create_feature_dict(features)

        self.truth_shower_sid, unique_truth_shower_hit_idx = np.unique(truth_sid, return_index=True)
        unique_truth_shower_hit_idx = unique_truth_shower_hit_idx[self.truth_shower_sid!=-1]
        self.truth_shower_sid = self.truth_shower_sid[self.truth_shower_sid!=-1]

        self.sid_to_truth_shower_sid_idx = dict()
        for i in range(len(self.truth_shower_sid)):
            self.sid_to_truth_shower_sid_idx[self.truth_shower_sid[i]] = i

        self.truth_dep_energy = pred_and_truth_dict['truthRechitsSum'][:, 0]  # Flatten it
        self.truth_shower_energy = pred_and_truth_dict['truthHitAssignedEnergies'][:, 0][unique_truth_shower_hit_idx]
        self.truth_shower_eta = pred_and_truth_dict['truthHitAssignedEtas'][unique_truth_shower_hit_idx][:, 0]
        self.truth_shower_phi = pred_and_truth_dict['truthHitAssignedPhis'][unique_truth_shower_hit_idx][:, 0]
        self.hit_energy = feature_dict['recHitEnergy'][:, 0]
        self.pred_beta = pred_and_truth_dict['predBeta'][:, 0]
        self.pred_energy = pred_and_truth_dict['predEnergy'][:, 0]
        self.pred_x = (pred_and_truth_dict['predX'] + feature_dict["recHitX"])[:, 0]
        self.pred_y = (pred_and_truth_dict['predY'] + feature_dict["recHitY"])[:, 0]
        self.pred_ccoords = pred_and_truth_dict['predCCoords']
        self.truth_energy = pred_and_truth_dict['truthHitAssignedEnergies'][:, 0]
        self.truth_x = pred_and_truth_dict['truthHitAssignedX'][:, 0]
        self.truth_y = pred_and_truth_dict['truthHitAssignedY'][:, 0]

        self.pred_and_truth_dict = pred_and_truth_dict
        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.is_soft = soft
        self.truth_sid = truth_sid
        self.results_dict = build_window_analysis_dict()

    def compute_and_match_predicted_showers(self):
        pred_sid, pred_shower_representative_hit_idx, assoidx = find_uniques_from_betas(self.pred_beta,
                                                                                        self.pred_ccoords,
                                                                                        dist_threshold=distance_threshold,
                                                                                        beta_threshold=beta_threshold,
                                                                                        soft=self.is_soft)



        pred_shower_sid = [pred_sid[x] for x in pred_shower_representative_hit_idx]
        pred_representative_coords = [self.pred_ccoords[x] for x in pred_shower_representative_hit_idx]

        self.pred_sid, self.pred_shower_sid = match_to_truth_2(self.truth_sid, pred_sid, pred_shower_sid,
                                                               self.hit_energy)
        self.pred_shower_representative_coords = pred_representative_coords
        self.pred_shower_representative_hit_idx = pred_shower_representative_hit_idx
        self.pred_shower_energy = [self.pred_energy[x] for x in pred_shower_representative_hit_idx]
        self.sid_to_pred_shower_sid_idx = dict()

        for i in range(len(self.pred_shower_sid)):
            self.sid_to_pred_shower_sid_idx[self.pred_shower_sid[i]] = i

    def match_ticl_showers(self):
        ticl_sid = self.pred_and_truth_dict['ticlHitAssignementIdx'][:, 0]
        ticl_energy = self.pred_and_truth_dict['ticlHitAssignedEnergies'][:, 0]

        ticl_sid[ticl_energy > 10 * self.truth_dep_energy] = 0
        ticl_energy[ticl_energy > 10 * self.truth_dep_energy] = 0

        ticl_shower_sid, ticl_shower_idx = np.unique(ticl_sid[ticl_sid >= 0], return_index=True)
        ticl_energy_2 = ticl_energy[ticl_sid >= 0]

        self.ticl_sid, self.ticl_shower_sid = match_to_truth_2(self.truth_sid, ticl_sid, ticl_shower_sid,
                                                               self.hit_energy)
        self.ticl_shower_energy = [ticl_energy_2[x] for x in ticl_shower_idx]
        self.sid_to_ticl_shower_sid_idx = dict()
        for i in range(len(self.ticl_shower_sid)):
            self.sid_to_ticl_shower_sid_idx[self.ticl_shower_sid[i]] = i

    def gather_truth_matchings(self):
        num_found = 0.
        num_found_ticl = 0.
        num_missed = 0.
        num_missed_ticl = 0.
        num_gt_showers = 0.

        predicted_truth_total = 0
        for i in range(len(self.truth_shower_sid)):
            sid = self.truth_shower_sid[i]

            self.results_dict['truth_shower_sid'].append(sid)
            self.results_dict['truth_shower_energy'].append(self.truth_shower_energy[i])
            sum = self.truth_energy[self.truth_sid == sid]
            self.results_dict['truth_shower_energy_sum'].append(sum)
            predicted_truth_total += self.truth_shower_energy[i]

            self.results_dict['truth_shower_eta'].append(self.truth_shower_eta[i])

            distances_with_other_truths = np.sqrt((self.truth_shower_eta[i] - self.truth_shower_eta) ** 2 + (
                np.arctan2(np.sin(self.truth_shower_phi[i] - self.truth_shower_phi),
                           np.cos(self.truth_shower_phi[i] - self.truth_shower_phi))) ** 2)
            distances_with_other_truths_excluduing_me = distances_with_other_truths[distances_with_other_truths != 0]
            distance_to_closest_truth = np.min(distances_with_other_truths_excluduing_me)
            density = self.truth_shower_energy[i] / (
                        np.sum(self.truth_shower_energy[distances_with_other_truths < 0.5]) + 1e-6)
            self.results_dict['truth_shower_local_density'].append(density)
            self.results_dict['truth_shower_closest_particle_distance'].append(distance_to_closest_truth)

            pred_match_shower_idx = self.sid_to_pred_shower_sid_idx[
                sid] if sid in self.sid_to_pred_shower_sid_idx else -1
            if pred_match_shower_idx > -1:
                predicted_energy = self.pred_shower_energy[pred_match_shower_idx]

                self.results_dict['truth_shower_found_or_not'].append(True)
                self.results_dict['truth_shower_matched_energy_sum'].append(
                    np.sum(self.hit_energy[self.pred_sid == sid]))
                self.results_dict['truth_shower_matched_energy_regressed'].append(predicted_energy)

                num_found += 1
            else:
                self.results_dict['truth_shower_found_or_not'].append(False)
                self.results_dict['truth_shower_matched_energy_sum'].append(-1)
                self.results_dict['truth_shower_matched_energy_regressed'].append(-1)

                num_missed += 1

            ticl_match_shower_idx = self.sid_to_ticl_shower_sid_idx[
                sid] if sid in self.sid_to_ticl_shower_sid_idx else -1
            if ticl_match_shower_idx > -1:
                predicted_energy = self.ticl_shower_energy[ticl_match_shower_idx]

                self.results_dict['truth_shower_found_or_not_ticl'].append(True)
                self.results_dict['truth_shower_matched_energy_sum_ticl'].append(
                    np.sum(self.hit_energy[self.ticl_sid == sid]))
                self.results_dict['truth_shower_matched_energy_regressed_ticl'].append(predicted_energy)

                num_found_ticl += 1
            else:
                self.results_dict['truth_shower_found_or_not_ticl'].append(False)
                self.results_dict['truth_shower_matched_energy_sum_ticl'].append(-1)
                self.results_dict['truth_shower_matched_energy_regressed_ticl'].append(-1)

                num_missed_ticl += 1

            num_gt_showers += 1
            self.results_dict['truth_shower_num_rechits'].append(len(self.truth_sid[self.truth_sid == sid]))

        self.results_dict['window_num_rechits'] = len(self.truth_sid)

        global window_id
        self.results_dict['truth_shower_sample_id'] = (
                window_id + 0 * np.array(self.results_dict['truth_shower_energy'])).tolist()

        self.results_dict['window_num_truth_showers'] = num_gt_showers
        self.results_dict['window_num_found_showers'] = num_found
        self.results_dict['window_num_missed_showers'] = num_missed

        self.results_dict['window_num_found_showers_ticl'] = num_found_ticl
        self.results_dict['window_num_missed_showers_ticl'] = num_missed_ticl

        self.results_dict['window_total_energy_truth'] = predicted_truth_total

    def gather_prediction_matchings(self):

        predicted_total_obc = 0

        num_fakes = 0
        num_predicted_showers = 0

        for i in range(len(self.pred_shower_sid)):
            sid = self.pred_shower_sid[i]
            rep_idx = self.pred_shower_representative_hit_idx[i]

            shower_energy_predicted = self.pred_energy[rep_idx]
            predicted_total_obc += shower_energy_predicted

            shower_eta_predicted = self.pred_x[rep_idx]
            shower_phi_predicted = self.pred_y[rep_idx]
            shower_energy_sum_predicted = np.sum(self.hit_energy[self.pred_sid == sid])

            self.results_dict['pred_shower_sid'].append(sid)

            self.results_dict['pred_shower_regressed_energy'].append(shower_energy_predicted)
            self.results_dict['pred_shower_energy_sum'].append(shower_energy_sum_predicted)
            self.results_dict['pred_shower_regressed_phi'].append(shower_phi_predicted)
            self.results_dict['pred_shower_regressed_eta'].append(shower_eta_predicted)

            truth_match_shower_idx = self.sid_to_truth_shower_sid_idx[
                sid] if sid in self.sid_to_truth_shower_sid_idx else -1

            if truth_match_shower_idx > -1:
                shower_energy_truth = self.truth_shower_energy[truth_match_shower_idx]
                shower_eta_truth = self.truth_shower_eta[truth_match_shower_idx]
                shower_phi_truth = self.truth_shower_phi[truth_match_shower_idx]

                shower_energy_sum_truth = np.sum(self.hit_energy[self.truth_sid == sid])

                self.results_dict['pred_shower_matched_energy'].append(shower_energy_truth)
                self.results_dict['pred_shower_matched_energy_sum'].append(shower_energy_sum_truth)
                self.results_dict['pred_shower_matched_phi'].append(shower_phi_truth)
                self.results_dict['pred_shower_matched_eta'].append(shower_eta_truth)

                # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

            else:
                num_fakes += 1
                self.results_dict['pred_shower_matched_energy'].append(-1)
                self.results_dict['pred_shower_matched_energy_sum'].append(-1)
                self.results_dict['pred_shower_matched_phi'].append(-1)
                self.results_dict['pred_shower_matched_eta'].append(-1)

            self.results_dict['pred_shower_sample_id'] = (
                    window_id + 0 * np.array(self.results_dict['pred_shower_regressed_energy'])).tolist()
            num_predicted_showers += 1

            self.results_dict['window_num_fake_showers'] = num_fakes
            self.results_dict['window_num_pred_showers'] = num_predicted_showers
            self.results_dict['window_total_energy_pred'] = predicted_total_obc

    def gather_ticl_matchings(self):
        predicted_total_ticl = 0

        num_fakes_ticl = 0
        num_predicted_showers_ticl = 0

        for i in range(len(self.ticl_shower_sid)):
            sid = self.ticl_shower_sid[i]
            shower_energy_predicted = self.ticl_shower_energy[i]
            predicted_total_ticl += shower_energy_predicted

            shower_energy_sum_predicted = np.sum(self.hit_energy[self.ticl_sid == sid])

            self.results_dict['ticl_shower_sid'].append(sid)
            self.results_dict['ticl_shower_regressed_energy'].append(shower_energy_predicted)
            self.results_dict['ticl_shower_energy_sum'].append(shower_energy_sum_predicted)
            self.results_dict['ticl_shower_regressed_phi'].append(0)
            self.results_dict['ticl_shower_regressed_eta'].append(0)

            truth_match_shower_idx = self.sid_to_truth_shower_sid_idx[
                sid] if sid in self.sid_to_truth_shower_sid_idx else -1

            if truth_match_shower_idx > -1:
                shower_energy_truth = self.truth_shower_energy[truth_match_shower_idx]
                shower_eta_truth = self.truth_shower_eta[truth_match_shower_idx]
                shower_phi_truth = self.truth_shower_phi[truth_match_shower_idx]

                shower_energy_sum_truth = np.sum(self.hit_energy[self.truth_sid == sid])

                self.results_dict['ticl_shower_matched_energy'].append(shower_energy_truth)
                self.results_dict['ticl_shower_matched_energy_sum'].append(shower_energy_sum_truth)
                self.results_dict['ticl_shower_matched_phi'].append(shower_phi_truth)
                self.results_dict['ticl_shower_matched_eta'].append(shower_eta_truth)

                # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

            else:
                num_fakes_ticl += 1
                self.results_dict['ticl_shower_matched_energy'].append(-1)
                self.results_dict['ticl_shower_matched_energy_sum'].append(-1)
                self.results_dict['ticl_shower_matched_phi'].append(-1)
                self.results_dict['ticl_shower_matched_eta'].append(-1)

            self.results_dict['ticl_shower_sample_id'] = (
                    window_id + 0 * np.array(self.results_dict['ticl_shower_regressed_energy'])).tolist()
            num_predicted_showers_ticl += 1

            self.results_dict['window_num_fake_showers_ticl'] = num_fakes_ticl
            self.results_dict['window_num_ticl_showers'] = num_predicted_showers_ticl
            self.results_dict['window_total_energy_ticl'] = predicted_total_ticl

    def analyse(self):
        self.compute_and_match_predicted_showers()
        self.match_ticl_showers()
        self.gather_truth_matchings()
        self.gather_prediction_matchings()
        self.gather_ticl_matchings()

        return self.results_dict


def analyse_one_window_cut(truth_sid, features, truth, prediction, beta_threshold,
                           distance_threshold, should_return_visualization_data=False, soft=False):
    global window_id

    results_dict = WindowAnalyser(truth_sid, features, truth, prediction, beta_threshold,
                                  distance_threshold, should_return_visualization_data=False, soft=soft).analyse()
    window_id += 1
    return results_dict

    start_f = time.time()

    # print('analyse_one_window_cut', window_id)

    '''

    some naming conventions:

    *never* use 'segment' anywhere in here. the whole function works on one segment, so no need to blow up variable names

    pred_foo always refers to per-hit prediction (dimension V x ... or V)
    pred_shower_foo  always refers to predicted shower quantities  (dimension N_predshowers x ... or N_predshowers)

    truth_bar always refers to per-hit truth (dimension V x ... or V)
    truth_shower_bar alywas refers to shower quantities  (dimension N_trueshowers x ... or N_trueshowers)

    keep variable names in singular, in the end these are almost all vectors anyway

    always indicate if an array is an index array by appending "_idx". Not "idxs", "indices" or nothing

    if indices refer to the hit vector, call them ..._hit_idx

    for better distinction between indices referring to vectors and simple "ids" given to showers, call the "id" showers .._id
    (for example 

    '''

    # TODO: Copy this code to Analyser class
    # if should_return_visualization_data:
    #     print("CHECK IMPLEMENTATION")
    #     raise Exception("Not implemented")
    #
    #     vis_dict = build_window_visualization_dict()
    #     vis_dict['truth_showers'] = truth_sid  # replace(truth_id, replace_dictionary)
    #
    #     replace_dictionary = copy.deepcopy(pred_shower_sid_to_truth_shower_sid)
    #     replace_dictionary_2 = copy.deepcopy(replace_dictionary)
    #     start_secondary_indicing_from = np.max(truth_sid) + 1
    #     for k, v in replace_dictionary.items():
    #         if v == -1 and k != -1:
    #             replace_dictionary_2[k] = start_secondary_indicing_from
    #             start_secondary_indicing_from += 1
    #     replace_dictionary = replace_dictionary_2
    #
    #     # print("===============")
    #     # print(np.unique(pred_sid), replace_dictionary)
    #     vis_dict['predicted_showers'] = replace(pred_sid, replace_dictionary)
    #
    #     # print(predicted_showers_found)
    #
    #     # print(np.mean((pred_sid==-1)).astype(np.float))
    #
    #     if use_ticl:
    #         replace_dictionary = copy.deepcopy(ticl_shower_sid_to_truth_shower_sid)
    #         replace_dictionary_2 = copy.deepcopy(replace_dictionary)
    #         start_secondary_indicing_from = np.max(truth_sid) + 1
    #         for k, v in replace_dictionary.items():
    #             if v == -1 and k != -1:
    #                 replace_dictionary_2[k] = start_secondary_indicing_from
    #                 start_secondary_indicing_from += 1
    #         replace_dictionary = replace_dictionary_2
    #
    #         # print(np.unique(ticl_sid), replace_dictionary)
    #         vis_dict['ticl_showers'] = replace(ticl_sid, replace_dictionary)
    #     else:
    #         vis_dict['ticl_showers'] = pred_sid * 10
    #
    #     vis_dict['pred_and_truth_dict'] = pred_and_truth_dict
    #     vis_dict['feature_dict'] = feature_dict
    #
    #     vis_dict['coords_representatives'] = pred_representative_coords
    #     vis_dict['identified_vertices'] = pred_sid
    #
    #     results_dict['visualization_data'] = vis_dict



num_rechits_per_segment = []
num_rechits_per_shower = []


def analyse_one_file(features, predictions, truth_in, soft=False):
    global num_visualized_segments, num_segments_to_visualize
    global dataset_analysis_dict

    predictions = tf.constant(predictions[0])

    row_splits = features[1][:, 0]

    features, _ = ragged_constructor((features[0], row_splits))
    truth, _ = ragged_constructor((truth_in[0], row_splits))

    hit_assigned_truth_id, row_splits = ragged_constructor((truth_in[0][:, 0][..., tf.newaxis], row_splits))

    # make 100% sure the cast doesn't hit the fan
    hit_assigned_truth_id = tf.where(hit_assigned_truth_id < -0.1, hit_assigned_truth_id - 0.1,
                                     hit_assigned_truth_id + 0.1)
    hit_assigned_truth_id = tf.cast(hit_assigned_truth_id[:, 0], tf.int32)

    num_unique = []
    shower_sizes = []

    # here ..._s refers to quantities per window/segment
    #
    for i in range(len(row_splits) - 1):
        hit_assigned_truth_id_s = hit_assigned_truth_id[row_splits[i]:row_splits[i + 1]].numpy()
        features_s = features[row_splits[i]:row_splits[i + 1]].numpy()
        truth_s = truth[row_splits[i]:row_splits[i + 1]].numpy()
        prediction_s = predictions[row_splits[i]:row_splits[i + 1]].numpy()

        if num_visualized_segments < num_segments_to_visualize:
            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s, prediction_s,
                                                          beta_threshold, distance_threshold, True, soft=soft)
        else:
            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s, prediction_s,
                                                          beta_threshold, distance_threshold, False, soft=soft)

        append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
        num_visualized_segments += 1

    i += 1


def main(files, pdfpath, dumppath, soft):
    global dataset_analysis_dict
    file_index = 0
    for file in files:
        print("\nFILE\n", file_index)
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'], soft=soft)
            file_index += 1
            # break

    if len(dumppath) > 0:
        print("Dumping analysis to bin file", dumppath)

        with open(dumppath, 'wb') as f:
            pickle.dump(dataset_analysis_dict, f)
    else:
        print("WARNING: No analysis output path specified. Skipped dumping of analysis.")

    make_plots_from_object_condensation_clustering_analysis(pdfpath, dataset_analysis_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('output',
                        help='Output directory with .bin.gz files (all will be analysed) or a text file containing lest of those which are to be analysed')
    parser.add_argument('-p',
                        help='Name of the output file (you have to manually append .pdf). Otherwise will be produced in the output directory.',
                        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-v', help='Visualize number of showers', default='10')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
    parser.add_argument('--analysisoutpath', help='Can be used to remake plots. Will dump analysis to a file.',
                        default='')
    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_segments_to_visualize = int(args.v)
    num_visualized_segments = 0
    dataset_analysis_dict = build_dataset_analysis_dict()

    beta_threshold = beta_threshold * 0 + float(args.b)
    distance_threshold = distance_threshold * 0 + float(args.d)
    iou_threshold = iou_threshold * 0 + float(args.i)

    dataset_analysis_dict['distance_threshold'] = distance_threshold
    dataset_analysis_dict['beta_threshold'] = distance_threshold
    dataset_analysis_dict['iou_threshold'] = distance_threshold

    files_to_be_tested = []
    pdfpath = ''
    if os.path.isdir(args.output):
        for x in os.listdir(args.output):
            if x.endswith('.bin.gz'):
                files_to_be_tested.append(os.path.join(args.output, x))
        pdfpath = args.output
    elif os.path.isfile(args.output):
        with open(args.output) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        files_to_be_tested = [x.strip() for x in content]
        pdfpath = os.path.split(pdfpath)[0]
    else:
        raise Exception('Error: couldn\'t locate output folder/file')

    print(files_to_be_tested)
    pdfpath = os.path.join(pdfpath, 'plots.pdf')
    if len(args.p) != 0:
        pdfpath = args.p

    main(files_to_be_tested, pdfpath, args.analysisoutpath, soft=args.soft)

