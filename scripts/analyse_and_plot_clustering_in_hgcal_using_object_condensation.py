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
from obc_data import build_dataset_analysis_dict, build_window_analysis_dict, append_window_dict_to_dataset_dict, build_window_visualization_dict
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


os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



ragged_constructor = RaggedConstructTensor()



def find_uniques_from_betas(betas, coords, dist_threshold, beta_threshold, soft):
    print('find_uniques_from_betas')
    
    from condensate_op import BuildCondensates
    '''
    
    .Output("asso_idx: int32")
    .Output("is_cpoint: int32")
    .Output("n_condensates: int32");
    '''
    row_splits = np.array([0,len(betas)],dtype='int32')
    asso_idx, is_cpoint, _ = BuildCondensates(coords, betas, row_splits, 
                                              radius=dist_threshold, min_beta=beta_threshold, soft=soft)
    allidxs = np.arange(len(betas))
    representative_indices = allidxs[is_cpoint>0]
    #this should be fixed downstream. and index to an array of indicies to get a vertex is not a good way to code this!
    labels = asso_idx.numpy().copy()
    for i in range(len(representative_indices)):
        ridx = representative_indices[i]
        labels[asso_idx == ridx] = i
        
    return labels,representative_indices, asso_idx.numpy()
    

def replace(arr, rep_dict):
    """Assumes all elements of "arr" are keys of rep_dict"""

    # Removing the explicit "list" breaks python3
    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))

    idces = np.digitize(arr, rep_keys, right=True)
    # Notice rep_keys[digitize(arr, rep_keys, right=True)] == arr

    return rep_vals[idces]



beta_threshold = 0.1
distance_threshold = 0.5
iou_threshold = 0.1

@jit(nopython=True)
def calculate_all_iou(index_shower_prediction, representative_indices, 
                  unique_showers_this_segment,
                  truth_showers_this_segment,
                  labels_for_all,
                  hit_weight,
                  iou_threshold):
    
    #hit_weight =  rechit_energies_this_segment * beta_all
    
    all_iou=[]
    
    for index_shower_prediction in range(len(representative_indices)):

        for i in range(len(unique_showers_this_segment)):
            if unique_showers_this_segment[i] == -1:
                continue
            
            intersection = (truth_showers_this_segment == unique_showers_this_segment[i]) * (labels_for_all == index_shower_prediction)
            
            if not np.any(intersection): #no intersection
                continue
            
            union = np.logical_or((truth_showers_this_segment == unique_showers_this_segment[i]), (labels_for_all == index_shower_prediction))
            
            overlap = np.sum(hit_weight * intersection) / np.sum(hit_weight * union)


            if overlap > iou_threshold:
                all_iou.append( (index_shower_prediction,unique_showers_this_segment[i],overlap) )
                #G.add_edge('p%d'%index_shower_prediction, 't%d'%unique_showers_this_segment[i], weight=overlap)

    return all_iou

def match_to_truth(truth_showers, unique_showers, unique_showers_e, unique_shower_eta, labels, clustering_coords_all_filtered, representative_indices, rechit_energies):
    print('match_to_truth')
    
    global iou_threshold
    truth_showers_found = {}
    truth_showers_found_e = {}
    truth_showers_found_eta = {}
    iii = 0
    for x in unique_showers:
        truth_showers_found[x] = -1
        truth_showers_found_e[x] = unique_showers_e[iii]
        truth_showers_found_eta[x] = unique_shower_eta[iii]
        iii += 1

    unique_labels = np.unique(labels)

    # print("HELLO")
    # print(unique_labels)



    predicted_showers_found = {}
    for x in unique_labels:
        predicted_showers_found[x] = -1


    unique_showers_this_segment = np.unique(truth_showers)
    representative_coords = []
    G = nx.Graph()
    for index_shower_prediction in unique_labels:
        if index_shower_prediction == -1:
            continue

        for i in range(len(unique_showers_this_segment)):
            if unique_showers_this_segment[i] == -1:
                continue

            # sum_truth = np.sum(rechit_energies_this_segment * (truth_showers_this_segment == unique_showers_this_segment[i]))
            # sum_predicted = np.sum(rechit_energies_this_segment * (labels_for_all == index_shower_prediction))
            
            intersection = (truth_showers == unique_showers_this_segment[i]) * (labels == index_shower_prediction)
            if not np.any(intersection):
                continue
            
            overlap = np.sum(rechit_energies * intersection) / np.sum(rechit_energies * np.logical_or((truth_showers == unique_showers_this_segment[i]), (labels == index_shower_prediction)))

            nume = np.sum(rechit_energies * (truth_showers == unique_showers_this_segment[i]) * (labels == index_shower_prediction))
            deno = np.sum(rechit_energies * np.logical_or((truth_showers == unique_showers_this_segment[i]), (labels == index_shower_prediction)))



            if overlap > iou_threshold:
                G.add_edge('p%d'%index_shower_prediction, 't%d'%unique_showers_this_segment[i], weight=overlap)


    X = nx.algorithms.max_weight_matching(G)
    for x,y in X:
        if x[0] == 'p':
            prediction_index = int(x[1:])
            truth_index = int(y[1:])
        else:
            truth_index = int(x[1:])
            prediction_index = int(y[1:])

        # print(x,y, truth_index, prediction_index)

        truth_showers_found[truth_index] = prediction_index
        predicted_showers_found[prediction_index] = truth_index

    print('match_to_truth end')
    return truth_showers_found, predicted_showers_found




window_id = 0
def analyse_one_window_cut(truth_id, features, truth, prediction, beta_threshold, 
                           distance_threshold, should_return_visualization_data=False, soft=False):
    global window_id
    print('analyse_one_window_cut', window_id)
    
    
    '''
    
    some naming conventions:
    
    *never* use 'segment' anywhere in here. the whole function works on one segment, so no need to blow up variable names
    
    pred_foo always refers to per-hit prediction (dimension V x ... or V)
    pred_shower_foo  alywas refers to predicted shower quantities  (dimension N_predshowers x ... or N_predshowers)
    
    truth_bar always refers to per-hit truth (dimension V x ... or V)
    truth_shower_bar alywas refers to shower quantities  (dimension N_trueshowers x ... or N_trueshowers)
    
    keep variable names in singular, in the end these are almost all vectors anyway
    
    always indicate if an array is an index array by appending "_idx". Not "idxs", "indices" or nothing
    
    if indices refer to the hit vector, call them ..._hit_idx
    
    for better distinction between indices referring to vectors and simple "ids" given to showers, call the "id" showers .._id
    (for example 
    
    '''

    
    results_dict = build_window_analysis_dict()

    features, prediction = index_dicts.split_feat_pred(prediction)
    pred_and_truth_dict = index_dicts.create_index_dict(truth, prediction)
    feature_dict = index_dicts.create_feature_dict(features)

    truth_shower_id,unique_truth_shower_hit_idx = np.unique(truth_id, return_index=True)
    
    print('truth_id',truth_id.shape)
    print('unique_truth_shower_hit_idx',unique_truth_shower_hit_idx.shape)
    print("pred_and_truth_dict['predEnergy'][:, 0]",pred_and_truth_dict['predEnergy'][:, 0].shape)

    # TODO: confirm
    
    truth_dep_energy = pred_and_truth_dict['truthRechitsSum'][:,0] # Flatten it
    
    truth_shower_energy = pred_and_truth_dict['truthHitAssignedEnergies'][:, 0][unique_truth_shower_hit_idx]
    truth_shower_eta = pred_and_truth_dict['truthHitAssignedEtas'][unique_truth_shower_hit_idx][:, 0]
    truth_shower_phi = pred_and_truth_dict['truthHitAssignedPhis'][unique_truth_shower_hit_idx][:, 0]
    
    hit_energy = feature_dict['recHitEnergy'][:, 0]
    
    pred_beta = pred_and_truth_dict['predBeta'][:, 0]
    pred_energy = pred_and_truth_dict['predEnergy'][:, 0]
    pred_x = (pred_and_truth_dict['predX'] + feature_dict["recHitX"])[:, 0]
    pred_y = (pred_and_truth_dict['predY'] + feature_dict["recHitY"])[:, 0]
    pred_ccoords = pred_and_truth_dict['predCCoords']


    # print(truth.shape)

    # TODO: verification required and change eta to x
    truth_energy = pred_and_truth_dict['truthHitAssignedEnergies'][:, 0]
    truth_x = pred_and_truth_dict['truthHitAssignedX'][:, 0]
    truth_y = pred_and_truth_dict['truthHitAssignedY'][:, 0]


    use_ticl = True
    try:
        ticl_id = pred_and_truth_dict['ticlHitAssignementIdx'][:, 0]
        ticl_energy = pred_and_truth_dict['ticlHitAssignedEnergies'][:, 0]

        ticl_id[ticl_energy > 10 * truth_dep_energy] = 0
        ticl_energy[ticl_energy > 10 * truth_dep_energy] = 0
    except Exception:
        use_ticl = False


    pred_sid, representative_indices, assoidx = find_uniques_from_betas(pred_beta, pred_ccoords,
                                                             dist_threshold=distance_threshold, 
                                                             beta_threshold=beta_threshold,
                                                             soft=soft)
    
    pred_shower_id = np.unique(pred_sid)


    print('calc distances')
    truth_showers_found = {}
    truth_showers_found_e = {}
    truth_showers_found_e_sum = {}
    truth_showers_found_eta = {}
    truth_showers_found_closest_particle_distance = {}
    truth_showers_found_local_density = {}
    
    iii = 0
    for x in truth_shower_id:
        truth_showers_found[x] = -1
        truth_showers_found_e[x] = truth_shower_energy[iii]
        truth_showers_found_e_sum[x] = np.sum(hit_energy[truth_id==x])
        truth_showers_found_eta[x] = truth_shower_eta[iii]

        # print(truth_shower_eta[iii])

        # fixme. this needs to be deltaphi! not just a simple phiA-phiB!
        distances_with_other_truths = np.sqrt((truth_shower_eta[iii] - truth_shower_eta) ** 2  + (truth_shower_phi[iii] - truth_shower_phi)**2)
        distances_with_other_truths_excluduing_me = distances_with_other_truths[distances_with_other_truths!=0]
        distance_to_closest_truth = np.min(distances_with_other_truths_excluduing_me)

        density = truth_shower_energy[iii]/(np.sum(truth_shower_energy[distances_with_other_truths<0.5])+1e-6)
        truth_showers_found_closest_particle_distance[x] = distance_to_closest_truth
        truth_showers_found_local_density[x] = density

        iii += 1
        results_dict['num_rechits_per_truth_shower'].append(len(truth_id[truth_id == x]))
    results_dict['num_rechits_per_window'] = len(truth_id)


    #fill representative coordinates
    representative_coords = []
    for index_shower_prediction in range(len(representative_indices)):
        representative_coords.append(pred_ccoords[representative_indices[index_shower_prediction]])

    print('match fill') ##this one is slow like hell!
    predicted_showers_found = {}
    predicted_showers_representative_index = {}
    for x in pred_shower_id:
        predicted_showers_found[x] = -1
        predicted_showers_representative_index[x] = -1

    # use the same truth matching for ticl!
    # reduces time by a factor of 4-5
    G = nx.Graph()
    start = time.time()
    hit_weight = hit_energy * pred_beta
    all_iou = calculate_all_iou(index_shower_prediction, representative_indices, truth_shower_id, truth_id, pred_sid, hit_weight, iou_threshold)    
    for iou in all_iou:
        G.add_edge('p%d'%iou[0], 't%d'%iou[1], weight=iou[2])


    print('match fill', time.time()-start, 's')
    print('match resolve')
    X = nx.algorithms.max_weight_matching(G)
    for x,y in X:
        if x[0] == 'p':
            prediction_index = int(x[1:])
            truth_index = int(y[1:])
        else:
            truth_index = int(x[1:])
            prediction_index = int(y[1:])

        # print(x,y, truth_index, prediction_index)

        truth_showers_found[truth_index] = prediction_index
        predicted_showers_found[prediction_index] = truth_index
        predicted_showers_representative_index[prediction_index] = representative_indices[prediction_index]


    if use_ticl:
        print('match ticl')
        truth_to_ticl, ticl_to_truth = match_to_truth(truth_id, truth_shower_id, truth_shower_energy, truth_shower_eta, ticl_id, pred_ccoords,
                       representative_indices, hit_energy)


    num_found = 0.
    num_found_ticl = 0.
    num_missed = 0.
    num_missed_ticl = 0.
    num_gt_showers = 0.

    num_fakes = 0.
    num_fakes_ticl = 0.
    num_predicted_showers = 0.
    num_predicted_showers_ticl = 0.

    predicted_truth_total = 0
    
    print('record results')

    for k,v in truth_showers_found.items():
        results_dict['truth_shower_energies'].append(truth_showers_found_e[k])
        results_dict['truth_shower_energies_sum'].append(truth_showers_found_e_sum[k])
        predicted_truth_total += truth_showers_found_e[k]

        results_dict['truth_shower_etas'].append(truth_showers_found_eta[k])
        results_dict['truth_shower_local_density'].append(truth_showers_found_local_density[k])
        results_dict['truth_shower_closest_particle_distance'].append(truth_showers_found_closest_particle_distance[k])

        if v > -1:
            predicted_energy = pred_energy[predicted_showers_representative_index[v]]

            results_dict['truth_showers_found_or_not'].append(True)
            results_dict['truth_shower_matched_energies_sum'].append(np.sum(hit_energy[pred_sid==v]))
            results_dict['truth_shower_matched_energies_regressed'].append(predicted_energy)

            num_found += 1
        else:
            results_dict['truth_showers_found_or_not'].append(False)
            results_dict['truth_shower_matched_energies_sum'].append(-1)
            results_dict['truth_shower_matched_energies_regressed'].append(-1)

            num_missed += 1


        if use_ticl:
            ticl_found = truth_to_ticl[k]

            if ticl_found > -1:
                predicted_energy = ticl_energy[ticl_id==ticl_found][0]

                results_dict['truth_showers_found_or_not_ticl'].append(True)
                results_dict['truth_shower_matched_energies_sum_ticl'].append(np.sum(hit_energy[ticl_id==v]))
                results_dict['truth_shower_matched_energies_regressed_ticl'].append(predicted_energy)

                num_found_ticl += 1
            else:
                results_dict['truth_showers_found_or_not_ticl'].append(False)
                results_dict['truth_shower_matched_energies_sum_ticl'].append(-1)
                results_dict['truth_shower_matched_energies_regressed_ticl'].append(-1)


                num_missed_ticl += 1

        num_gt_showers += 1


    results_dict['truth_showers_sample_id'] = (window_id + 0*np.array(results_dict['truth_shower_energies'])).tolist()


    # results_dict['found_showers_predicted_energies'] = []
    # results_dict['found_showers_target_energies'] = []
    # results_dict['found_showers_predicted_sum'] = []
    # results_dict['found_showers_truth_sum'] = []
    # results_dict['found_showers_predicted_phi'] = []
    # results_dict['found_showers_target_phi'] = []
    # results_dict['found_showers_predicted_eta'] = []
    # results_dict['found_showers_target_eta'] = []



    results_dict['predicted_showers_regressed_energy'] = []
    results_dict['predicted_showers_matched_energy'] = []
    results_dict['predicted_showers_predicted_energy_sum'] = []
    results_dict['predicted_showers_matched_energy_sum'] = []
    results_dict['predicted_showers_regressed_phi'] = []
    results_dict['predicted_showers_matched_phi'] = []
    results_dict['predicted_showers_regressed_eta'] = []
    results_dict['predicted_showers_matched_eta'] = []
    results_dict['predicted_showers_sample_id'] = []


    results_dict['ticl_showers_regressed_energy'] = []
    results_dict['ticl_showers_matched_energy'] = []
    results_dict['ticl_showers_predicted_energy_sum'] = []
    results_dict['ticl_showers_matched_energy_sum'] = []
    results_dict['ticl_showers_regressed_phi'] = []
    results_dict['ticl_showers_matched_phi'] = []
    results_dict['ticl_showers_regressed_eta'] = []
    results_dict['ticl_showers_matched_eta'] = []
    results_dict['ticl_showers_sample_id'] = []


    predicted_ticl_total = 0
    if use_ticl:
        for k, v in ticl_to_truth.items():
            shower_energy_predicted = ticl_energy[ticl_id==k][0]
            predicted_ticl_total += shower_energy_predicted
            
            shower_eta_predicted = 0#pred_x[filtered_index_predicted]
            shower_phi_predicted = 0#pred_y[filtered_index_predicted]
            shower_energy_sum_predicted = np.sum(hit_energy[ticl_id == k])

            results_dict['ticl_showers_regressed_energy'].append(shower_energy_predicted)
            results_dict['ticl_showers_predicted_energy_sum'].append(shower_energy_sum_predicted)
            results_dict['ticl_showers_regressed_phi'].append(shower_phi_predicted)
            results_dict['ticl_showers_regressed_eta'].append(shower_eta_predicted)

            if v > -1:
                shower_energy_truth = truth_energy[truth_id == v][0]
                shower_eta_truth = truth_x[truth_id == v][0]
                shower_phi_truth = truth_y[truth_id == v][0]
                shower_energy_sum_truth = np.sum(hit_energy[truth_id == v])


                results_dict['ticl_showers_matched_energy'].append(shower_energy_truth)
                results_dict['ticl_showers_matched_energy_sum'].append(shower_energy_sum_truth)
                results_dict['ticl_showers_matched_phi'].append(shower_phi_truth)
                results_dict['ticl_showers_matched_eta'].append(shower_eta_truth)

                # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

            else:
                num_fakes_ticl += 1
                results_dict['ticl_showers_matched_energy'].append(-1)
                results_dict['ticl_showers_matched_energy_sum'].append(-1)
                results_dict['ticl_showers_matched_phi'].append(-1)
                results_dict['ticl_showers_matched_eta'].append(-1)

            num_predicted_showers_ticl += 1

        results_dict['ticl_showers_sample_id'] = (window_id + 0 * np.array(results_dict['ticl_showers_regressed_energy'])).tolist()

        # try:
        #     print("T", num_found_ticl / num_gt_showers, num_missed_ticl / num_gt_showers, num_fakes_ticl / num_predicted_showers_ticl)
        # except ZeroDivisionError:
        #     pass


    predicted_total_obc = 0

    for k, v in predicted_showers_found.items():
        filtered_index_predicted = predicted_showers_representative_index[k]

        shower_energy_predicted = pred_energy[filtered_index_predicted]
        predicted_total_obc += shower_energy_predicted

        shower_eta_predicted = pred_x[filtered_index_predicted]
        shower_phi_predicted = pred_y[filtered_index_predicted]
        shower_energy_sum_predicted = np.sum(hit_energy[pred_sid==k])

        results_dict['predicted_showers_regressed_energy'].append(shower_energy_predicted)
        results_dict['predicted_showers_predicted_energy_sum'].append(shower_energy_sum_predicted)
        results_dict['predicted_showers_regressed_phi'].append(shower_phi_predicted)
        results_dict['predicted_showers_regressed_eta'].append(shower_eta_predicted)

        if v > -1:
            shower_energy_truth = truth_energy[truth_id==v][0]
            shower_eta_truth = truth_x[truth_id==v][0]
            shower_phi_truth = truth_y[truth_id==v][0]
            shower_energy_sum_truth = np.sum(hit_energy[truth_id==v])

            # results_dict['found_showers_predicted_sum'].append(shower_energy_sum_predicted)
            # results_dict['found_showers_truth_sum'].append(shower_energy_sum_truth)
            # results_dict['found_showers_predicted_energies'].append(shower_energy_predicted)
            # results_dict['found_showers_target_energies'].append(shower_energy_truth)
            # results_dict['found_showers_predicted_phi'].append(shower_phi_predicted)
            # results_dict['found_showers_target_phi'].append(shower_phi_truth)
            # results_dict['found_showers_predicted_eta'].append(shower_eta_predicted)
            # results_dict['found_showers_target_eta'].append(shower_eta_truth)

            results_dict['predicted_showers_matched_energy'].append(shower_energy_truth)
            results_dict['predicted_showers_matched_energy_sum'].append(shower_energy_sum_truth)
            results_dict['predicted_showers_matched_phi'].append(shower_phi_truth)
            results_dict['predicted_showers_matched_eta'].append(shower_eta_truth)

            # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

        else:
            num_fakes += 1
            results_dict['predicted_showers_matched_energy'].append(-1)
            results_dict['predicted_showers_matched_energy_sum'].append(-1)
            results_dict['predicted_showers_matched_phi'].append(-1)
            results_dict['predicted_showers_matched_eta'].append(-1)


        num_predicted_showers += 1

    results_dict['predicted_showers_sample_id'] = (
                window_id + 0 * np.array(results_dict['predicted_showers_regressed_energy'])).tolist()
    # try:
    #     print(num_found / num_gt_showers, num_missed / num_gt_showers, num_fakes / num_predicted_showers)
    # except ZeroDivisionError:
    #     pass

    results_dict['num_real_showers'] = num_gt_showers
    results_dict['num_found_showers'] = num_found
    results_dict['num_missed_showers'] = num_missed
    results_dict['num_fake_showers'] = num_fakes
    results_dict['num_predicted_showers'] = num_predicted_showers


    results_dict['num_found_showers_ticl'] = num_found_ticl
    results_dict['num_missed_showers_ticl'] = num_missed_ticl
    results_dict['num_fake_showers_ticl'] = num_fakes_ticl
    results_dict['num_predicted_showers_ticl'] = num_predicted_showers_ticl

    results_dict['predicted_total_obc'] = predicted_total_obc
    results_dict['predicted_total_ticl'] = predicted_ticl_total
    results_dict['predicted_total_truth'] = predicted_truth_total

    representative_coords = np.array(representative_coords)
    
    
    print('visualise')

    if should_return_visualization_data:
        vis_dict = build_window_visualization_dict()
        vis_dict['truth_showers'] = truth_id# replace(truth_id, replace_dictionary)

        replace_dictionary = copy.deepcopy(predicted_showers_found)
        replace_dictionary_2 = copy.deepcopy(replace_dictionary)
        start_secondary_indicing_from = np.max(truth_id) + 1
        for k, v in replace_dictionary.items():
            if v == -1 and k != -1:
                replace_dictionary_2[k] = start_secondary_indicing_from
                start_secondary_indicing_from += 1
        replace_dictionary = replace_dictionary_2

        # print("===============")
        # print(np.unique(pred_sid), replace_dictionary)
        vis_dict['predicted_showers'] = replace(pred_sid, replace_dictionary)

        # print(predicted_showers_found)

        # print(np.mean((pred_sid==-1)).astype(np.float))

        if use_ticl:
            replace_dictionary = copy.deepcopy(ticl_to_truth)
            replace_dictionary_2 = copy.deepcopy(replace_dictionary)
            start_secondary_indicing_from = np.max(truth_id) + 1
            for k, v in replace_dictionary.items():
                if v == -1 and k != -1:
                    replace_dictionary_2[k] = start_secondary_indicing_from
                    start_secondary_indicing_from += 1
            replace_dictionary = replace_dictionary_2

            # print(np.unique(ticl_id), replace_dictionary)
            vis_dict['ticl_showers'] = replace(ticl_id, replace_dictionary)
        else:
            vis_dict['ticl_showers'] = pred_sid*10

        vis_dict['pred_and_truth_dict'] = pred_and_truth_dict
        vis_dict['feature_dict'] = feature_dict

        vis_dict['coords_representatives'] = representative_coords
        vis_dict['identified_vertices'] = pred_sid


        results_dict['visualization_data'] = vis_dict

    window_id += 1
    
    print('analyse_one_window_cut end')
    return results_dict




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
    
    #make 100% sure the cast doesn't hit the fan
    hit_assigned_truth_id = tf.where(hit_assigned_truth_id<-0.1, hit_assigned_truth_id - 0.1, hit_assigned_truth_id+0.1)
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
            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s, prediction_s, beta_threshold, distance_threshold, True, soft=soft)
        else:
            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s, prediction_s, beta_threshold, distance_threshold, False, soft=soft)

        append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
        num_visualized_segments += 1


    i += 1





def main(files, pdfpath, dumppath,soft):
    global dataset_analysis_dict
    for file in files:
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'], soft=soft)
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
    parser.add_argument('output', help='Output directory with .bin.gz files (all will be analysed) or a text file containing lest of those which are to be analysed')
    parser.add_argument('-p', help='Name of the output file (you have to manually append .pdf). Otherwise will be produced in the output directory.', default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-v', help='Visualize number of showers', default='10')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
    parser.add_argument('--analysisoutpath', help='Can be used to remake plots. Will dump analysis to a file.', default='')
    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_segments_to_visualize = int(args.v)
    num_visualized_segments = 0
    dataset_analysis_dict = build_dataset_analysis_dict()

    beta_threshold = beta_threshold*0 + float(args.b)
    distance_threshold = distance_threshold*0 + float(args.d)
    iou_threshold = iou_threshold*0 + float(args.i)

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

