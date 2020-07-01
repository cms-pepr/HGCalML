#!/usr/bin/env python3

from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
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

from ragged_plotting_tools import make_plots_from_object_condensation_clustering_analysis

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


# tf.compat.v1.disable_eager_execution()


os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



ragged_constructor = RaggedConstructTensor()



def find_uniques_from_betas(betas, coords, dist_threshold):

    n2_distances = np.sqrt(np.sum(np.square(np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)), axis=-1))
    betas_checked = np.zeros_like(betas) - 1

    index = 0

    arange_vector = np.arange(len(betas))

    representative_indices = []

    while True:
        betas_remaining = betas[betas_checked==-1]
        arange_remaining = arange_vector[betas_checked==-1]

        if len(betas_remaining)==0:
            break

        max_beta = arange_remaining[np.argmax(betas_remaining)]

        representative_indices.append(max_beta)


        n2 = n2_distances[max_beta]

        distances_less = np.logical_and(n2<dist_threshold, betas_checked==-1)
        betas_checked[distances_less] = index

        index += 1


    return betas_checked, representative_indices

def replace(arr, rep_dict):
    """Assumes all elements of "arr" are keys of rep_dict"""

    # Removing the explicit "list" breaks python3
    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))

    idces = np.digitize(arr, rep_keys, right=True)
    # Notice rep_keys[digitize(arr, rep_keys, right=True)] == arr

    return rep_vals[idces]


def assign_prediction_labels_to_full_unfiltered_vertices(beta_all, clustering_coords_all, labels, clustering_coords_all_filtered, betas_filtered, distance_threshold):
    unique_labels = np.unique(labels)

    labels_all = np.zeros_like(beta_all) - 1

    centers = []
    labelsc = []

    replacedict = {}
    iii = 0

    replacedict[-1] = -1

    for x in unique_labels:
        if x == -1:
            continue
        center = clustering_coords_all_filtered[labels==x][np.argmax(betas_filtered[labels==x])]
        centers.append(center[np.newaxis, ...])
        labelsc.append(x)

        replacedict[iii] = x


        distances = np.sqrt(np.sum(np.square(clustering_coords_all - center[np.newaxis, ...]), axis=-1))
        labels_all[distances < distance_threshold] = x

        iii += 1

    # centers = np.concatenate(centers, axis=0)

    # labels_all = np.argmin(np.sqrt(np.sum(np.square(np.expand_dims(clustering_coords_all, axis=1) - np.expand_dims(centers, axis=0)), axis=-1)), axis=-1)
    labels_all = replace(labels_all, replacedict)

    return labels_all



# found_truth_energies_energies_truth = []
# found_truth_energies_is_it_found = []
#
# found_2_truth_energies_predicted_sum = []
# found_2_truth_energies_truth_sum = []
#
# found_2_truth_target_energies = []
# found_2_truth_predicted_energies = []
#
# found_2_truth_rotational_distance = []


beta_threshold = 0.1
distance_threshold = 0.5
iou_threshold = 0.1


def match_to_truth(truth_showers, unique_showers, unique_showers_e, unique_shower_eta, labels, clustering_coords_all_filtered, representative_indices, rechit_energies):
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

    print("HELLO")
    print(unique_labels)



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
            overlap = np.sum(rechit_energies * (truth_showers == unique_showers_this_segment[i]) * (labels == index_shower_prediction)) / np.sum(rechit_energies * np.logical_or((truth_showers == unique_showers_this_segment[i]), (labels == index_shower_prediction)))

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

        print(x,y, truth_index, prediction_index)

        truth_showers_found[truth_index] = prediction_index
        predicted_showers_found[prediction_index] = truth_index

    return truth_showers_found, predicted_showers_found


def analyse_one_window_cut(truth_showers_this_segment, x_this_segment, y_this_segment, pred_this_segment, beta_threshold, distance_threshold, should_return_visualization_data=False, input_log_energy=True):
    results_dict = build_window_analysis_dict()

    unique_showers_this_segment,unique_showers_indices = np.unique(truth_showers_this_segment, return_index=True)
    truth_energies_this_segment = y_this_segment[:, 1]
    unique_showers_energies = truth_energies_this_segment[unique_showers_indices]
    unique_showers_eta = y_this_segment[:, 9][unique_showers_indices]

    rechit_energies_this_segment = x_this_segment[:, 0]




    beta_all = pred_this_segment[:, -6]


    energy_regressed_all = pred_this_segment[:, -5]
    eta_regressed_all = pred_this_segment[:, -4] + x_this_segment[:, 1]
    phi_regressed_all = pred_this_segment[:, -3] + x_this_segment[:, 2]

    is_spectator = beta_all>beta_threshold


    beta_all_filtered = beta_all[is_spectator==1]

    # print(y_this_segment.shape)


    energy_truth_all = y_this_segment[:, 1]
    eta_truth_all = y_this_segment[:, 8]
    phi_truth_all = y_this_segment[:, 9]


    use_ticl = True
    try:
        ticl_instance_id = y_this_segment[:, 17]
        ticl_true_energy = y_this_segment[:, 18]
    except Exception:
        use_ticl = False



    energy_regressed_filtered = energy_regressed_all[is_spectator==1]
    eta_regressed_filtered = eta_regressed_all[is_spectator==1]
    phi_regressed_filtered = phi_regressed_all[is_spectator==1]

    energy_truth_filtered = energy_truth_all[is_spectator==1]
    eta_truth_filtered = eta_truth_all[is_spectator==1]
    phi_truth_filtered = phi_truth_all[is_spectator==1]

    y_all_filtered = y_this_segment[is_spectator==1]
    x_filtered = x_this_segment[is_spectator==1]

    prediction_filtered = pred_this_segment[is_spectator==1]


    clustering_coords_all = pred_this_segment[:, -2:]
    clustering_coords_all_filtered = clustering_coords_all[is_spectator==1, :]

    labels, representative_indices = find_uniques_from_betas(beta_all_filtered, clustering_coords_all_filtered, dist_threshold=distance_threshold)
    labels_for_all = assign_prediction_labels_to_full_unfiltered_vertices(beta_all, clustering_coords_all, labels, clustering_coords_all_filtered, beta_all_filtered, distance_threshold=distance_threshold)

    unique_labels = np.unique(labels_for_all)

    truth_showers_found = {}
    truth_showers_found_e = {}
    truth_showers_found_eta = {}
    iii = 0
    for x in unique_showers_this_segment:
        truth_showers_found[x] = -1
        truth_showers_found_e[x] = unique_showers_energies[iii]
        truth_showers_found_eta[x] = unique_showers_eta[iii]
        iii += 1
        results_dict['num_rechits_per_truth_shower'].append(len(truth_showers_this_segment[truth_showers_this_segment == x]))
    results_dict['num_rechits_per_window'] = len(truth_showers_this_segment)


    predicted_showers_found = {}
    predicted_showers_representative_index = {}
    for x in unique_labels:
        predicted_showers_found[x] = -1
        predicted_showers_representative_index[x] = -1


    print("12345")
    print(predicted_showers_found)
    print()

    representative_coords = []
    G = nx.Graph()
    for index_shower_prediction in range(len(representative_indices)):
        representative_coords.append(clustering_coords_all_filtered[representative_indices[index_shower_prediction]])

        for i in range(len(unique_showers_this_segment)):
            if unique_showers_this_segment[i] == -1:
                continue

            # sum_truth = np.sum(rechit_energies_this_segment * (truth_showers_this_segment == unique_showers_this_segment[i]))
            # sum_predicted = np.sum(rechit_energies_this_segment * (labels_for_all == index_shower_prediction))
            overlap = np.sum(rechit_energies_this_segment * (truth_showers_this_segment == unique_showers_this_segment[i]) * (labels_for_all == index_shower_prediction)) / np.sum(rechit_energies_this_segment * np.logical_or((truth_showers_this_segment == unique_showers_this_segment[i]), (labels_for_all == index_shower_prediction)))

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

        print(x,y, truth_index, prediction_index)

        truth_showers_found[truth_index] = prediction_index
        predicted_showers_found[prediction_index] = truth_index
        predicted_showers_representative_index[prediction_index] = representative_indices[prediction_index]


    print("Hello XYZ")
    print(predicted_showers_found)
    print()


    if use_ticl:
        truth_to_ticl, ticl_to_truth = match_to_truth(truth_showers_this_segment, unique_showers_this_segment, unique_showers_energies, unique_showers_eta, ticl_instance_id, clustering_coords_all_filtered,
                       representative_indices, rechit_energies_this_segment)


    num_found = 0.
    num_missed = 0.
    num_gt_showers = 0.

    num_fakes = 0.
    num_predicted_showers = 0.



    for k,v in truth_showers_found.items():
        results_dict['truth_shower_energies'].append(truth_showers_found_e[k])
        results_dict['truth_shower_etas'].append(truth_showers_found_eta[k])

        if v > -1:
            results_dict['truth_showers_found_or_not'].append(True)

            num_found += 1
        else:
            results_dict['truth_showers_found_or_not'].append(False)
            num_missed += 1

        num_gt_showers += 1


    results_dict['found_showers_predicted_energies'] = []
    results_dict['found_showers_target_energies'] = []
    results_dict['found_showers_predicted_sum'] = []
    results_dict['found_showers_truth_sum'] = []
    results_dict['found_showers_predicted_phi'] = []
    results_dict['found_showers_target_phi'] = []
    results_dict['found_showers_predicted_eta'] = []
    results_dict['found_showers_target_eta'] = []



    results_dict['predicted_showers_regressed_energy'] = []
    results_dict['predicted_showers_matched_energy'] = []
    results_dict['predicted_showers_predicted_energy_sum'] = []
    results_dict['predicted_showers_matched_energy_sum'] = []
    results_dict['predicted_showers_regressed_phi'] = []
    results_dict['predicted_showers_matched_phi'] = []
    results_dict['predicted_showers_regressed_eta'] = []
    results_dict['predicted_showers_matched_eta'] = []


    for k, v in predicted_showers_found.items():
        filtered_index_predicted = predicted_showers_representative_index[k]
        
        shower_energy_predicted = energy_regressed_filtered[filtered_index_predicted]
        if input_log_energy:
            shower_energy_predicted = np.exp(shower_energy_predicted) - 1
        
        shower_eta_predicted = eta_regressed_filtered[filtered_index_predicted]
        shower_phi_predicted = phi_regressed_filtered[filtered_index_predicted]
        shower_energy_sum_predicted = np.sum(rechit_energies_this_segment[labels_for_all==k])

        results_dict['predicted_showers_regressed_energy'].append(shower_energy_predicted)
        results_dict['predicted_showers_predicted_energy_sum'].append(shower_energy_sum_predicted)
        results_dict['predicted_showers_regressed_phi'].append(shower_phi_predicted)
        results_dict['predicted_showers_regressed_eta'].append(shower_eta_predicted)

        if v > -1:
            shower_energy_truth = energy_truth_all[truth_showers_this_segment==v][0]
            shower_eta_truth = eta_truth_all[truth_showers_this_segment==v][0]
            shower_phi_truth = phi_truth_all[truth_showers_this_segment==v][0]
            shower_energy_sum_truth = np.sum(rechit_energies_this_segment[truth_showers_this_segment==v])

            results_dict['found_showers_predicted_sum'].append(shower_energy_sum_predicted)
            results_dict['found_showers_truth_sum'].append(shower_energy_sum_truth)
            results_dict['found_showers_predicted_energies'].append(shower_energy_predicted)
            results_dict['found_showers_target_energies'].append(shower_energy_truth)
            results_dict['found_showers_predicted_phi'].append(shower_phi_predicted)
            results_dict['found_showers_target_phi'].append(shower_phi_truth)
            results_dict['found_showers_predicted_eta'].append(shower_eta_predicted)
            results_dict['found_showers_target_eta'].append(shower_eta_truth)

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

    try:
        print(num_found / num_gt_showers, num_missed / num_gt_showers, num_fakes / num_predicted_showers)
    except ZeroDivisionError:
        pass

    results_dict['num_real_showers'] = num_gt_showers
    results_dict['num_found_showers'] = num_found
    results_dict['num_missed_showers'] = num_missed
    results_dict['num_fake_showers'] = num_fakes
    results_dict['num_predicted_showers'] = num_predicted_showers

    representative_coords = np.array(representative_coords)

    if should_return_visualization_data:
        vis_dict = build_window_visualization_dict()
        vis_dict['truth_showers'] = truth_showers_this_segment# replace(truth_showers_this_segment, replace_dictionary)

        replace_dictionary = copy.deepcopy(predicted_showers_found)
        replace_dictionary_2 = copy.deepcopy(replace_dictionary)
        start_secondary_indicing_from = np.max(truth_showers_this_segment) + 1
        for k, v in replace_dictionary.items():
            if v == -1 and k != -1:
                replace_dictionary_2[k] = start_secondary_indicing_from
                start_secondary_indicing_from += 1
        replace_dictionary = replace_dictionary_2

        print("===============")
        print(np.unique(labels_for_all), replace_dictionary)
        vis_dict['predicted_showers'] = replace(labels_for_all, replace_dictionary)

        print(predicted_showers_found)

        print(np.mean((labels_for_all==-1)).astype(np.float))

        if use_ticl:
            replace_dictionary = copy.deepcopy(ticl_to_truth)
            replace_dictionary_2 = copy.deepcopy(replace_dictionary)
            start_secondary_indicing_from = np.max(truth_showers_this_segment) + 1
            for k, v in replace_dictionary.items():
                if v == -1 and k != -1:
                    replace_dictionary_2[k] = start_secondary_indicing_from
                    start_secondary_indicing_from += 1
            replace_dictionary = replace_dictionary_2

            print(np.unique(ticl_instance_id), replace_dictionary)
            vis_dict['ticl_showers'] = replace(ticl_instance_id, replace_dictionary)
        else:
            vis_dict['ticl_showers'] = labels_for_all*10

        vis_dict['x'] = x_this_segment
        vis_dict['y'] = y_this_segment
        vis_dict['prediction_all'] = pred_this_segment
        vis_dict['coords_representatives'] = representative_coords
        vis_dict['identified_vertices'] = labels_for_all


        results_dict['visualization_data'] = vis_dict

    return results_dict




num_rechits_per_segment = []
num_rechits_per_shower = []


def analyse_one_file(features, predictions, truth, input_log_energy=True):
    global num_visualized_segments, num_segments_to_visualize
    global dataset_analysis_dict

    predictions = tf.constant(predictions[0])

    row_splits = features[1][:, 0]

    x_data, _ = ragged_constructor((features[0], row_splits))
    y_data, _ = ragged_constructor((truth[0], row_splits))
    truth_showers, row_splits = ragged_constructor((truth[0][:, 0][..., tf.newaxis], row_splits))

    truth_showers = tf.cast(truth_showers[:, 0], tf.int32)

    num_unique = []
    shower_sizes = []

    for i in range(len(row_splits) - 1):
        truth_showers_this_segment = truth_showers[row_splits[i]:row_splits[i + 1]].numpy()
        x_this_segment = x_data[row_splits[i]:row_splits[i + 1]].numpy()
        y_this_segment = y_data[row_splits[i]:row_splits[i + 1]].numpy()
        pred_this_segment = predictions[row_splits[i]:row_splits[i + 1]].numpy()

        if num_visualized_segments < num_segments_to_visualize:
            window_analysis_dict = analyse_one_window_cut(truth_showers_this_segment, x_this_segment, y_this_segment, pred_this_segment, beta_threshold, distance_threshold, True, input_log_energy=input_log_energy)
        else:
            window_analysis_dict = analyse_one_window_cut(truth_showers_this_segment, x_this_segment, y_this_segment, pred_this_segment, beta_threshold, distance_threshold, False, input_log_energy=input_log_energy)

        append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
        num_visualized_segments += 1


    i += 1





def main(files, pdfpath, dumppath,input_log_energy=True):
    global dataset_analysis_dict
    for file in files:
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'], input_log_energy=input_log_energy)
            break

    make_plots_from_object_condensation_clustering_analysis(pdfpath, dataset_analysis_dict)

    if len(dumppath) > 0:
        print("Dumping analysis to bin file", dumppath)

        with open(dumppath, 'wb') as f:
            pickle.dump(dataset_analysis_dict, f)
    else:
        print("WARNING: No analysis output path specified. Skipped dumping of analysis.")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('output', help='Output directory with .bin.gz files (all will be analysed) or a text file containing lest of those which are to be analysed')
    parser.add_argument('-p', help='Name of the output file (you have to manually append .pdf). Otherwise will be produced in the output directory.', default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-v', help='Visualize number of showers', default='10')
    parser.add_argument('--notlogE', help='prediction is linear energy not log(E+1)', action='store_true')
    parser.add_argument('--analysisoutpath', help='Can be used to remake plots. Will dump analysis to a file.', default='')
    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    input_log_energy = not args.notlogE
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

    main(files_to_be_tested, pdfpath, args.analysisoutpath, input_log_energy=input_log_energy)

