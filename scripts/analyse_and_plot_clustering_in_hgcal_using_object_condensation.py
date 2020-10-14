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
import index_dicts
import time

from ragged_plotting_tools import make_plots_from_object_condensation_clustering_analysis
from ragged_plotting_tools import analyse_one_window_cut

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply

from numba import jit

# tf.compat.v1.disable_eager_execution()


# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


ragged_constructor = RaggedConstructTensor()




num_rechits_per_segment = []
num_rechits_per_shower = []

window_id = 0
def analyse_one_file(features, predictions, truth_in, soft=False):
    global num_visualized_segments, num_segments_to_visualize
    global dataset_analysis_dict, window_id

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
                                                          beta_threshold, distance_threshold, iou_threshold, window_id, True, soft=soft)
        else:
            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s, prediction_s,
                                                          beta_threshold, distance_threshold, iou_threshold, window_id, False, soft=soft)

        append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
        num_visualized_segments += 1
        window_id += 1

    i += 1


def main(files, pdfpath, dumppath, soft):
    global dataset_analysis_dict, fake_max_iou_values
    file_index = 0
    for file in files:
        print("\nFILE\n", file_index)
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'], soft=soft)
            file_index += 1
            if file_index == 3:
                break

    if len(dumppath) > 0:
        print("Dumping analysis to bin file", dumppath)

        with open(dumppath, 'wb') as f:
            pickle.dump(dataset_analysis_dict, f)
    else:
        print("WARNING: No analysis output path specified. Skipped dumping of analysis.")

    print("Number of total fakes is ", num_total_fakes)

    np.savetxt('max_fake_iou.txt', fake_max_iou_values, delimiter=',')
    # 0/0
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
    dataset_analysis_dict['beta_threshold'] = beta_threshold
    dataset_analysis_dict['iou_threshold'] = iou_threshold
    dataset_analysis_dict['soft'] = bool(args.soft)


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


