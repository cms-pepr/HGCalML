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

from ragged_plotting_tools import do_analysis_plots_to_pdf, get_analysis_plotting_configuration
from ragged_plotting_tools import analyse_window_cut
from datastructures import TrainData_OC

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
def analyse_one_file(_features, predictions, truth_in, soft=False):
    global num_visualized_segments, num_segments_to_visualize
    global dataset_analysis_dict, window_id


    # predictions = tf.constant(predictions[0])

    row_splits = _features[1][:, 0]

    features = _features[0]

    truth_idx = _features[2].astype(np.int32)
    truth_energy = _features[4]
    truth_position = _features[6]
    truth_time = _features[8]
    truth_pid = _features[10]


    pred_beta = predictions[0]
    pred_ccoords = predictions[1]
    pred_energy = predictions[2]
    pred_position = predictions[3]
    pred_time = predictions[4]
    pred_pid = predictions[5]

    _ , row_splits = ragged_constructor((_features[10], row_splits))

    truth_all = truth_in[0]


    num_unique = []
    shower_sizes = []

    # here ..._s refers to quantities per window/segment
    #
    for i in range(len(row_splits) - 1):
        features_s = features[row_splits[i]:row_splits[i + 1]]
        truth_idx_s = truth_idx[row_splits[i]:row_splits[i + 1]]
        truth_energy_s = truth_energy[row_splits[i]:row_splits[i + 1]]
        truth_position_s = truth_position[row_splits[i]:row_splits[i + 1]]
        truth_time_s = truth_time[row_splits[i]:row_splits[i + 1]]
        truth_pid_s = truth_pid[row_splits[i]:row_splits[i + 1]]

        pred_beta_s = pred_beta[row_splits[i]:row_splits[i + 1]]
        pred_ccoords_s = pred_ccoords[row_splits[i]:row_splits[i + 1]]
        pred_energy_s = pred_energy[row_splits[i]:row_splits[i + 1]]
        pred_position_s = pred_position[row_splits[i]:row_splits[i + 1]]
        pred_time_s = pred_time[row_splits[i]:row_splits[i + 1]]
        pred_pid_s = pred_pid[row_splits[i]:row_splits[i + 1]]
        truth_s = truth_all[row_splits[i]:row_splits[i + 1]]

        td = TrainData_OC()  # contains all dicts
        analysis_input = dict()
        analysis_input["feat_all"] = td.createFeatureDict(features_s, addxycomb=False)
        analysis_input["truth_sid"] = truth_idx_s
        analysis_input["truth_energy"] = truth_energy_s
        analysis_input["truth_position"] = truth_position_s
        analysis_input["truth_time"] = truth_time_s
        analysis_input["truth_pid"] = truth_pid_s

        analysis_input["truth_all"] = td.createTruthDict(truth_s)

        analysis_input["pred_beta"] = pred_beta_s
        analysis_input["pred_ccoords"] = pred_ccoords_s
        analysis_input["pred_energy"] = pred_energy_s
        analysis_input["pred_position"] = pred_position_s
        analysis_input["pred_time"] = pred_time_s
        analysis_input["pred_pid"] = pred_pid_s

        if num_visualized_segments < num_segments_to_visualize:
            window_analysis_dict = analyse_window_cut(analysis_input, beta_threshold, distance_threshold, iou_threshold,
                                                      window_id, True, soft=soft)
        else:
            window_analysis_dict = analyse_window_cut(analysis_input, beta_threshold, distance_threshold, iou_threshold,
                                                      window_id, False, soft=soft)

        append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
        num_visualized_segments += 1
        window_id += 1

    i += 1


def main(files, pdfpath, dumppath, soft, run_for=-1):
    global dataset_analysis_dict, fake_max_iou_values
    file_index = 0
    for file in files:
        print("\nFILE\n", file_index)
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            # print("XYZ", len(data_dict['features']), len(data_dict['predicted']), len(data_dict['truth']))
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'], soft=soft)
            file_index += 1
            if file_index == run_for-1:
                break

    if len(dumppath) > 0:
        print("Dumping analysis to bin file", dumppath)

        with open(dumppath, 'wb') as f:
            pickle.dump(dataset_analysis_dict, f)
    else:
        print("WARNING: No analysis output path specified. Skipped dumping of analysis.")

    # print("Number of total fakes is ", num_total_fakes)

    # np.savetxt('max_fake_iou.txt', fake_max_iou_values, delimiter=',')
    # 0/0
    istoyset = dataset_analysis_dict['istoyset']
    do_analysis_plots_to_pdf(pdfpath, dataset_analysis_dict, plotting_config=get_analysis_plotting_configuration('toy_set_without_ticl' if istoyset else 'standard_hgcal_without_ticl'))


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
    parser.add_argument('-n', help='Use number of files', default='-1')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
    parser.add_argument('--istoyset', help='if its the toyset', action='store_true')
    parser.add_argument('--analysisoutpath', help='Can be used to remake plots. Will dump analysis to a file.',
                        default='')
    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_segments_to_visualize = int(args.v)
    num_visualized_segments = 0
    dataset_analysis_dict = build_dataset_analysis_dict()

    beta_threshold = float(args.b)
    distance_threshold = float(args.d)
    iou_threshold = float(args.i)
    n_files = int(args.n)

    dataset_analysis_dict['distance_threshold'] = distance_threshold
    dataset_analysis_dict['beta_threshold'] = beta_threshold
    dataset_analysis_dict['iou_threshold'] = iou_threshold
    dataset_analysis_dict['soft'] = bool(args.soft)
    dataset_analysis_dict['istoyset'] = bool(args.istoyset)


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

    main(files_to_be_tested, pdfpath, args.analysisoutpath, soft=args.soft, run_for=n_files)


