#!/usr/bin/env python3
from experiment_database_manager import ExperimentDatabaseManager
import os
import uuid

import mgzip

import sql_credentials
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import numpy

numpy.set_printoptions(threshold=500)
import numpy as np

np.set_printoptions(threshold=500)
numpy.set_printoptions(threshold=500)
import argparse
import matplotlib.pyplot as plt
import gzip
import pickle
from matching_and_analysis import build_dataset_analysis_dict, append_endcap_dict_to_dataset_dict

import tensorflow as tf

from matching_and_analysis import do_analysis_plots_simplified_to_pdf, get_analysis_plotting_configuration
from matching_and_analysis import analyse_hgcal_endcap
import time




x = tf.constant([1,2,3])


num_rechits_per_segment = []
num_rechits_per_shower = []

endcap_id = 0
def analyse_one_file(file_data, soft=False):
    global num_visualized_segments, num_segments_to_visualize
    global dataset_analysis_dict, endcap_id

    for endcap_features, endcap_truth, endcap_predictions in file_data:
        print("Endcap ", endcap_id)
        if num_visualized_segments < num_segments_to_visualize:
            endcap_analysis_dict = analyse_hgcal_endcap(endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,
                                                        endcap_id, True, soft=soft)
        else:
            endcap_analysis_dict = analyse_hgcal_endcap(endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,
                                                        endcap_id, False, soft=soft)

        append_endcap_dict_to_dataset_dict(dataset_analysis_dict, endcap_analysis_dict)
        num_visualized_segments += 1
        endcap_id += 1


from matching_and_analysis import analyse_multiple_endcaps_multi_cpu, analyse_multiple_endcaps_single_cpu
def main(files, pdfpath, dumppath, soft, database_table_prefix, run_for=-1):
    global dataset_analysis_dict, fake_max_iou_values
    file_index = 0

    t1 = time.time()
    for file in files:
        print("\nFILE\n", file_index)
        with gzip.open(file, 'rb') as f:
            data_loaded = pickle.load(f)
            # print("XYZ", len(data_dict['features']), len(data_dict['predicted']), len(data_dict['truth']))
            file_results = analyse_multiple_endcaps_multi_cpu(data_loaded, soft=soft, beta_threshold=beta_threshold, distance_threshold=distance_threshold, iou_threshold=iou_threshold)
            for r in file_results:
                append_endcap_dict_to_dataset_dict(dataset_analysis_dict, r)
            # analyse_one_file(data_loaded, soft=soft)
            if file_index == run_for-1:
                break
            file_index += 1

    print("It took", time.time()-t1,"seconds")

    if len(dumppath) > 0:
        print("Dumping analysis to bin file", dumppath)

        with mgzip.open(dumppath, 'wb', thread=8, blocksize=2*10**7) as f:
            pickle.dump(dataset_analysis_dict, f)
    else:
        print("WARNING: No analysis output path specified. Skipped dumping of analysis.")

    # print("Number of total fakes is ", num_total_fakes)

    # np.savetxt('max_fake_iou.txt', fake_max_iou_values, delimiter=',')
    # 0/0

    plotter = HGCalAnalysisPlotter()

    plotter.add_data_from_analysis_dict(dataset_analysis_dict)
    if len(pdfpath) != 0:
        plotter.write_to_pdf(pdfpath)

    if len(database_table_prefix) != 0:
        print("Will write plots to database")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.set_experiment('analysis_plotting_experiments')
        plotter.write_data_to_database(database_manager, database_table_prefix)
        database_manager.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('output',
                        help='Output directory with .bin.gz files or a txt file with full paths of the bin gz files')
    parser.add_argument('-p',
                        help='Path of analysis pdf file (otherwise, it won\'t be produced)',
                        default='')
    parser.add_argument('-database_table_prefix',
                        help='Database table prefix if you wish to write plots to the database. Leave empty if you don\'t wanna write to database',
                        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-v', help='Visualize number of showers', default='10')
    parser.add_argument('-n', help='Use number of files', default='-1')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
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
    dataset_analysis_dict['istoyset'] = False


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
    pdfpath = ''
    if len(args.p) != 0:
        pdfpath = args.p


    main(files_to_be_tested, pdfpath, args.analysisoutpath, soft=args.soft, database_table_prefix=args.database_table_prefix, run_for=n_files)


