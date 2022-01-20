#!/usr/bin/env python3

raise NotImplementedError('Needs to be revamped with the new code. To be done soon')
import os
import gzip
import pickle

import mgzip

import matching_and_analysis
import argparse
import hplots.hgcal_analysis_plotter as hp
import sql_credentials
from experiment_database_manager import ExperimentDatabaseManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Visualize predictions from object condensation and plot relevant results')
    parser.add_argument('file',
                        help='A .bin.gz file to visualize.')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-n', help='nth endcap to visualize in this file', default='3')
    parser.add_argument('-m', help='Matching type. Look at matching_analysis.py to find the matching type.', default='0')
    parser.add_argument('-npasses', help='Number of passes if multi pass matching algorithm is used.', default='1')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
    parser.add_argument('--html', help='Write to html in plotly instead of visualizing', default='')
    parser.add_argument('--et', help='Energy type. See matching_and_analysis.py for options. Control+F for \'ENERGY_GATHER_TYPE_PRED_ENERGY\'', default='1')

    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_visualized_segments = 0

    beta_threshold = float(args.b)
    distance_threshold = float(args.d)
    iou_threshold = float(args.i)

    matching_type = int(args.m)
    energy_gather_type = int(args.et)

    # matching_type = matching_and_analysis.MATCHING_TYPE_IOU_MAX if matching_type==0 else matching_and_analysis.MATCHING_TYPE_MAX_FOUND

    metadata = matching_and_analysis.build_metadeta_dict(beta_threshold=beta_threshold,
                                                         distance_threshold=distance_threshold,
                                                         iou_threshold=iou_threshold,
                                                         matching_type=matching_type,
                                                         with_local_distance_scaling=False,
                                                         passes=int(args.npasses),
                                                         energy_gather_type=energy_gather_type
                                                         )

    with gzip.open(args.file, 'rb') as f:
        data_loaded = pickle.load(f)
    endcap_data = data_loaded[int(args.n)]
    graph_analyzer = matching_and_analysis.OCRecoGraphAnalyzer(metadata)
    graph_analyzer.analyse(endcap_data[0], endcap_data[2], endcap_data[1], return_rechit_data=True)
    visualizer = matching_and_analysis.OCMatchingVisualizer(graph_analyzer.non_reduced_graph)
    if len(args.html) == 0:
        visualizer.visualize()
    else:
        visualizer.write_to_html(args.html)




