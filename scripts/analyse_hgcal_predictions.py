#!/usr/bin/env python3
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
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('inputdir',
                        help='Directory with .bin.gz files or a txt file with full paths of the bin gz files from the prediction.')
    parser.add_argument('-p',
                        help='Output directory for the final analysis pdf file (otherwise, it won\'t be produced)',
                        default='')
    parser.add_argument('--database_table_prefix',
                        help='Database table prefix if you wish to write plots to the database. Leave empty if you don\'t wanna write to database',
                        default='')
    parser.add_argument('--database_file', help='database file, otherwise remote server',
                        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-v', help='Ignore, functionality removed', default='0')
    parser.add_argument('-m', help='Matching type. 0 for IOU based matching, 1 for f score based matching', default='2')
    parser.add_argument('--et', help='Energy type. See matching_and_analysis.py for options. Control+F for \'ENERGY_GATHER_TYPE_PRED_ENERGY\'', default='1')
    parser.add_argument('--soft', help='uses soft object condensation', action='store_true')
    parser.add_argument('--analysisoutpath', help='Will dump analysis to a file to remake plots without re-running everything.',
                        default='')
    parser.add_argument('--local_distance_scaling', help='With local distance scaling', action='store_true')
    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_segments_to_visualize = int(args.v)
    num_visualized_segments = 0

    beta_threshold = float(args.b)
    distance_threshold = float(args.d)
    iou_threshold = float(args.i)
    database_table_prefix = args.database_table_prefix

    matching_type = int(args.m)
    # matching_type = matching_and_analysis.MATCHING_TYPE_IOU_MAX if matching_type==0 else matching_and_analysis.MATCHING_TYPE_MAX_FOUND

    energy_gather_type = int(args.et)

    metadata = matching_and_analysis.build_metadeta_dict(beta_threshold=beta_threshold,
                                                         distance_threshold=distance_threshold,
                                                         iou_threshold=iou_threshold,
                                                         matching_type=matching_type,
                                                         with_local_distance_scaling=args.local_distance_scaling,
                                                         energy_gather_type=energy_gather_type
                                                         )

    files_to_be_tested = []
    pdfpath = ''
    if os.path.isdir(args.inputdir):
        for x in os.listdir(args.inputdir):
            if x.endswith('.bin.gz'):
                files_to_be_tested.append(os.path.join(args.inputdir, x))
        pdfpath = args.inputdir
    elif os.path.isfile(args.inputdir):
        with open(args.inputdir) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        files_to_be_tested = [x.strip() for x in content]
        pdfpath = os.path.split(pdfpath)[0]
    else:
        raise Exception('Error: couldn\'t locate input folder/file')

    print(files_to_be_tested)
    pdfpath = ''
    if len(args.p) != 0:
        pdfpath = args.p


    if False:
        all_data = []
        for file in files_to_be_tested:
            print("Reading", file)
            with mgzip.open(file, 'rb') as f:
                data_loaded = pickle.load(f)
                all_data.append(data_loaded)
        analysed_graphs, metadata = matching_and_analysis.OCAnlayzerWrapper(metadata).analyse_from_data(
            all_data)
    else:
        analysed_graphs, metadata = matching_and_analysis.OCAnlayzerWrapper(metadata).analyse_from_files(files_to_be_tested)
    plotter = hp.HGCalAnalysisPlotter()
    plotter.add_data_from_analysed_graph_list(analysed_graphs, metadata)
    if len(pdfpath) > 0:
        plotter.write_to_pdf(pdfpath=pdfpath)

    if len(args.analysisoutpath)!=0:
        with gzip.open(args.analysisoutpath, 'wb') as f:
            pickle.dump((analysed_graphs, metadata), f)

    if len(database_table_prefix) != 0:
        print("Will write plots to database")
        database_file = args.database_file

        if len(database_file) == 0:
            database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        else:
            database_manager = ExperimentDatabaseManager(file=database_file, cache_size=40)

        database_manager.set_experiment('analysis_plotting_experiments')
        plotter.write_data_to_database(database_manager, database_table_prefix)
        database_manager.close()




