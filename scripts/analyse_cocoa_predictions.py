#!/usr/bin/env python3
"""
Analysis script to run of the predictions of the model.
"""

import pdb
import os
import argparse
import pickle
import gzip
import mgzip
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from OCHits2Showers import OCHits2ShowersLayer, OCHits2ShowersLayer_HDBSCAN
from OCHits2Showers import process_endcap2, OCGatherEnergyCorrFac2, OCGatherEnergyCorrFac_new
from ShowersMatcher2 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
import extra_plots as ep
from visualize_event import dataframe_to_plot, matched_plot
from plot_cocoa import plot_everything


def analyse(preddir,
            pdfpath,
            beta_threshold,
            distance_threshold,
            iou_threshold,
            matching_mode,
            analysisoutpath,
            nfiles,
            nevents,
            local_distance_scaling,
            de_e_cut,
            angle_cut,
            extra=False,
            shower0=False,
        ):
   
    hits2showers = OCHits2ShowersLayer(
            beta_threshold,
            distance_threshold,
            local_distance_scaling)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut, shower0)


    files_to_be_tested = [
        os.path.join(preddir, x)
        for x in os.listdir(preddir)
        if (x.endswith('.bin.gz') and x.startswith('pred'))]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    features = []
    truth = []
    prediction = []
    alpha_ids = []
    matched = []
    event_id = 0

    ###############################################################################################
    ### Loop over all events ######################################################################
    ###############################################################################################

    for i, file in enumerate(files_to_be_tested):
        print(f"Analysing file {i+1}/{len(files_to_be_tested)}")
        with mgzip.open(file, 'rb') as analysis_file:
            file_data = pickle.load(analysis_file)
        for j, endcap_data in enumerate(file_data):
            if (nevents != -1) and (j > nevents):
                continue
            # print(f"Analysing endcap {j+1}/{len(file_data)}")
            features_dict, truth_dict, predictions_dict = endcap_data
            features.append(features_dict)            
            truth.append(truth_dict)

            print(f"Analyzing event {event_id}")

            pred_sid, _, alpha_idx, _, _  = hits2showers(
            predictions_dict['pred_ccoords'],
            predictions_dict['pred_beta'],
            predictions_dict['pred_dist'])
            
            predictions_dict['pred_sid'] = pred_sid
            prediction.append(predictions_dict)
            
            
            if not isinstance(alpha_idx, np.ndarray):
                alpha_idx = alpha_idx.numpy()
            alpha_idx = np.reshape(alpha_idx, newshape=(-1,))

            alpha_ids.append(alpha_idx)
            
            showers_matcher.set_inputs(
                features_dict=features_dict,
                truth_dict=truth_dict,
                predictions_dict=predictions_dict,
                pred_alpha_idx=alpha_idx
            )
            showers_matcher.process(extra=extra)
            dataframe = showers_matcher.get_result_as_dataframe()
            matched_truth_sid, matched_pred_sid = showers_matcher.get_matched_hit_sids()
            matched.append((matched_truth_sid, matched_pred_sid))

            dataframe['event_id'] = event_id
            dataframe['pred_id_value'] = dataframe['pred_id'].apply(lambda x: np.argmax(x))
            showers_dataframe = pd.concat((showers_dataframe, dataframe))
            
            event_id += 1

    ###############################################################################################
    ### Write to file #############################################################################
    ###############################################################################################
    if len(analysisoutpath) > 0:
        print("Writing dataframes to pickled file", analysisoutpath)
        
        scalar_variables = {
        'beta_threshold': str(beta_threshold),
        'distance_threshold': str(distance_threshold),
        'iou_threshold': str(iou_threshold),
        'matching_mode': str(matching_mode),
        'de_e_cut': str(de_e_cut),
        'angle_cut': str(angle_cut),
        }
        
        analysis_data = {
            'showers_dataframe' : showers_dataframe,
            'scalar_variables' : scalar_variables,
            'alpha_ids'        : alpha_ids,
            'matched': matched,
        }
        
        analysis_data['features'] = features
        analysis_data['truth'] = truth
        analysis_data['prediction'] = prediction

        with gzip.open(analysisoutpath, 'wb') as output_file:
            print("Writing dataframes to pickled file",analysisoutpath)
            pickle.dump(analysis_data, output_file)
            
    ###############################################################################################
    ### Make Plots ################################################################################
    ###############################################################################################
    
    if(len(pdfpath) > 0):
        print('Starting to plot', pdfpath)
        plot_everything(showers_dataframe, pdfpath)
    print("DONE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('preddir',
        help='Directory with .bin.gz files or a txt file with full paths of the \
            bin-gz files from the prediction.')
    parser.add_argument('-p',
        help="Output directory for the final analysis pdf file (otherwise, it won't be produced)",
        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-m', help='Matching mode', default='iou_max')
    parser.add_argument('--analysisoutpath',
        help='Will dump analysis data to a file to remake plots without re-running everything.',
        default='')
    parser.add_argument('--nfiles',
        help='Maximum number of files. -1 for everything in the preddir',
        default=-1)
    parser.add_argument('--nevents', help='Maximum number of events (per file)', default=-1)
    parser.add_argument('--no_local_distance_scaling', help='With local distance scaling',
        action='store_true')
    parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
    parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
    parser.add_argument('--extra',
        help="Calculate more information for showers",
        action='store_true')
    parser.add_argument('--shower0',
        help="Only match with truth shower ID=0",
        action='store_true')

    args = parser.parse_args()

    analyse(preddir=args.preddir,
            pdfpath=args.p,
            beta_threshold=float(args.b),
            distance_threshold=float(args.d),
            iou_threshold=float(args.i),
            matching_mode=args.m,
            analysisoutpath=args.analysisoutpath,
            nfiles=int(args.nfiles),
            local_distance_scaling=not args.no_local_distance_scaling,
            de_e_cut=float(args.de_e_cut),
            angle_cut=float(args.angle_cut),
            nevents=int(args.nevents),
            extra=args.extra,
            shower0=args.shower0
            )
    
