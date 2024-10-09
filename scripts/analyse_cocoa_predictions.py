#!/usr/bin/env python3
"""
Analysis script adapted from analyse_hgcal_predictions to run of the predictions of the model for the COCOA dataset.
Call with python analyse_cocoa_predictions.py -h for help.
"""

import os
import argparse
import pickle
import gzip
import mgzip
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import fastjet as fj
from scipy.optimize import linear_sum_assignment

from OCHits2Showers import OCHits2ShowersLayer
from ShowersMatcher2 import ShowersMatcher
from plot_cocoa import plot_everything


def analyse(preddir,
            outputpath,
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
            datasetname='Quark Jet'
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
    jets_dataframe = pd.DataFrame()
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

            #Create showers
            pred_sid, _, alpha_idx, _, _  = hits2showers(
            predictions_dict['pred_ccoords'],
            predictions_dict['pred_beta'],
            predictions_dict['pred_dist'])
            
            predictions_dict['pred_sid'] = pred_sid
            prediction.append(predictions_dict)
            
            #Match showers
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
            
            ###############################################################################################
            ### Jet Clustering ############################################################################
            ###############################################################################################

            has_truth = np.isnan(dataframe['truthHitAssignedEnergies']) == False
            has_pred = np.isnan(dataframe['pred_energy']) == False
                    
            E_truth = np.asarray(dataframe[has_truth]['truthHitAssignedEnergies'], dtype=np.double)
            eta_truth = dataframe[has_truth]['truthHitAssignedZ']
            phi_truth = np.arctan2(dataframe[has_truth]['truthHitAssignedY'], dataframe[has_truth]['truthHitAssignedX'])
            pt_truth = E_truth / np.cosh(eta_truth)
            px_truth = np.asarray(pt_truth * np.cos(phi_truth), dtype=np.double)
            py_truth = np.asarray(pt_truth * np.sin(phi_truth), dtype=np.double)
            pz_truth = np.asarray(pt_truth * np.sinh(eta_truth), dtype=np.double)       
            
            particles_truth = np.array([fj.PseudoJet(px_truth[i], py_truth[i], pz_truth[i], E_truth[i]) for i in range(len(dataframe[has_truth]['truthHitAssignedX']))])

            E_pred = np.asarray(dataframe[has_pred]['pred_energy'], dtype=np.double)
            eta_pred = dataframe[has_pred]['pred_pos'].apply(lambda x: x[2])
            phi_pred = dataframe[has_pred]['pred_pos'].apply(lambda x: np.arctan2(x[1], x[0]))
            pt_pred = E_pred / np.cosh(eta_pred)
            px_pred = np.asarray(pt_pred * np.cos(phi_pred), dtype=np.double)
            py_pred = np.asarray(pt_pred * np.sin(phi_pred), dtype=np.double)
            pz_pred = np.asarray(pt_pred * np.sinh(eta_pred), dtype=np.double)
            particles_pred = np.array([fj.PseudoJet(px_pred[i], py_pred[i], pz_pred[i], E_pred[i]) for i in range(len(dataframe[has_pred]['pred_pos']))])
            
            # Choose jet clustering algorithm and parameters
            R = 0.4 # Jet radius parameter
            jet_def = fj.JetDefinition(fj.antikt_algorithm, R)

            # Cluster jets
            cluster_sequence_truth = fj.ClusterSequence(particles_truth.tolist(), jet_def)
            cluster_sequence_pred = fj.ClusterSequence(particles_pred.tolist(), jet_def)
            jets_truth = [jet for jet in cluster_sequence_truth.inclusive_jets() if len(jet.constituents()) >= 2]
            jets_pred = [jet for jet in cluster_sequence_pred.inclusive_jets() if len(jet.constituents()) >= 2]
            
            jet_truth_data = {
                'true_pt': [jet.pt() for jet in jets_truth],
                'true_eta': [jet.eta() for jet in jets_truth],
                'true_phi': [jet.phi() for jet in jets_truth],
                'true_mass': [jet.m() for jet in jets_truth],
                'true_n_constituents': [len(jet.constituents()) for jet in jets_truth]
            }
            jet_pred_dta = {
                'pred_pt': [jet.pt() for jet in jets_pred],
                'pred_eta': [jet.eta() for jet in jets_pred],
                'pred_phi': [jet.phi() for jet in jets_pred],
                'pred_mass': [jet.m() for jet in jets_pred],
                'pred_n_constituents': [len(jet.constituents()) for jet in jets_pred]
            }
            jet_truth_df = pd.DataFrame(jet_truth_data)
            jet_pred_df = pd.DataFrame(jet_pred_dta)
            
            jet_truth_df['event_id'] = event_id
            jet_pred_df['event_id'] = event_id
            
            # Match jets
            n = max(len(jet_truth_df), len(jet_pred_df))
            C = np.zeros((n, n))
            
            for a in range(len(jet_truth_df)):
                for b in range(len(jet_pred_df)):
                    delta_eta = jet_truth_df['true_eta'].iloc[a] - jet_pred_df['pred_eta'].iloc[b]
                    delta_phi = jet_truth_df['true_phi'].iloc[a] - jet_pred_df['pred_phi'].iloc[b]
                    C[a, b] = 1/np.sqrt(delta_eta**2 + delta_phi**2)
            row_id, col_id = linear_sum_assignment(C, maximize=True)
            
            matched_jets = []
            for a, b in zip(row_id, col_id):
                if C[a, b] > 0:
                    matched_jets.append({
                        'matched': True,
                        'true_pt': jet_truth_df['true_pt'].iloc[a],
                        'true_eta': jet_truth_df['true_eta'].iloc[a],
                        'true_phi': jet_truth_df['true_phi'].iloc[a],
                        'true_mass': jet_truth_df['true_mass'].iloc[a],
                        'true_n_constituents': jet_truth_df['true_n_constituents'].iloc[a],
                        'pred_pt': jet_pred_df['pred_pt'].iloc[b],
                        'pred_eta': jet_pred_df['pred_eta'].iloc[b],
                        'pred_phi': jet_pred_df['pred_phi'].iloc[b],
                        'pred_mass': jet_pred_df['pred_mass'].iloc[b],
                        'pred_n_constituents': jet_pred_df['pred_n_constituents'].iloc[b]
                    })
                else:
                    if C[a, :].max() == 0 and a<len(jet_truth_df):
                        matched_jets.append({
                            'matched': False,
                            'true_pt': jet_truth_df['true_pt'].iloc[a],
                            'true_eta': jet_truth_df['true_eta'].iloc[a],
                            'true_phi': jet_truth_df['true_phi'].iloc[a],
                            'true_mass': jet_truth_df['true_mass'].iloc[a],
                            'true_n_constituents': jet_truth_df['true_n_constituents'].iloc[a],
                            'pred_pt': np.nan,
                            'pred_eta': np.nan,
                            'pred_phi': np.nan,
                            'pred_mass': np.nan,
                            'pred_n_constituents': np.nan
                        })
                    elif C[:, b].max() == 0 and b<len(jet_pred_df):
                        matched_jets.append({
                            'matched': False,
                            'true_pt': np.nan,
                            'true_eta': np.nan,
                            'true_phi': np.nan,
                            'true_mass': np.nan,
                            'true_n_constituents': np.nan,
                            'pred_pt': jet_pred_df['pred_pt'].iloc[b],
                            'pred_eta': jet_pred_df['pred_eta'].iloc[b],
                            'pred_phi': jet_pred_df['pred_phi'].iloc[b],
                            'pred_mass': jet_pred_df['pred_mass'].iloc[b],
                            'pred_n_constituents': jet_pred_df['pred_n_constituents'].iloc[b]
                        })
            matched_jets_df = pd.DataFrame(matched_jets)
            matched_jets_df['event_id'] = event_id
            
            # Append the matched jets dataframe to the existing jets dataframe
            jets_dataframe = pd.concat([jets_dataframe, matched_jets_df])
            
            event_id += 1

    ###############################################################################################
    ### Write to file #############################################################################
    ###############################################################################################
    if len(analysisoutpath) > 0:
                
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
            'jets_dataframe' : jets_dataframe,
            'scalar_variables' : scalar_variables,
            'alpha_ids'        : alpha_ids,
            'matched': matched,
            'features': features,
            'truth': truth,
            'prediction': prediction
        }

        with gzip.open(analysisoutpath, 'wb') as output_file:
            print("Writing dataframes to pickled file",analysisoutpath)
            pickle.dump(analysis_data, output_file)
            
    ###############################################################################################
    ### Make Plots ################################################################################
    ###############################################################################################
  
    if(len(outputpath) > 0):
        print('Starting to plot', outputpath)
        plot_everything(showers_dataframe, prediction, truth, features, jets_dataframe, outputpath, datasetname)
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
    parser.add_argument('-m', help='Matching mode', default='cocoa')
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
    parser.add_argument('--gluon', action='store_true', help='Writing Gluon dataset')

    args = parser.parse_args()
    
    if args.gluon:
        datasetname='Gluon Jet'
    else:
        datasetname='Quark Jet'

    analyse(preddir=args.preddir,
            outputpath=args.p,
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
            shower0=args.shower0,
            datasetname=datasetname,
            )
    
