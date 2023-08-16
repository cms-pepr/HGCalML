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

from OCHits2Showers import OCHits2ShowersLayer
from OCHits2Showers import process_endcap2, OCGatherEnergyCorrFac2
from ShowersMatcher2 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
import extra_plots as ep
from visualize_event import dataframe_to_plot, matched_plot


def analyse(preddir, pdfpath, beta_threshold, distance_threshold, iou_threshold,
        matching_mode, analysisoutpath, nfiles, nevents,
        local_distance_scaling, is_soft, de_e_cut, angle_cut,
        energy_mode='hits'):
    """
    Analyse model predictions
    This includes:
    * building showers
    * matching them to truth showers
    * calculating efficiencies
    * calculating energy resolution
    * calculating pid prediction accuracy
    * plotting all of the above
    """

    hits2showers = OCHits2ShowersLayer(
        beta_threshold,
        distance_threshold,
        local_distance_scaling)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    energy_gatherer = OCGatherEnergyCorrFac2()

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
    processed = []
    alpha_ids = []
    noise_masks = []
    matched = []
    event_id = 0
    n_hits_orig = [] # Non noise hits
    n_hits_filtered = [] # Non noise hits after filter
    n_noise_orig = [] # Noise hits
    n_noise_filtered = [] # Noise hits after filter
    e_hits_orig = [] # Non noise hits energy
    e_hits_filtered = [] # Non noise hits energy after filter
    e_noise_orig = [] # Noise hits energy
    e_noise_filtered = [] # Noise hits energy after filter

    ###############################################################################################
    ### Loop over all events ######################################################################
    ###############################################################################################

    for i, file in enumerate(files_to_be_tested):
        print(f"Analysing file {i}/{len(files_to_be_tested)}")
        with mgzip.open(file, 'rb') as analysis_file:
            file_data = pickle.load(analysis_file)
        for j, endcap_data in enumerate(file_data):
            print(f"Event {j} out of {len(file_data)}")
            if (nevents != -1) and (j > nevents):
                continue
            print(f"Analysing endcap {j}/{len(file_data)}")
            features_dict, truth_dict, predictions_dict = endcap_data
            features.append(features_dict)
            prediction.append(predictions_dict)
            truth.append(truth_dict)

            try:
                noise_mask = predictions_dict['no_noise_sel']
                includes_mask = True
            except:
                includes_mask = False
                noise_mask = np.ones_like(features_dict['recHitX']).astype(int)
            noise_masks.append(noise_mask)
            truth_df = ep.dictlist_to_dataframe([truth_dict], add_event_id=False)
            features_df = ep.dictlist_to_dataframe([features_dict], add_event_id=False)
            filtered_features = ep.filter_features_dict(features_dict, noise_mask)
            filtered_truth = ep.filter_truth_dict(truth_dict, noise_mask)
            filtered_truth_df = ep.dictlist_to_dataframe([filtered_truth], add_event_id=False)
            filtered_features_df = ep.dictlist_to_dataframe([filtered_features], add_event_id=False)
            n_hits_orig.append(np.sum(truth_df.truthHitAssignementIdx != -1))
            n_hits_filtered.append(np.sum(filtered_truth_df.truthHitAssignementIdx != -1))
            n_noise_orig.append(np.sum(truth_df.truthHitAssignementIdx == -1))
            n_noise_filtered.append(np.sum(filtered_truth_df.truthHitAssignementIdx == -1))
            e_hits_orig.append(np.sum(
                features_df[truth_df.truthHitAssignementIdx != -1].recHitEnergy))
            e_hits_filtered.append(np.sum(
                filtered_features_df[filtered_truth_df.truthHitAssignementIdx != -1].recHitEnergy))
            e_noise_orig.append(np.sum(
                features_df[truth_df.truthHitAssignementIdx == -1].recHitEnergy))
            e_noise_filtered.append(np.sum(
                filtered_features_df[filtered_truth_df.truthHitAssignementIdx == -1].recHitEnergy))

            processed_pred_dict, pred_shower_alpha_idx = process_endcap2(
                    hits2showers,
                    energy_gatherer,
                    filtered_features,
                    predictions_dict,
                    energy_mode=energy_mode)

            alpha_ids.append(pred_shower_alpha_idx)
            processed.append(processed_pred_dict)
            showers_matcher.set_inputs(
                features_dict=filtered_features,
                truth_dict=filtered_truth,
                predictions_dict=processed_pred_dict,
                pred_alpha_idx=pred_shower_alpha_idx
            )
            showers_matcher.process()
            dataframe = showers_matcher.get_result_as_dataframe()
            matched_truth_sid, matched_pred_sid = showers_matcher.get_matched_hit_sids()
            matched.append((matched_truth_sid, matched_pred_sid))

            dataframe['event_id'] = event_id
            showers_dataframe = pd.concat((showers_dataframe, dataframe))
            processed_dataframe = ep.dictlist_to_dataframe(processed[-1:])

            eventsdir = os.path.join('.', 'events')
            if not os.path.isdir(eventsdir):
                os.mkdir(eventsdir)
            if event_id < 10:
                # make 3d plot of the event and save it
                tmp_feat = ep.dictlist_to_dataframe([filtered_features], add_event_id=False)
                tmp_truth = ep.dictlist_to_dataframe([filtered_truth], add_event_id=False)
                # tmp_feat.drop(['event_id'], inplace=True)
                # tmp_truth.drop(['event_id'], inplace=True)
                s_feat = tmp_feat.shape
                s_processed = processed_dataframe.shape
                if s_processed[0] != s_feat[0]:
                    pdb.set_trace()
                # print("tmp_feat: ", tmp_feat.shape, "\ntmp_truth: ", tmp_truth.shape)
                # print("processed: ", processed_dataframe.shape)
                tmp_predbeta = pd.DataFrame(predictions_dict['pred_beta'], columns=['pred_beta'])
                full_df = pd.concat((tmp_feat, tmp_truth, processed_dataframe, tmp_predbeta), axis=1)
                fig_truth = dataframe_to_plot(full_df, truth=True)
                fig_pred = dataframe_to_plot(full_df, truth=False)
                fig_truth.write_html(os.path.join('.', 'events', f'event_{event_id}_truth.html'))
                fig_pred.write_html(os.path.join('.', 'events', f'event_{event_id}_pred.html'))
                fig_cluster_pca = dataframe_to_plot(full_df, truth=False, clusterspace='pca')
                fig_cluster_first = dataframe_to_plot(full_df, truth=False, clusterspace=(0,1,2))
                fig_cluster_pca.write_html(os.path.join('.', 'events', f'event_{event_id}_cluster_pca.html'))
                fig_cluster_first.write_html(os.path.join('.', 'events', f'event_{event_id}_cluster_first.html'))
                fig_cluster_pca_truth = dataframe_to_plot(full_df, truth=True, clusterspace='pca')
                fig_cluster_first_truth = dataframe_to_plot(full_df, truth=True, clusterspace=(0,1,2))
                fig_cluster_pca_truth.write_html(os.path.join('.', 'events', f'event_{event_id}_cluster_pca_truth.html'))
                fig_cluster_first_truth.write_html(os.path.join('.', 'events', f'event_{event_id}_cluster_first_truth.html'))
                fig_matched = matched_plot(filtered_truth, filtered_features, processed_dataframe, dataframe)
                fig_matched.write_html(os.path.join('.', 'events', f'event_{event_id}_matched.html'))

            event_id += 1

    noise_df = pd.DataFrame({
        'n_hits_orig': n_hits_orig,
        'n_hits_filtered': n_hits_filtered,
        'n_noise_orig': n_noise_orig,
        'n_noise_filtered': n_noise_filtered,
        'e_hits_orig': e_hits_orig,
        'e_hits_filtered': e_hits_filtered,
        'e_noise_orig': e_noise_orig,
        'e_noise_filtered': e_noise_filtered,
    })

    ###############################################################################################
    ### New plotting stuff ########################################################################
    ###############################################################################################

    ### Noise Filter Performance ##################################################################
    try:
        fig = ep.noise_performance(noise_df)
        fig.savefig(os.path.join('.', 'noise_performance.jpg'))
    except:
        print("Noise overview failed")

    ### Prediction overview #######################################################################
    try:
        fig = ep.prediction_overview(prediction)
        fig.savefig(os.path.join('.', 'prediction_overview.jpg'))
    except:
        print("Overview failed")

    ### Classification ############################################################################
    try:
        fig = ep.classification_plot(showers_dataframe)
        fig.savefig(os.path.join('.', 'classification_plot.jpg'))
    except:
        print("Classification failed")

    ### Tracks versus hits ########################################################################
    try:
        fig = ep.tracks_vs_hits(showers_dataframe)
        fig.savefig(os.path.join('.', 'median_ratios.jpg'))
    except:
        print("Tracks-vs-hits failed")

    ### Efficiency plots ##########################################################################
    try:
        fig_eff = ep.efficiency_plot(showers_dataframe)
        fig_eff.savefig(os.path.join('.', 'efficiency.jpg'))
    except:
        print("Efficiency failed")

    ### Resolution plots ##########################################################################
    try:
        fig_res = ep.energy_resolution(showers_dataframe)
        fig_res.savefig(os.path.join('.', 'energy_resolution.jpg'))
    except:
        print("Resolution plot failed")

    ### Energy uncertainty plot ###################################################################
    try:
        fig_unc = ep.within_uncertainty(showers_dataframe)
        fig_unc.savefig(os.path.join('.', 'within_uncertainty.jpg'))
    except:
        print("Uncertainty plot failed")


    ### low-high difference plot ##################################################################

    try:
        fig_low_high = ep.plot_high_low_difference(prediction)
        fig_low_high.savefig(os.path.join('.', 'low_high_difference.jpg'))
    except:
        print("low-high difference plot failed")

    # This is only to write to pdf files
    scalar_variables = {
        'beta_threshold': str(beta_threshold),
        'distance_threshold': str(distance_threshold),
        'iou_threshold': str(iou_threshold),
        'matching_mode': str(matching_mode),
        'is_soft': str(is_soft),
        'de_e_cut': str(de_e_cut),
        'angle_cut': str(angle_cut),
    }

    if len(analysisoutpath) > 0:
        analysis_data = {
            'showers_dataframe' : showers_dataframe,
            'events_dataframe' : None,
            'scalar_variables' : scalar_variables,
            'alpha_ids'        : alpha_ids,
            'noise_masks': noise_masks,
            'matched': matched,
        }
        if not args.slim:
            analysis_data['processed_dataframe'] = ep.dictlist_to_dataframe(processed)
            analysis_data['features'] = features
            analysis_data['truth'] = truth
            analysis_data['prediction'] = prediction
        with gzip.open(analysisoutpath, 'wb') as output_file:
            print("Writing dataframes to pickled file",analysisoutpath)
            pickle.dump(analysis_data, output_file)

    if len(pdfpath)>0:
        plotter = HGCalAnalysisPlotter()
        plotter.set_data(showers_dataframe, None, '', pdfpath, scalar_variables=scalar_variables)
        plotter.process()

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
    parser.add_argument('--no_local_distance_scaling', help='With local distance scaling',
        action='store_true')
    parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
    parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
    parser.add_argument('--no_soft', help='Use condensate op', action='store_true')
    parser.add_argument('--nevents', help='Maximum number of events (per file)', default=-1)
    parser.add_argument('--emode', help='Mode how energy is calculated', default='hits')
    parser.add_argument('--slim',
        help="Produce only a small analysis.bin.gz file. \
            Only applicable if --analysisoutpath is set",
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
            is_soft=not args.no_soft,
            de_e_cut=float(args.de_e_cut),
            angle_cut=float(args.angle_cut),
            nevents=int(args.nevents),
            energy_mode=str(args.emode),
            )
