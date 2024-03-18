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

from OCHits2Showers import OCHits2ShowersLayer, OCHits2ShowersLayer_HDBSCAN
from OCHits2Showers import process_endcap2, OCGatherEnergyCorrFac2, OCGatherEnergyCorrFac_new
from ShowersMatcher2 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
import extra_plots as ep
from visualize_event import dataframe_to_plot, matched_plot


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
            is_soft,
            de_e_cut,
            angle_cut,
            energy_mode='hits',
            slim=True,
            use_hdbscan=False,
            min_cluster_size=50,
            min_samples=100,
            mask_radius=None,
            extra=False,
            hdf=False,
            eta_phi_mask=False,
        ):
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

    if use_hdbscan:
        hits2showers = OCHits2ShowersLayer_HDBSCAN
    else:
        hits2showers = OCHits2ShowersLayer(
            beta_threshold,
            distance_threshold,
            local_distance_scaling)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    # energy_gatherer = OCGatherEnergyCorrFac2()
    energy_gatherer = OCGatherEnergyCorrFac_new()

    files_to_be_tested = [
        os.path.join(preddir, x)
        for x in os.listdir(preddir)
        if (x.endswith('.bin.gz') and x.startswith('pred'))]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    matched_showers = pd.DataFrame()
    features = []
    truth = []
    prediction = []
    processed = []
    alpha_ids = []
    noise_masks = []
    masks = []
    pred_masks = []
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
        print(f"Analysing file {i+1}/{len(files_to_be_tested)}")
        with mgzip.open(file, 'rb') as analysis_file:
            file_data = pickle.load(analysis_file)
        for j, endcap_data in enumerate(file_data):
            if (nevents != -1) and (j > nevents):
                continue
            # print(f"Analysing endcap {j+1}/{len(file_data)}")
            features_dict, truth_dict, predictions_dict = endcap_data
            features.append(features_dict)
            prediction.append(predictions_dict)
            truth.append(truth_dict)

            if 'no_noise_sel' in predictions_dict.keys():
                no_noise_indices = predictions_dict['no_noise_sel'] #Shape [N_filtered, 1]
            else:
                n_feat = features_dict['recHitEnergy'].shape[0]
                no_noise_indices = tf.reshape(np.arange(n_feat), (-1, 1))
            zeros = tf.zeros_like(truth_dict['truthHitAssignedEta'][:,0], dtype=tf.bool)
            ones = tf.ones_like(no_noise_indices[:,0], dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(zeros, no_noise_indices, ones)
            if eta_phi_mask:
                # if eta_phi_mask.sum() == 0: continue
                shower_mask = truth_dict['truthHitAssignementIdx'] == 0
                phi_true = truth_dict['truthHitAssignedPhi'][shower_mask][0] # Between -pi and pi
                eta_true = truth_dict['truthHitAssignedEta'][shower_mask][0] # Between -pi and pi
                delta_eta = np.abs(truth_dict['truthHitAssignedEta'] - eta_true)
                delta_phi0 = truth_dict['truthHitAssignedPhi'] - phi_true
                delta_phi1 = delta_phi0 + 2*np.pi
                delta_phi2 = delta_phi0 - 2*np.pi
                delta_phi = np.min(
                        [np.abs(delta_phi0), np.abs(delta_phi1), np.abs(delta_phi2)],
                        axis=0)
                delta_R  = np.sqrt(delta_eta**2 + delta_phi**2)
                delta_mask = delta_R < 0.5 # Shape [N_unfiltered, 1]
                # feature_mask = delta_mask
                # Noise mask is a tensor of indices
                noise_mask = tf.tensor_scatter_nd_update(zeros, no_noise_indices, ones)
                mask = np.logical_and(delta_mask[:,0], noise_mask) # Shape [N_orig,]
                pred_mask = tf.gather_nd(delta_mask[:,0], no_noise_indices)
                pred_masks.append(pred_mask)
                pred_keys = [
                        'pred_beta', 'pred_ccoords', 'pred_energy_corr_factor',
                        'pred_energy_low_quantile', 'pred_energy_high_quantile',
                        'pred_pos', 'pred_time', 'pred_id', 'pred_dist', 'rechit_energy',
                        'no_noise_sel']
                for key in pred_keys:
                    if key is 'no_noise_sel':
                        try:
                            predictions_dict[key] = predictions_dict[key][pred_mask]
                        except KeyError:
                            pass
                    else:
                        predictions_dict[key] = predictions_dict[key][pred_mask]

            print(f"Analyzing event {event_id}")

            noise_masks.append(no_noise_indices)
            masks.append(mask)
            truth_df = ep.dictlist_to_dataframe([truth_dict], add_event_id=False)
            features_df = ep.dictlist_to_dataframe([features_dict], add_event_id=False)
            filtered_features = ep.filter_features_dict(features_dict, no_noise_indices)
            filtered_truth = ep.filter_truth_dict(truth_dict, no_noise_indices)
            filtered_features = dict(features_df[np.array(mask)])
            filtered_truth = dict(truth_df[np.array(mask)])
            for key in filtered_features.keys():
                filtered_features[key] = np.array(filtered_features[key]).reshape((-1,1))
            for key in filtered_truth.keys():
                filtered_truth[key] = np.array(filtered_truth[key]).reshape((-1,1))

            # filtered_truth_df = ep.dictlist_to_dataframe([filtered_truth], add_event_id=False)
            # filtered_features_df = ep.dictlist_to_dataframe([filtered_features], add_event_id=False)
            filtered_truth_df =  truth_df[np.array(mask)]
            filtered_features_df = features_df[np.array(mask)]
            """
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
            """

            if use_hdbscan:
                shower0_mask = filtered_truth_df['truthHitAssignementIdx'] == 0
                shower0_ccoords = predictions_dict['pred_ccoords'][shower0_mask]
                n_cluster_space = shower0_ccoords.shape[1]
                shower0_betas = predictions_dict['pred_beta'][shower0_mask]
                mask_center = shower0_ccoords[np.argmax(shower0_betas)].reshape((n_cluster_space,))

            mask_center = None

            showers_matcher.process(extra=extra)
            dataframe = showers_matcher.get_result_as_dataframe()
            matched_truth_sid, matched_pred_sid = showers_matcher.get_matched_hit_sids()
            matched.append((matched_truth_sid, matched_pred_sid))

            dataframe['event_id'] = event_id
            showers_dataframe = pd.concat((showers_dataframe, dataframe))
            processed_dataframe = ep.dictlist_to_dataframe(processed[-1:])
            matched_df = dataframe.dropna()
            matched_df = matched_df[['truthHitAssignementIdx', 'pred_sid']]
            map_truthkey = dict(zip(matched_df.truthHitAssignementIdx.values, matched_df.pred_sid.values))
            map_predkey = dict(zip(matched_df.pred_sid.values, matched_df.truthHitAssignementIdx.values))
            df = pd.DataFrame()
            df['truth_sid'] = truth_dict['truthHitAssignementIdx'][df['no_noise_sel']][:,0]

            mapped_pred_sid = []
            for sid in df.pred_sid.values:
                if sid in map_predkey.keys():
                    mapped_pred_sid.append(map_predkey[sid])
                else:
                    mapped_pred_sid.append(-1)
            df['mapped_pred_sid'] = mapped_pred_sid
            df['recHitEnergy'] = filtered_features['recHitEnergy']
            matched_df.columns = ['truth_sid', 'pred_sid']

            n_pred, n_truth, n_pred_and_truth, n_pred_not_truth = [], [], [], []
            e_pred, e_truth, e_pred_and_truth, e_pred_not_truth = [], [], [], []
            for sid in matched_df.truth_sid.values:
                n_truth.append(df[df.truth_sid == sid].shape[0])
                e_truth.append(df[df.truth_sid == sid]['recHitEnergy'].sum())
                n_pred.append(df[df.mapped_pred_sid == sid].shape[0])
                e_pred.append(df[df.mapped_pred_sid == sid]['recHitEnergy'].sum())
                n_pred_and_truth.append(df[(df.truth_sid == sid) & (df.mapped_pred_sid == sid)].shape[0])
                e_pred_and_truth.append(df[(df.truth_sid == sid) & (df.mapped_pred_sid == sid)]['recHitEnergy'].sum())
                n_pred_not_truth.append(df[(df.truth_sid != sid) & (df.mapped_pred_sid == sid)].shape[0])
                e_pred_not_truth.append(df[(df.truth_sid != sid) & (df.mapped_pred_sid == sid)]['recHitEnergy'].sum())
            matched_df['n_truth'] = n_truth
            matched_df['e_truth'] = e_truth
            matched_df['n_pred'] = n_pred
            matched_df['e_pred'] = e_pred
            matched_df['n_pred_and_truth'] = n_pred_and_truth
            matched_df['e_pred_and_truth'] = e_pred_and_truth
            matched_df['n_pred_not_truth'] = n_pred_not_truth
            matched_df['e_pred_not_truth'] = e_pred_not_truth
            matched_df['event_id'] = event_id
            matched_showers = pd.concat((matched_showers, matched_df))


            eventsdir = os.path.join('.', 'events')
            if not os.path.isdir(eventsdir):
                os.mkdir(eventsdir)
            if event_id > 2: break

            # pdb.set_trace()
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
            fig_truth = dataframe_to_plot(full_df, truth=True, allgrey=True)
            fig_pred = dataframe_to_plot(full_df, truth=False)
            event_dir = os.path.join(args.picturepath, 'events')
            # print(f"Saving to {event_dir}")
            if not os.path.exists(event_dir):
                os.mkdir(event_dir)
            fig_truth.write_html(os.path.join(event_dir, f'event_{event_id}_truth.html'))
            fig_pred.write_html(os.path.join(event_dir, f'event_{event_id}_pred.html'))
            fig_cluster_pca = dataframe_to_plot(full_df, truth=False, clusterspace='pca')
            fig_cluster_first = dataframe_to_plot(full_df, truth=False, clusterspace=(0,1,2))
            fig_cluster_pca.write_html(os.path.join(event_dir, f'event_{event_id}_cluster_pca.html'))
            fig_cluster_first.write_html(os.path.join(args.picturepath, 'events', f'event_{event_id}_cluster_first.html'))
            fig_cluster_pca_truth = dataframe_to_plot(full_df, truth=True, clusterspace='pca')
            fig_cluster_first_truth = dataframe_to_plot(full_df, truth=True, clusterspace=(0,1,2))
            fig_cluster_pca_truth.write_html(os.path.join(event_dir, f'event_{event_id}_cluster_pca_truth.html'))
            fig_cluster_first_truth.write_html(os.path.join(event_dir, f'event_{event_id}_cluster_first_truth.html'))
            fig_matched = matched_plot(filtered_truth, filtered_features, processed_dataframe, dataframe)
            fig_matched.write_html(os.path.join(event_dir, f'event_{event_id}_matched.html'))
            fig_class_hit = ep.classification_hitbased(
                    filtered_truth, predictions_dict,
                    weighted=True, normalize='true')
            fig_class_hit.savefig(os.path.join(event_dir, f'event_{event_id}_classification_plot_hits.jpg'))

            event_id += 1

    """
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
    """
    noise_df = pd.DataFrame()

    ###############################################################################################
    ### New plotting stuff ########################################################################
    ###############################################################################################

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
            'matched_showers': matched_showers,
        }
        if not slim:
            analysis_data['processed_dataframe'] = ep.dictlist_to_dataframe(processed)
            analysis_data['features'] = features
            analysis_data['truth'] = truth
            analysis_data['prediction'] = prediction
        if not hdf:
            with gzip.open(analysisoutpath, 'wb') as output_file:
                print("Writing dataframes to pickled file",analysisoutpath)
                pickle.dump(analysis_data, output_file)
        else:
            showers_dataframe.to_hdf(analysisoutpath, key='showers')



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
    parser.add_argument('--picturepath',
        help='Will dump pictures in this directory, creates it if does not exist yet',
        default='.')
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
    parser.add_argument('--min_cluster_size',
                        help='parameter used for HDBSCAN (only relevant if option --hdbscan is active)',
                        default=50)
    parser.add_argument('--min_samples',
                        help='parameter used for HDBSCAN (only relevant if option --hdbscan is active)',
                        default=100)
    parser.add_argument('--hdf',
        help="Save only shower datafram with in hdf format",
        action='store_true')
    parser.add_argument('--eta_phi_mask',
        help="Mask anything that is further than R=0.5 from the shower that has to be matched",
        action='store_true')
    parser.add_argument('--hdbscan',
        help="Do not use the default clustering algorightm but use HDBSCAN instead",
        action='store_true')
    parser.add_argument('--mask_radius', help='Filter around first shower incluster space', default=None)
    parser.add_argument('--slim',
        help="Produce only a small analysis.bin.gz file. \
            Only applicable if --analysisoutpath is set",
        action='store_true')
    parser.add_argument('--extra',
        help="Calculate more information for showers",
        action='store_true')


    args = parser.parse_args()
    if not os.path.exists(args.picturepath):
        os.mkdir(args.picturepath)

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
            slim=args.slim,
            use_hdbscan=args.hdbscan,
            min_cluster_size=int(args.min_cluster_size),
            min_samples=int(args.min_samples),
            mask_radius=args.mask_radius,
            extra=args.extra,
            hdf=args.hdf,
            eta_phi_mask=args.eta_phi_mask,
            )
