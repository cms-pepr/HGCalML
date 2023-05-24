#!/usr/bin/env python3
import os
import gzip
import pickle
import numpy as np
import mgzip
import argparse
import time
import pandas as pd
from globals import pu

from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from OCHits2Showers import process_endcap2, OCGatherEnergyCorrFac2
from ShowersMatcher2 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter


def filter_truth_dict(truth_dict, mask):
    n_orig = truth_dict['truthHitAssignementIdx'].shape[0]
    n_filtered = mask.shape[0]
    filtered = {}
    for key, item in truth_dict.items():
        if item.shape[0] == n_orig:
            item_filtered = item[mask]
            filtered[key] = item_filtered.reshape(-1, 1)
        else:
            print(key, " untouched")
    return filtered


def filter_features_dict(features_dict, mask):
    n_orig = features_dict['recHitEnergy'].shape[0]
    n_filtered = mask.shape[0]
    filtered = {}
    for key, item in features_dict.items():
        if item.shape[0] == n_orig:
            item_filtered = item[mask]
            filtered[key] = item_filtered.reshape(-1,1)
        else:
            print(key, " untouched")
    return filtered




def analyse(preddir, pdfpath, beta_threshold, distance_threshold, iou_threshold,
        matching_mode, analysisoutpath, nfiles, nevents,
        local_distance_scaling, is_soft, de_e_cut, angle_cut,
        kill_pu=False, filter_pu=False, toydata=False, energy_mode='hits', raw=False):

    hits2showers = OCHits2ShowersLayer(
            beta_threshold, distance_threshold, local_distance_scaling)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    energy_gatherer = OCGatherEnergyCorrFac2()

    files_to_be_tested = [
        os.path.join(preddir, x)
        for x in os.listdir(preddir)
        if (x.endswith('.bin.gz') and x.startswith('pred'))]
    if toydata:
        extra_files = [
            os.path.join(preddir, x)
            for x in os.listdir(preddir)
            if ( x.endswith('.pkl') and x.startswith('pred') )]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    processed_dataframe = pd.DataFrame()
    features = []
    truth = []
    prediction = []
    alpha_ids = []
    noise_masks = []
    event_id = 0
    matched = []

    for i, file in enumerate(files_to_be_tested):
        print("Analysing file %d/%d"% (i, len(files_to_be_tested)))
        with mgzip.open(file, 'rb') as f:
            file_data = pickle.load(f)
            if toydata:
                with open(extra_files[i], 'rb') as xf:
                    xfile = pickle.load(xf)
            for j, endcap_data in enumerate(file_data):
                if (nevents != -1) and (j > nevents):
                    continue
                if toydata:
                    t_min_bias = xfile[j][0][0]
                print("Analysing endcap %d/%d" % (j, len(file_data)))
                stopwatch = time.time()
                features_dict, truth_dict, predictions_dict = endcap_data
                features.append(features_dict)
                prediction.append(predictions_dict)
                truth.append(truth_dict)
                if filter_pu and not toydata:
                    print("Filter PU only possible if t_min_bias is provided.\
                            currently this only exists for toydata")
                if filter_pu and toydata:
                    pu_filter = np.array(t_min_bias == 0).flatten()
                    for key in predictions_dict.keys():
                        try:
                            predictions_dict[key] = predictions_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                    for key in features_dict.keys():
                        try:
                            features_dict[key] = features_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                    for key in truth_dict.keys():
                        try:
                            truth_dict[key] = truth_dict[key][pu_filter]
                        except:
                            print(key, "no filter applied")
                else:
                    pu_filter = None

                noise_mask = predictions_dict['no_noise_sel']
                noise_masks.append(noise_mask)
                filtered_features = filter_features_dict(features_dict, noise_mask)
                filtered_truth = filter_truth_dict(truth_dict, noise_mask)

                processed_pred_dict, pred_shower_alpha_idx = process_endcap2(
                        hits2showers,
                        energy_gatherer,
                        filtered_features,
                        predictions_dict,
                        energy_mode=energy_mode)
                alpha_ids.append(pred_shower_alpha_idx)
                columns = list(processed_pred_dict.keys())[:-1] #Drops the row_splits
                proc = pd.DataFrame()
                for key in columns:
                    array = np.array(processed_pred_dict[key])
                    if len(array.shape) < 2:
                        continue
                    if array.shape[1] > 1:
                        for i in range(array.shape[1]):
                            proc[key+'_'+str(i)] = array[:,i]
                    else:
                        array = array.reshape(-1)
                        proc[key] = array
                proc['event_id'] = event_id


                print('took',time.time()-stopwatch,'s for inference clustering')
                stopwatch = time.time()
                showers_matcher.set_inputs(
                    features_dict=filtered_features,
                    truth_dict=filtered_truth,
                    predictions_dict=processed_pred_dict,
                    pred_alpha_idx=pred_shower_alpha_idx
                )
                showers_matcher.process()
                print('took',time.time()-stopwatch,'s to match')
                stopwatch = time.time()
                dataframe = showers_matcher.get_result_as_dataframe()
                matched_truth_sid, matched_pred_sid = showers_matcher.get_matched_hit_sids()
                matched.append((matched_truth_sid, matched_pred_sid))

                print(dataframe.head())
                print('took',time.time()-stopwatch,'s to make data frame')
                dataframe['event_id'] = event_id
                event_id += 1
                if kill_pu:
                    if len(dataframe[dataframe['truthHitAssignementIdx']>=pu.t_idx_offset]):
                        print('\nWARNING REMOVING PU TRUTH MATCHED SHOWERS, HACK.\n')
                        dataframe = dataframe[dataframe['truthHitAssignementIdx']<pu.t_idx_offset]
                showers_dataframe = pd.concat((showers_dataframe, dataframe))
                processed_dataframe = pd.concat((processed_dataframe, proc))

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
            analysis_data['processed_dataframe'] = processed_dataframe
            analysis_data['features'] = features
            analysis_data['truth'] = truth
            analysis_data['prediction'] = prediction
        with gzip.open(analysisoutpath, 'wb') as f:
            print("Writing dataframes to pickled file",analysisoutpath)
            pickle.dump(analysis_data,f)

    if len(pdfpath)>0:
        plotter = HGCalAnalysisPlotter()
        plotter.set_data(showers_dataframe, None, '', pdfpath, scalar_variables=scalar_variables)
        plotter.process()


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
    parser.add_argument('--filter_pu', help='Filter PU', action='store_true')
    parser.add_argument('--toydata', help='Use toy detector', action='store_true')
    parser.add_argument('--nevents', help='Maximum number of events (per file)', default=-1)
    parser.add_argument('--emode', help='Mode how energy is calculated', default='hits')
    parser.add_argument('--raw', help="Ignore energy correction factor", action='store_true')
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
            filter_pu=bool(args.filter_pu),
            toydata=bool(args.toydata),
            nevents=int(args.nevents),
            energy_mode=str(args.emode),
            raw=bool(args.raw),)
