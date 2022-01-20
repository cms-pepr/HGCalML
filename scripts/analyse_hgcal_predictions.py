import os
import gzip
import pickle

import mgzip

import argparse

import pandas as pd

from OCHits2Showers import OCHits2Showers
from ShowersMatcher import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter

def analyse(preddir, pdfpath, beta_threshold, distance_threshold, iou_threshold, matching_mode, analysisoutpath, nfiles,
            local_distance_scaling, is_soft, op, de_e_cut, angle_cut):
    hits2showers = OCHits2Showers(beta_threshold, distance_threshold, is_soft, local_distance_scaling, op=op)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    files_to_be_tested = [os.path.join(preddir, x) for x in os.listdir(preddir)]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    event_id = 0

    for i, file in enumerate(files_to_be_tested):
        print("Analysing file", i)
        with mgzip.open(file, 'rb') as f:
            file_data = pickle.load(f)
            for j, endcap_data in enumerate(file_data):
                print("Analysing endcap",j)
                features_dict, truth_dict, predictions_dict = endcap_data
                processed_pred_dict, pred_shower_alpha_idx = hits2showers.call(features_dict, predictions_dict)
                showers_matcher.set_inputs(
                    features_dict=features_dict,
                    truth_dict=truth_dict,
                    predictions_dict=processed_pred_dict,
                    pred_alpha_idx=pred_shower_alpha_idx
                )
                showers_matcher.process()

                dataframe = showers_matcher.get_result_as_dataframe()
                dataframe['event_id'] = event_id
                event_id += 1
                showers_dataframe = pd.concat((showers_dataframe, dataframe))

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
        }
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
                        help='Directory with .bin.gz files or a txt file with full paths of the bin gz files from the prediction.')
    parser.add_argument('-p',
                        help='Output directory for the final analysis pdf file (otherwise, it won\'t be produced)',
                        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-m', help='Matching mode', default='iou_max')
    parser.add_argument('--analysisoutpath', help='Will dump analysis data to a file to remake plots without re-running everything.',
                        default='')
    parser.add_argument('--nfiles', help='Maximum number of files. -1 for everything in the preddir',
                        default=-1)
    parser.add_argument('--no_local_distance_scaling', help='With local distance scaling', action='store_true')
    parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
    parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
    parser.add_argument('--no_op', help='Use condensate op', action='store_true')
    parser.add_argument('--no_soft', help='Use condensate op', action='store_true')

    args = parser.parse_args()

    analyse(preddir=args.preddir, pdfpath=args.p, beta_threshold=float(args.b), distance_threshold=float(args.d),
            iou_threshold=float(args.i), matching_mode=args.m, analysisoutpath=args.analysisoutpath,
            nfiles=int(args.nfiles), local_distance_scaling=not args.no_local_distance_scaling,
            is_soft=not args.no_soft, op=not args.no_op, de_e_cut=float(args.de_e_cut), angle_cut=float(args.angle_cut))


