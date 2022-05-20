#!/usr/bin/env python3
import os
import gzip
import pickle

import mgzip

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random

from OCHits2Showers import OCHits2Showers
from ShowersMatcher import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter

preddir = '/eos/user/j/jfli/predictions/lr5e-4'
beta_threshold=0.01
distance_threshold=0.05
is_soft=not True
local_distance_scaling=True
op=not True
nfiles=-1

hits2showers = OCHits2Showers(beta_threshold, distance_threshold, is_soft, local_distance_scaling, op=op)

files_to_be_tested = [os.path.join(preddir, x) for x in os.listdir(preddir) if x.endswith('.bin.gz')]
if nfiles!=-1:
	files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

showers_dataframe = pd.DataFrame()
event_id = 0

fig = plt.figure()
ax = plt.axes(projection='3d')

n_events = 1 # random.shuffle(
for i, file in enumerate(files_to_be_tested):
	print("Analysing file", i, file)
	with gzip.open(file, 'rb') as f:
		file_data = pickle.load(f)
		for j, endcap_data in enumerate(file_data):
			print("Analysing endcap",j)
			stopwatch = time.time()
			features_dict, truth_dict, predictions_dict = endcap_data
			processed_pred_dict, pred_shower_alpha_idx = hits2showers.call(features_dict, predictions_dict)
			print('took',time.time()-stopwatch,'s for inference clustering')

			# print(processed_pred_dict.keys())
			# print(features_dict.keys())
			# print(truth_dict.keys())
			# print(predictions_dict.keys())

			check = processed_pred_dict['pred_sid']
			# check = truth_dict['truthHitAssignementIdx']
			print("len unique: ", len(np.unique(check)))
			count = 0
			for k, val in enumerate(np.unique(check)):
				indices = np.where(check==val)
				if val==-1 or len(indices[0]) != 4:
					continue
				print(len(indices[0]))
				ax.scatter3D(features_dict['recHitX'][indices[0]], features_dict['recHitY'][indices[0]], features_dict['recHitZ'][indices[0]])
				count += 1
				if count == 5:
					break
				# ax.scatter3D(predictions_dict['recHitX'][indices[0]], predictions_dict['recHitY'][indices[0]], predictions_dict['recHitZ'][indices[0]])
			
			# xcoords = [asdf[0] for asdf in predictions_dict['pred_ccoords']]
			# ycoords = [asdf[1] for asdf in predictions_dict['pred_ccoords']]
			# zcoords = [asdf[2] for asdf in predictions_dict['pred_ccoords']]

			# # print((np.array(truth_dict['truthHitAssignementIdx'].squeeze())))
			# print(predictions_dict['pred_ccoords'])
			# # with open('/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/3Dcoords.pickle', 'wb') as pkl:
			# # 	pickle.dump(predictions_dict['pred_ccoords'], pkl, protocol=pickle.HIGHEST_PROTOCOL)

			# # quit()

			# ax.scatter3D(xcoords, ycoords, zcoords)
			# fig = px.scatter_3d(x=xcoords, y=ycoords, z=zcoords,# )
			# 					color=(truth_dict['truthHitAssignementIdx'].squeeze()))
			# 					# size=(np.array(predictions_dict['pred_beta']).squeeze()))


			# fig.write_html('/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/3d_pred.html')
			# quit()
			# print(len(np.where(processed_pred_dict['pred_sid']==0)[0]))
	if i==n_events-1: break

plt.savefig('/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/3d_pred1.png')

# def analyse(preddir, pdfpath, beta_threshold, distance_threshold, iou_threshold, matching_mode, analysisoutpath, nfiles,
#             local_distance_scaling, is_soft, op, de_e_cut, angle_cut, kill_pu=True):
#     hits2showers = OCHits2Showers(beta_threshold, distance_threshold, is_soft, local_distance_scaling, op=op)
#     # showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

#     files_to_be_tested = [os.path.join(preddir, x) for x in os.listdir(preddir) if x.endswith('.bin.gz')]
#     if nfiles!=-1:
#         files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

#     showers_dataframe = pd.DataFrame()
#     event_id = 0

#     for i, file in enumerate(files_to_be_tested):
#         print("Analysing file", i, file)
#         with gzip.open(file, 'rb') as f:
#             file_data = pickle.load(f)
#             for j, endcap_data in enumerate(file_data):
#                 print("Analysing endcap",j)
#                 stopwatch = time.time()
#                 features_dict, truth_dict, predictions_dict = endcap_data
#                 processed_pred_dict, pred_shower_alpha_idx = hits2showers.call(features_dict, predictions_dict)
#                 print('took',time.time()-stopwatch,'s for inference clustering')
#                 continue
#                 stopwatch = time.time()
#                 showers_matcher.set_inputs(
#                     features_dict=features_dict,
#                     truth_dict=truth_dict,
#                     predictions_dict=processed_pred_dict,
#                     pred_alpha_idx=pred_shower_alpha_idx
#                 )
#                 showers_matcher.process()
#                 print('took',time.time()-stopwatch,'s to match')
#                 stopwatch = time.time()
#                 dataframe = showers_matcher.get_result_as_dataframe()
#                 print('took',time.time()-stopwatch,'s to make data frame')
#                 dataframe['event_id'] = event_id
#                 event_id += 1
#                 if kill_pu:
#                     from globals import pu
#                     if len(dataframe[dataframe['truthHitAssignementIdx']>=pu.t_idx_offset]):
#                         print('\nWARNING REMOVING PU TRUTH MATCHED SHOWERS, HACK.\n')
#                         dataframe = dataframe[dataframe['truthHitAssignementIdx']<pu.t_idx_offset]
#                 showers_dataframe = pd.concat((showers_dataframe, dataframe))

#     # This is only to write to pdf files
#     scalar_variables = {
#         'beta_threshold': str(beta_threshold),
#         'distance_threshold': str(distance_threshold),
#         'iou_threshold': str(iou_threshold),
#         'matching_mode': str(matching_mode),
#         'is_soft': str(is_soft),
#         'de_e_cut': str(de_e_cut),
#         'angle_cut': str(angle_cut),
#     }

#     if len(analysisoutpath) > 0:
#         analysis_data = {
#             'showers_dataframe' : showers_dataframe,
#             'events_dataframe' : None,
#             'scalar_variables' : scalar_variables,
#         }
#         with gzip.open(analysisoutpath, 'wb') as f:
#             print("Writing dataframes to pickled file",analysisoutpath)
#             pickle.dump(analysis_data,f)

#     if len(pdfpath)>0:
#         plotter = HGCalAnalysisPlotter()
#         plotter.set_data(showers_dataframe, None, '', pdfpath, scalar_variables=scalar_variables)
#         plotter.process()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         'Analyse predictions from object condensation and plot relevant results')
#     parser.add_argument('preddir',
#                         help='Directory with .bin.gz files or a txt file with full paths of the bin gz files from the prediction.')
#     parser.add_argument('-p',
#                         help='Output directory for the final analysis pdf file (otherwise, it won\'t be produced)',
#                         default='')
#     parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
#     parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
#     parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
#     parser.add_argument('-m', help='Matching mode', default='iou_max')
#     parser.add_argument('--analysisoutpath', help='Will dump analysis data to a file to remake plots without re-running everything.',
#                         default='')
#     parser.add_argument('--nfiles', help='Maximum number of files. -1 for everything in the preddir',
#                         default=-1)
#     parser.add_argument('--no_local_distance_scaling', help='With local distance scaling', action='store_true')
#     parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
#     parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
#     parser.add_argument('--no_op', help='Use condensate op', action='store_true')
#     parser.add_argument('--no_soft', help='Use condensate op', action='store_true')

#     args = parser.parse_args()

#     analyse(preddir=args.preddir, pdfpath=args.p, beta_threshold=float(args.b), distance_threshold=float(args.d),
#             iou_threshold=float(args.i), matching_mode=args.m, analysisoutpath=args.analysisoutpath,
#             nfiles=int(args.nfiles), local_distance_scaling=not args.no_local_distance_scaling,
#             is_soft=not args.no_soft, op=not args.no_op, de_e_cut=float(args.de_e_cut), angle_cut=float(args.angle_cut))


