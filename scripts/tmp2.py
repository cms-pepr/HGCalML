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
import random

from OCHits2Showers import OCHits2Showers
from ShowersMatcher import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter

preddir = '/eos/user/j/jfli/predictions/lr5e-4'
beta_threshold=0.01
distance_threshold=0.05
is_soft=not True
local_distance_scaling=not True
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
		print(len(file_data))

		for j, endcap_data in enumerate(file_data):

			print("Analysing endcap",j)
			stopwatch = time.time()
			features_dict, truth_dict, predictions_dict = endcap_data
			processed_pred_dict, pred_shower_alpha_idx = hits2showers.call(features_dict, predictions_dict)
			print('took',time.time()-stopwatch,'s for inference clustering')

			# # print(processed_pred_dict.keys())
			# # print(features_dict.keys())
			# # print(truth_dict.keys())
			# # print(predictions_dict.keys())
			# print(len(np.unique(processed_pred_dict['pred_sid'])))
			# for k, val in enumerate(np.unique(processed_pred_dict['pred_sid'])):
			# 	indices = np.where(processed_pred_dict['pred_sid']==val)
			# 	# print(len(indices[0]))
			# 	ax.scatter3D(features_dict['recHitX'][indices[0]], features_dict['recHitY'][indices[0]], features_dict['recHitZ'][indices[0]])
			# 	ax.scatter3D(features_dict['recHitX'][indices[0]], features_dict['recHitY'][indices[0]], features_dict['recHitZ'][indices[0]])

			xcoords = [asdf[0] for asdf in predictions_dict['pred_ccoords']]
			ycoords = [asdf[1] for asdf in predictions_dict['pred_ccoords']]
			zcoords = [asdf[2] for asdf in predictions_dict['pred_ccoords']]

			ax.scatter3D(xcoords, ycoords, zcoords)

			# print(len(np.where(processed_pred_dict['pred_sid']==0)[0]))

	if i==n_events-1: break

plt.savefig('/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/3d_pred.png')