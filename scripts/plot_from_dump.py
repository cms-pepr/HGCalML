#!/usr/bin/env python3
from experiment_database_manager import ExperimentDatabaseManager
import gzip

import numpy as np
import pickle
import sys

from matplotlib.backends.backend_pdf import PdfPages

import sql_credentials
from matching_and_analysis import convert_dataset_dict_elements_to_numpy
from hplots.response_fo_truth_energy_plot import ResponseFoTruthEnergy
from hplots.general_2d_plot_extensions import ResponseFoIouPlot, EfficiencyFoTruthEnergyPlot, FakeRateFoPredEnergyPlot
from hplots.general_2d_plot_extensions import ResolutionFoTruthEnergyPlot, ResponseFoTruthEnergyPlot
from hplots.general_2d_plot import General2dBinningPlot

with gzip.open(sys.argv[1], 'rb') as f:
    dataset_analysis_dict = pickle.load(f)


pdfpath = sys.argv[2]
# database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
# database_manager.set_experiment('alpha_experiment_june_pca_double_cords_2', '/mnt/ceph/users/sqasim/trainings/group_june/alpha_experiment_june_pca_double_cords_2')
dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)

pdf = PdfPages(pdfpath)

tags = dict()
tags['beta_threshold'] = dataset_analysis_dict['beta_threshold']
tags['distance_threshold'] = dataset_analysis_dict['distance_threshold']
tags['iou_threshold'] = dataset_analysis_dict['iou_threshold']
tags['soft'] = dataset_analysis_dict['soft']


efficiency_plot = EfficiencyFoTruthEnergyPlot()
x_values = dataset_analysis_dict['truth_shower_energy']
y_values = dataset_analysis_dict['truth_shower_found_or_not']
efficiency_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
efficiency_plot.draw()
pdf.savefig()
# efficiency_plot.write_to_database(database_manager, 'eff_plot_alpha')



response_plot = ResponseFoTruthEnergyPlot()
filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']!=-1
# filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']*0 == 0
x_values = dataset_analysis_dict['truth_shower_energy'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter] / dataset_analysis_dict['truth_shower_energy'][filter]
response_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
response_plot.draw()
pdf.savefig()

resolution_plot = ResolutionFoTruthEnergyPlot()
filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']!=-1
x_values = dataset_analysis_dict['truth_shower_energy'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter] / dataset_analysis_dict['truth_shower_energy'][filter]
resolution_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
resolution_plot.draw()
pdf.savefig()




response_plot = ResponseFoTruthEnergyPlot(y_label='Response computed from rechit sum')
filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']!=-1
# filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']*0 == 0
x_values = dataset_analysis_dict['truth_shower_energy'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_sum'][filter] / dataset_analysis_dict['truth_shower_energy'][filter]
response_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
response_plot.draw()
pdf.savefig()

resolution_plot = ResolutionFoTruthEnergyPlot(y_label='Resolution computed from rechit sum')
filter = dataset_analysis_dict['truth_shower_matched_energy_regressed']!=-1
x_values = dataset_analysis_dict['truth_shower_energy'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_sum'][filter] / dataset_analysis_dict['truth_shower_energy'][filter]
resolution_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
resolution_plot.draw()
pdf.savefig()


# response_plot = ResponseFoTruthEnergy()
# response_plot.add_model_predicted_values(dataset_analysis_dict['truth_shower_energy'],
#                         dataset_analysis_dict['truth_shower_matched_energy_regressed'], tags)
# # response_plot.write_to_database(database_manager, 'response_plot_alpha')
#
# response_plot.draw()
# pdf.savefig()

fake_rate_plot = FakeRateFoPredEnergyPlot()
x_values = dataset_analysis_dict['pred_shower_regressed_energy']
y_values = dataset_analysis_dict['pred_shower_matched_energy']==-1
fake_rate_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
fake_rate_plot.draw()
pdf.savefig()



response_iou_plot = ResponseFoIouPlot()
filter = dataset_analysis_dict['truth_shower_matched_iou_pred']>=0
# dataset_analysis_dict['truth_shower_energy']
# dataset_analysis_dict['truth_shower_matched_energy_sum']
# dataset_analysis_dict['truth_shower_matched_energy_regressed']
print(filter.shape, filter.dtype, dataset_analysis_dict['truth_shower_matched_energy_sum'].shape)
x_values = dataset_analysis_dict['truth_shower_matched_iou_pred'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_sum'][filter] / dataset_analysis_dict['truth_shower_energy'][filter]
response_iou_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
response_iou_plot.draw()
pdf.savefig()

response_iou_plot = ResponseFoIouPlot(y_label='Regressed energy')
filter = dataset_analysis_dict['truth_shower_matched_iou_pred']>=0
x_values = dataset_analysis_dict['truth_shower_matched_iou_pred'][filter]
y_values = dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter]
response_iou_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
response_iou_plot.draw()
pdf.savefig()

response_iou_plot = ResponseFoIouPlot(y_label='Truth energy')
filter = dataset_analysis_dict['truth_shower_matched_iou_pred']>=0
x_values = dataset_analysis_dict['truth_shower_matched_iou_pred'][filter]
y_values = dataset_analysis_dict['truth_shower_energy'][filter]
response_iou_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
response_iou_plot.draw()
pdf.savefig()

pdf.close()
#
# database_manager.close()
