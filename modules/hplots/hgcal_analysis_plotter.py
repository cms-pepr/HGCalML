import os
import shutil

from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot, ResolutionFoEnergyPlot, ResolutionFoTruthEta, \
    ResolutionFoLocalShowerEnergyFraction
from hplots.general_2d_plot_extensions import EfficiencyFakeRateFoPt, ResolutionFoPt, ResponseFoPt
from hplots.general_2d_plot_extensions import FakeRateFoPredEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoEnergyPlot
from hplots.general_2d_plot_extensions import EnergyFoundFoPredEnergyPlot
from hplots.general_2d_plot_extensions import EnergyFoundFoTruthEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoLocalShowerEnergyFractionPlot
from hplots.general_2d_plot_extensions import EfficiencyFoLocalShowerEnergyFractionPlot
from hplots.general_2d_plot_extensions import EfficiencyFoTruthEtaPlot
from hplots.general_2d_plot_extensions import FakeRateFoPredEtaPlot
from hplots.general_2d_plot_extensions import ResponseFoTruthEtaPlot
from hplots.general_2d_plot_extensions import EfficiencyFoTruthPIDPlot
from hplots.general_2d_plot_extensions import ResponseFoTruthPIDPlot


from hplots.general_hist_extensions import ResponseHisto, Multi4HistEnergy, Multi4HistPt

import numpy as np
import matplotlib.pyplot as plt
import experiment_database_reading_manager
from hplots.general_hist_plot import GeneralHistogramPlot
import matching_and_analysis
from matching_and_analysis import one_hot_encode_id

from hplots.pid_plots import ConfusionMatrixPlot, RocCurvesPlot


class HGCalAnalysisPlotter:
    def __init__(self, plots = ['settings',
                                'efficiency_fo_truth',
                                'fake_rate_fo_pred',
                                'response_fo_truth',
                                'response_fo_pred',
                                'response_sum_fo_truth',
                                'energy_resolution',
                                'energy_found_fo_truth',
                                'energy_found_fo_pred',
                                'efficiency_fo_local_shower_energy_fraction',
                                'efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy',
                                'response_fo_local_shower_energy_fraction',
                                'response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy',
                                'efficiency_fo_truth_eta',
                                'efficiency_fo_truth_eta_flat_spectrum_wrt_energy',
                                'fake_rate_fo_pred_eta',
                                'response_fo_truth_eta',
                                'response_fo_truth_eta_flat_spectrum_wrt_energy',
                                'response_fo_pred_eta',
                                'efficiency_fo_truth_pid',
                                'response_fo_truth_pid',
                                'confusion_matrix',
                                'roc_curves',
                                'resolution_fo_true_energy',
                                'resolution_fo_local_shower_fraction',
                                'resolution_fo_eta',
                                'resolution_sum_fo_true_energy',
                                'resolution_sum_fo_local_shower_fraction',
                                'resolution_sum_fo_eta',
                                'efficiency_fo_pt',
                                'fake_rate_fo_pt',
                                'response_fo_pt',
                                'resolution_fo_pt',
                                'response_histogram',
                                'response_histogram_divided',
                                'response_pt_histogram',
                                'response_pt_histogram_divided',
                                ],log_of_distributions=True):


        self.efficiency_plot = EfficiencyFoTruthEnergyPlot(histogram_log=log_of_distributions)
        self.fake_rate_plot = FakeRateFoPredEnergyPlot(histogram_log=log_of_distributions)
        self.response_plot = ResponseFoEnergyPlot(histogram_log=log_of_distributions)
        self.response_fo_pred_plot = ResponseFoEnergyPlot(x_label='Pred energy [GeV]', y_label='Response mean (pred energy/truth energy)', histogram_log=log_of_distributions)
        self.response_sum_plot = ResponseFoEnergyPlot(y_label='Response (sum/truth)', histogram_log=log_of_distributions)
        self.response_fo_local_shower_energy_fraction = ResponseFoLocalShowerEnergyFractionPlot()
        self.response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy = ResponseFoLocalShowerEnergyFractionPlot(y_label='Response (Energy spectrum flattened)')

        self.efficiency_fo_local_shower_energy_fraction = EfficiencyFoLocalShowerEnergyFractionPlot()
        self.efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy = EfficiencyFoLocalShowerEnergyFractionPlot(y_label='Reconstruction Efficiency (Energy spectrum flattened)')

        self.efficiency_fo_truth_eta_plot = EfficiencyFoTruthEtaPlot(histogram_log=log_of_distributions)

        self.efficiency_fo_truth_eta_plot_flat_spectrum_wrt_energy = EfficiencyFoTruthEtaPlot(histogram_log=log_of_distributions,
                                                                                     y_label='Reconstruction Efficiency (Energy spectrum flattened)')

        self.fake_rate_fo_pred_eta_plot = FakeRateFoPredEtaPlot(histogram_log=log_of_distributions)

        self.response_fo_truth_eta_plot = ResponseFoTruthEtaPlot(histogram_log=log_of_distributions)
        self.response_fo_truth_eta_plot_flat_spectrum_wrt_energy = ResponseFoTruthEtaPlot(histogram_log=log_of_distributions, y_label='Response (Energy spectrum flattened)')
        self.response_fo_pred_eta_plot = ResponseFoTruthEtaPlot(x_label='abs(Pred eta)', y_label='Response mean (pred energy/truth energy)', histogram_log=log_of_distributions)
        self.efficiency_fo_truth_pid_plot = EfficiencyFoTruthPIDPlot(histogram_log=log_of_distributions)
        self.response_fo_truth_pid_plot = ResponseFoTruthPIDPlot(histogram_log=log_of_distributions)
        self.energy_found_fo_truth_plot = EnergyFoundFoTruthEnergyPlot()
        self.energy_found_fo_pred_plot = EnergyFoundFoPredEnergyPlot()
        self.confusion_matrix_plot = ConfusionMatrixPlot()
        self.roc_curves = RocCurvesPlot()

        self.efficiency_fo_pt = EfficiencyFakeRateFoPt()
        self.fake_rate_fo_pt = EfficiencyFakeRateFoPt(y_label='Fake rate', title='Fake rate comparison')
        self.response_fo_pt = ResponseFoPt(x_label='pT (GeV)')
        self.resolution_fo_pt = ResolutionFoPt(x_label='pT (GeV)')

        self.response_histogam = ResponseHisto()
        self.response_histogam_divided = Multi4HistEnergy()


        self.response_pt_histogam = ResponseHisto(x_label='${p_T}_{true}/{p_T}_{pred}$')
        self.response_pt_histogam_divided = Multi4HistPt()

        self.resolution_fo_true_energy = ResolutionFoEnergyPlot()
        self.resolution_fo_true_eta = ResolutionFoTruthEta()
        self.resolution_fo_local_shower_energy_fraction = ResolutionFoLocalShowerEnergyFraction()

        self.resolution_sum_fo_true_energy = ResolutionFoEnergyPlot(y_label='Resolution (truth, dep pred)')
        self.resolution_sum_fo_true_eta = ResolutionFoTruthEta(y_label='Resolution (truth, dep pred)')
        self.resolution_sum_fo_local_shower_energy_fraction = ResolutionFoLocalShowerEnergyFraction(y_label='Resolution (truth, dep pred)')

        # TODO: for Nadya
        self.resolution_histogram_plot = GeneralHistogramPlot(bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),x_label='Resolution (to be done)', y_label='Frequency', title='Energy resolution (to be done, placeholder)')

        self.dist_thresholds = []
        self.beta_thresholds = []
        self.iou_thresholds = []
        self.matching_types = []

        self.plots = set(plots)

        self.pred_energy_matched = []
        self.truth_energy_matched = []
        self.reco_scores = []

    def _draw_numerics(self):
        text_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal',
                     'verticalalignment': 'bottom'}
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_visible(False)
        ax.axis('off')

        bs = ','.join(['%.5f'%x for x in self.beta_thresholds])
        ds = ','.join(['%.5f'%x for x in self.dist_thresholds])
        iss = ','.join(['%.5f'%x for x in self.iou_thresholds])
        matching_types = ','.join([str(x) for x in self.matching_types])

        eprecisions = self.pred_energy_matched
        erecalls = self.truth_energy_matched
        fscores = self.reco_scores


        if len(self.pred_energy_matched) == 0:
            eprecisions = self.pred_energy_matched + [-1]
        if len(self.truth_energy_matched) == 0:
            erecalls = self.truth_energy_matched + [-1]
        if len(self.reco_scores) == 0:
            fscores = self.reco_scores + [-1]
        sp = ','.join(['%.5f'%x for x in eprecisions])
        sr = ','.join(['%.5f'%x for x in erecalls])
        sf = ','.join(['%.5f'%x for x in fscores])

        s = 'Beta threshold: %s\nDist threshold: %s\niou  threshold: %s\nMatching types: %s\n' \
            "%% pred energy matched: %s\n%% truth energy matched: %s\nReco score: %s" % (bs, ds, iss, matching_types, sp, sr, sf)

        plt.text(0, 1, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                 fontdict=text_font)

    def write_data_to_database(self, database_manager, table_prefix):
        self.efficiency_plot.write_to_database(database_manager, table_prefix+'_efficiency_plot')
        self.fake_rate_plot.write_to_database(database_manager, table_prefix+'_fake_rate_plot')
        self.response_plot.write_to_database(database_manager, table_prefix+'_response_plot')
        self.response_fo_pred_plot.write_to_database(database_manager, table_prefix+'_response_fo_pred_plot')
        self.response_sum_plot.write_to_database(database_manager, table_prefix+'_response_sum_plot')
        self.resolution_histogram_plot.write_to_database(database_manager, table_prefix+'_resolution_histogram_plot')
        self.energy_found_fo_truth_plot.write_to_database(database_manager, table_prefix+'_energy_found_fo_truth_energy')
        self.energy_found_fo_pred_plot.write_to_database(database_manager, table_prefix+'_energy_found_fo_pred_energy')
        self.response_fo_local_shower_energy_fraction.write_to_database(database_manager, table_prefix+'_response_fo_local_shower_energy_fraction')
        self.efficiency_fo_local_shower_energy_fraction.write_to_database(database_manager, table_prefix+'_efficiency_fo_local_shower_energy_fraction')
        self.efficiency_fo_truth_eta_plot.write_to_database(database_manager, table_prefix+'_efficiency_fo_truth_eta')
        self.fake_rate_fo_pred_eta_plot.write_to_database(database_manager, table_prefix+'_fake_rate_fo_pred_eta')
        self.response_fo_truth_eta_plot.write_to_database(database_manager, table_prefix+'_response_fo_truth_eta')
        self.response_fo_pred_eta_plot.write_to_database(database_manager, table_prefix+'_response_fo_pred_eta')
        self.efficiency_fo_truth_pid_plot.write_to_database(database_manager, table_prefix+'_efficiency_fo_truth_pid')
        self.response_fo_truth_pid_plot.write_to_database(database_manager, table_prefix+'_response_fo_truth_pid')
        self.confusion_matrix_plot.write_to_database(database_manager, table_prefix+'_confusion_matrix')


        database_manager.flush()

    def add_data_from_database(self, database_reading_manager, table_prefix, experiment_name=None, condition=None):
        self.efficiency_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_plot', experiment_name=experiment_name, condition=condition)
        self.fake_rate_plot.read_from_database(database_reading_manager, table_prefix + '_fake_rate_plot', experiment_name=experiment_name, condition=condition)
        self.response_plot.read_from_database(database_reading_manager, table_prefix + '_response_plot', experiment_name=experiment_name, condition=condition)

        try:
            self.response_fo_pred_plot.read_from_database(database_reading_manager, table_prefix + '_response_fo_pred_plot', experiment_name=experiment_name, condition=condition)
        except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
            print("Skipping response fo pred plot, table doesn't exist")


        if 'energy_found_fo_truth' in self.plots:
            try:
                self.energy_found_fo_truth_plot.read_from_database(database_reading_manager, table_prefix + '_energy_found_fo_truth_energy', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping energy found fo truth plot, table doesn't exist")

        if 'energy_found_fo_pred' in self.plots:
            try:
                self.energy_found_fo_truth_plot.read_from_database(database_reading_manager, table_prefix + '_energy_found_fo_pred_energy', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping energy_found_fo_pred, table doesn't exist")


        if 'response_fo_local_shower_energy_fraction' in self.plots:
            try:
                self.response_fo_local_shower_energy_fraction.read_from_database(database_reading_manager, table_prefix + '_response_fo_local_shower_energy_fraction', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping response_fo_local_shower_energy_fraction, table doesn't exist")


        if 'efficiency_fo_local_shower_energy_fraction' in self.plots:
            try:
                self.efficiency_fo_local_shower_energy_fraction.read_from_database(database_reading_manager, table_prefix + 'efficiency_fo_local_shower_energy_fraction', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping efficiency_fo_local_shower_energy_fraction, table doesn't exist")


        if 'efficiency_fo_truth_eta' in self.plots:
            try:
                self.efficiency_fo_truth_eta_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_fo_truth_eta', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping efficiency_fo_truth_eta_plot plot, table doesn't exist")


        # if 'efficiency_fo_truth_eta_flat_spectrum_wrt_energy' in self.plots:
        #     try:
        #         self.efficiency_fo_truth_eta_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_fo_truth_eta', experiment_name=experiment_name, condition=condition)
        #     except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
        #         print("Skipping efficiency_fo_truth_eta_plot plot, table doesn't exist")


        if 'fake_rate_fo_pred_eta' in self.plots:
            try:
                self.fake_rate_fo_pred_eta_plot.read_from_database(database_reading_manager, table_prefix + '_fake_rate_fo_pred_eta', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping fake_rate_fo_pred_eta_plot, table doesn't exist")


        if 'response_fo_truth_eta' in self.plots:
            try:
                self.response_fo_truth_eta_plot.read_from_database(database_reading_manager, table_prefix + '_response_fo_truth_eta', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping response_fo_truth_eta_plot, table doesn't exist")

        if 'response_fo_pred_eta' in self.plots:
            try:
                self.response_fo_pred_eta_plot.read_from_database(database_reading_manager, table_prefix + '_response_fo_pred_eta', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping response_fo_pred_eta_plot, table doesn't exist")

        if 'efficiency_fo_truth_pid' in self.plots:
            try:
                self.efficiency_fo_truth_pid_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_fo_truth_pid', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping efficiency_fo_truth_pid_plot, table doesn't exist")

        if 'response_fo_truth_pid' in self.plots:
            try:
                self.response_fo_truth_pid_plot.read_from_database(database_reading_manager, table_prefix + '_response_fo_truth_pid', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping response_fo_truth_pid_plot, table doesn't exist")

        if 'confusion_matrix' in self.plots:
            try:
                self.confusion_matrix_plot.read_from_database(database_reading_manager, table_prefix + '_confusion_matrix', experiment_name=experiment_name, condition=condition)
            except experiment_database_reading_manager.ExperimentDatabaseReadingManager.TableDoesNotExistError:
                print("Skipping confusion_matrix, table doesn't exist")



        self.response_sum_plot.read_from_database(database_reading_manager, table_prefix + '_response_sum_plot', experiment_name=experiment_name, condition=condition)
        self.resolution_histogram_plot.read_from_database(database_reading_manager, table_prefix+'_resolution_histogram_plot', experiment_name=experiment_name, condition=condition)

        tags = self.efficiency_plot.get_tags()

        self.beta_thresholds += [x['beta_threshold'] for x in tags]
        self.dist_thresholds += [x['distance_threshold'] for x in tags]
        self.iou_thresholds += [x['iou_threshold'] for x in tags]
        self.soft += [x['soft'] for x in tags]

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()
        self.soft = np.unique(self.soft).tolist()


        if 'reco_score' in tags[0]:
            self.reco_scores += [x['reco_score'] for x in tags]
            self.reco_scores = np.unique(self.reco_scores).tolist()


            self.pred_energy_matched += [x['pred_energy_percentage_matched'] for x in tags]
            self.pred_energy_matched = np.unique(self.pred_energy_matched).tolist()

            self.truth_energy_matched += [x['truth_energy_percentage_matched'] for x in tags]
            self.truth_energy_matched = np.unique(self.truth_energy_matched).tolist()

    def add_data_from_analysed_graph_list(self, analysed_graphs, metadata, label='', additional_tags=dict()):
        tags = dict()
        tags['beta_threshold'] = float(metadata['beta_threshold'])
        tags['distance_threshold'] = float(metadata['distance_threshold'])
        tags['iou_threshold'] = float(metadata['iou_threshold'])
        tags['matching_type'] = str(metadata['matching_type_str'])
        tags['label'] = str(label)

        skip = {'beta_threshold', 'distance_threshold', 'iou_threshold', 'matching_type', 'label', 'matching_type_str'}

        for key, value in metadata.items():
            if key in skip:
                continue
            if type(value) is float or type(value) is int:
                if np.isfinite(value):
                    tags[key] = value
            if type(value) is str:
                if len(value) < 100:
                    tags[key] = value

        for key, value in additional_tags.items():
            tags[key] = value

        self.beta_thresholds.append(float(tags['beta_threshold']))
        self.dist_thresholds.append(float(tags['distance_threshold']))
        self.iou_thresholds.append(float(tags['iou_threshold']))
        self.matching_types.append(str(metadata['matching_type_str']))

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()


        if 'reco_score' in metadata:
            self.reco_scores.append(metadata['reco_score'])
            self.reco_scores = np.unique(self.reco_scores).tolist()

            self.pred_energy_matched.append(metadata['pred_energy_percentage_matched'])
            self.pred_energy_matched = np.unique(self.pred_energy_matched).tolist()

            self.truth_energy_matched.append(metadata['truth_energy_percentage_matched'])
            self.truth_energy_matched = np.unique(self.truth_energy_matched).tolist()




        if 'efficiency_fo_truth' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = np.not_equal(y, -1)


            self.efficiency_plot.add_raw_values(x, y, tags)


        if 'fake_rate_fo_pred' in self.plots:

            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = np.equal(y, -1)
            self.fake_rate_plot.add_raw_values(x,y, tags)



        if 'response_fo_truth' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.response_plot.add_raw_values(x[filter], y[filter] / x[filter], tags)

        if 'response_fo_pred' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.response_fo_pred_plot.add_raw_values(x[filter], x[filter] / y[filter], tags)

        if 'response_sum_fo_truth' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.response_sum_plot.add_raw_values(x[filter], y[filter] / x[filter], tags)


        # TODO: Nadya Just adding a histogram of all the truth shower energy as a placeholder
        # if 'energy_resolution' in self.plots:
        #     self.resolution_histogram_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'][filter_truth_found], tags)

        if 'energy_found_fo_truth' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y[y==-1] = 0
            self.energy_found_fo_truth_plot.add_raw_values(x, np.minimum(x,y), tags=tags)
        if 'energy_found_fo_pred' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y[y==-1] = 0
            self.energy_found_fo_pred_plot.add_raw_values(x, np.minimum(x,y), tags=tags)

        if 'response_fo_local_shower_energy_fraction' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.response_fo_local_shower_energy_fraction.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy' in self.plots:
            e, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True,
                                                                     not_found_value=-1, sum_multi=True)

            bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
            freq = []
            for i in range(len(bins) - 1):
                l = bins[i]
                h = bins[i + 1]
                filter = np.logical_and(e >= l, e < h)
                s = float(np.sum(filter)) / float((h-l))
                freq.append(s)

            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)

            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            z = np.searchsorted(bins, e) - 1
            z = np.minimum(np.maximum(z,0), len(freq)-1)

            weights = np.array([1./freq[x] for x in z])

            filter = y!=-1
            self.response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy.add_raw_values(l[filter], y[filter] / x[filter], tags, weights=weights)

        if 'efficiency_fo_local_shower_energy_fraction' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = y!=-1
            self.efficiency_fo_local_shower_energy_fraction.add_raw_values(x, y, tags)

        if 'efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy' in self.plots:
            e, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True,
                                                                     not_found_value=-1, sum_multi=True)

            bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
            freq = []
            for i in range(len(bins) - 1):
                l = bins[i]
                h = bins[i + 1]
                filter = np.logical_and(e >= l, e < h)
                s = float(np.sum(filter)) / float((h-l))
                freq.append(s)

            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            z = np.searchsorted(bins, e) - 1
            z = np.minimum(np.maximum(z,0), len(freq)-1)

            weights = np.array([1./freq[x] for x in z])

            y = y!=-1
            self.efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy.add_raw_values(x, y, tags, weights=weights)

        if 'efficiency_fo_truth_eta' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x = np.abs(x)
            #print("ZZZZZZ", np.min(x), np.max(x), np.mean(x))
            x[x>3] = 3.01
            y = y!=-1
            self.efficiency_fo_truth_eta_plot.add_raw_values(x, y, tags)

        if 'efficiency_fo_truth_eta_flat_spectrum_wrt_energy' in self.plots:
            e, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True,
                                                                     not_found_value=-1, sum_multi=True)

            bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
            freq = []
            for i in range(len(bins) - 1):
                l = bins[i]
                h = bins[i + 1]
                filter = np.logical_and(e >= l, e < h)
                s = float(np.sum(filter)) / float((h-l))
                freq.append(s)

            x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True,
                                                                     not_found_value=-1, sum_multi=True)

            z = np.searchsorted(bins, e) - 1
            z = np.minimum(np.maximum(z,0), len(freq)-1)

            weights = np.array([1./freq[x] for x in z])


            x = np.abs(x)
            x[x > 3] = 3.01
            y = y != -1
            self.efficiency_fo_truth_eta_plot_flat_spectrum_wrt_energy.add_raw_values(x, y, tags, weights=weights)

        if 'fake_rate_fo_pred_eta' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'dep_eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x = np.abs(x)
            x[x>3] = 3.01
            y = np.equal(y, -1)
            self.fake_rate_fo_pred_eta_plot.add_raw_values(x,y, tags)

        if 'response_fo_truth_eta' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[l>3] = 3.01
            filter = y!=-1
            self.response_fo_truth_eta_plot.add_raw_values(l[filter], y[filter] / x[filter], tags)


        if 'response_fo_truth_eta_flat_spectrum_wrt_energy' in self.plots:
            e, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True,
                                                                     not_found_value=-1, sum_multi=True)

            bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
            freq = []
            for i in range(len(bins) - 1):
                l = bins[i]
                h = bins[i + 1]
                filter = np.logical_and(e >= l, e < h)
                s = float(np.sum(filter)) / float((h-l))
                freq.append(s)


            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)

            z = np.searchsorted(bins, e) - 1
            z = np.minimum(np.maximum(z,0), len(freq)-1)

            weights = np.array([1./freq[x] for x in z])
            l = np.abs(l)
            l[l>3] = 3.01
            filter = y!=-1

            self.response_fo_truth_eta_plot_flat_spectrum_wrt_energy.add_raw_values(l[filter], y[filter] / x[filter], tags, weights=weights)

        if 'response_fo_pred_eta' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'dep_eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[l>3] = 3.01
            filter = y!=-1
            self.response_fo_pred_eta_plot.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'efficiency_fo_truth_pid' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x = np.abs(x)
            x[(x > 29) & (x < 100)] = 29
            x[x==111] = 31
            x[x==211] = 32
            x[x==113] = 33
            x[x==213] = 34
            x[x==115] = 35
            x[x==215] = 36
            x[x==117] = 37
            x[x==217] = 38
            x[x==119] = 39
            x[x==219] = 40
            x[x==130] = 41
            x[x==310] = 42
            x[x==311] = 43
            x[x==321] = 44
            x[x==2212] = 46
            x[x==2112] = 47
            x[x>=100] = 49
            y = y!=-1
            self.efficiency_fo_truth_pid_plot.add_raw_values(x, y, tags)

        if 'response_fo_truth_pid' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[(l > 29) & (l < 100)] = 29
            l[l==111] = 31
            l[l==211] = 32
            l[l==113] = 33
            l[l==213] = 34
            l[l==115] = 35
            l[l==215] = 36
            l[l==117] = 37
            l[l==217] = 38
            l[l==119] = 39
            l[l==219] = 40
            l[l==130] = 41
            l[l==310] = 42
            l[l==311] = 43
            l[l==321] = 44
            l[l==2212] = 46
            l[l==2112] = 47
            l[l>=100] = 49
            filter = y!=-1
            self.response_fo_truth_pid_plot.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'confusion_matrix' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'pid_probability', numpy=False, not_found_value=None, sum_multi=True)

            filter = np.argwhere(np.array([a is not None for a in y], np.bool))
            filter = filter[:, 0]


            x = [x[i] for i in filter]
            y = [y[i] for i in filter]

            x = np.array(x)
            y = np.array(y)

            y = np.argmax(y, axis=1)
            x = np.argmax(one_hot_encode_id(x, n_classes=4), axis=1)

            self.confusion_matrix_plot.classes = metadata['classes']
            self.confusion_matrix_plot.add_raw_values(x, y)

        if 'roc_curves' in self.plots:
            x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pid', 'pid_probability',
                                                                     numpy=False, not_found_value=None, sum_multi=True)

            filter = np.argwhere(np.array([a is not None for a in y], np.bool))
            filter = filter[:, 0]

            x = [x[i] for i in filter]
            y = [y[i] for i in filter]

            x = np.array(x)
            y = np.array(y)


            self.roc_curves.classes = metadata['classes']
            x = one_hot_encode_id(x, n_classes=len(metadata['classes']))
            self.roc_curves.add_raw_values(x, y)

        if 'resolution_fo_true_energy' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_fo_true_energy.add_raw_values(x[filter], y[filter] / x[filter], tags)


        if 'resolution_fo_local_shower_fraction' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_fo_local_shower_energy_fraction.add_raw_values(l[filter], y[filter] / x[filter], tags)


        if 'resolution_fo_eta' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[l>3] = 3.01
            filter = y!=-1
            self.resolution_fo_true_eta.add_raw_values(l[filter], y[filter] / x[filter], tags)


        if 'resolution_sum_fo_true_energy' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_sum_fo_true_energy.add_raw_values(x[filter], y[filter] / x[filter], tags)


        if 'resolution_sum_fo_local_shower_fraction' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_sum_fo_local_shower_energy_fraction.add_raw_values(l[filter], y[filter] / x[filter], tags)


        if 'resolution_sum_fo_eta' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l = np.abs(l)
            l[l>3] = 3.01
            filter = y!=-1
            self.resolution_sum_fo_true_eta.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'efficiency_fo_pt' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = np.not_equal(y, -1)
            self.efficiency_fo_pt.add_raw_values(x, y, tags)


        if 'fake_rate_fo_pt' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'pt', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = np.not_equal(y, -1)
            self.fake_rate_fo_pt.add_raw_values(x, y, tags)


        if 'response_fo_pt' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.response_fo_pt.add_raw_values(l[filter], y[filter] / x[filter], tags)


        if 'resolution_fo_pt' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            l,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_fo_pt.add_raw_values(l[filter], y[filter] / x[filter], tags)

        if 'response_histogram' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_histogam.add_raw_values(data, tags)


        if 'response_histogram_divided' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_histogam_divided.add_raw_values(x[filter], data, tags)

        if 'response_pt_histogram' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'pt', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_pt_histogam.add_raw_values(data, tags)


        if 'response_pt_histogram_divided' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'pt', 'pt', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1

            data = y[filter] / x[filter]
            data[data>3] = 3

            self.response_pt_histogam_divided.add_raw_values(x[filter], data, tags)



    def write_to_pdf(self, pdfpath, formatter=lambda x:''):
        if os.path.exists(pdfpath):
            if os.path.isdir(pdfpath):
                shutil.rmtree(pdfpath)
            else:
                os.unlink(pdfpath)

        os.mkdir(pdfpath)

        pdf_efficiency = PdfPages(os.path.join(pdfpath,'efficiency.pdf'))
        pdf_response = PdfPages(os.path.join(pdfpath,'response.pdf'))
        pdf_pid = PdfPages(os.path.join(pdfpath,'pid.pdf'))
        pdf_fake_rate = PdfPages(os.path.join(pdfpath,'fake_rate.pdf'))
        pdf_others = PdfPages(os.path.join(pdfpath,'others.pdf'))
        pdf_resolution = PdfPages(os.path.join(pdfpath,'resolution.pdf'))


        pdf_response_histos = PdfPages(os.path.join(pdfpath,'response_histos.pdf'))

        if 'settings' in self.plots:
            self._draw_numerics()
            pdf_others.savefig()

        if 'efficiency_fo_truth' in self.plots:
            self.efficiency_plot.draw(formatter)
            pdf_efficiency.savefig()

        if 'fake_rate_fo_pred' in self.plots:
            self.fake_rate_plot.draw(formatter)
            pdf_fake_rate.savefig()

        if 'response_fo_truth' in self.plots:
            self.response_plot.draw(formatter)
            pdf_response.savefig()

        if 'response_fo_pred' in self.plots:
            self.response_fo_pred_plot.draw(formatter)
            pdf_response.savefig()

        if 'response_sum_fo_truth' in self.plots:
            self.response_sum_plot.draw(formatter)
            pdf_response.savefig()

        if 'energy_resolution' in self.plots:
            # TODO: remove comments when added
            # self.resolution_histogram_plot.draw(formatter)
            # pdf.savefig()
            pass

        if 'response_fo_local_shower_energy_fraction' in self.plots:
            self.response_fo_local_shower_energy_fraction.draw(formatter)
            pdf_response.savefig()

        if 'response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy' in self.plots:
            self.response_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy.draw(formatter)
            pdf_response.savefig()

        if 'efficiency_fo_local_shower_energy_fraction' in self.plots:
            self.efficiency_fo_local_shower_energy_fraction.draw(formatter)
            pdf_efficiency.savefig()

        if 'efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy' in self.plots:
            self.efficiency_fo_local_shower_energy_fraction_flat_spectrum_wrt_energy.draw(formatter)
            pdf_efficiency.savefig()

        if 'efficiency_fo_truth_eta' in self.plots:
            self.efficiency_fo_truth_eta_plot.draw(formatter)
            pdf_efficiency.savefig()

        if 'efficiency_fo_truth_eta_flat_spectrum_wrt_energy' in self.plots:
            self.efficiency_fo_truth_eta_plot_flat_spectrum_wrt_energy.draw(formatter)
            pdf_efficiency.savefig()

        if 'fake_rate_fo_pred_eta' in self.plots:
            self.fake_rate_fo_pred_eta_plot.draw(formatter)
            pdf_fake_rate.savefig()

        if 'response_fo_truth_eta' in self.plots:
            self.response_fo_truth_eta_plot.draw(formatter)
            pdf_response.savefig()

        if 'response_fo_truth_eta_flat_spectrum_wrt_energy' in self.plots:
            self.response_fo_truth_eta_plot_flat_spectrum_wrt_energy.draw(formatter)
            pdf_response.savefig()

        if 'response_fo_pred_eta' in self.plots:
            self.response_fo_pred_eta_plot.draw(formatter)
            pdf_response.savefig()

        if 'efficiency_fo_truth_pid' in self.plots:
            self.efficiency_fo_truth_pid_plot.draw(formatter)
            pdf_efficiency.savefig()

        if 'response_fo_truth_pid' in self.plots:
            self.response_fo_truth_pid_plot.draw(formatter)
            pdf_response.savefig()

        if 'confusion_matrix' in self.plots:
            fig = self.confusion_matrix_plot.draw(formatter)
            pdf_pid.savefig(fig)

            self.confusion_matrix_plot.dont_plot(3)
            fig = self.confusion_matrix_plot.draw(formatter)
            pdf_pid.savefig(fig)
            self.confusion_matrix_plot.dont_plot(None)

        if 'roc_curves' in self.plots:
            for i in range(4):
                self.roc_curves.set_primary_class(i)
                fig = self.roc_curves.draw(formatter)
                pdf_pid.savefig(figure=fig)

            for i in range(3):
                self.roc_curves.set_primary_class(i)
                self.roc_curves.dont_plot(3)
                fig = self.roc_curves.draw(formatter)
                pdf_pid.savefig(figure=fig)

            # self.roc_curves.set_primary_class(0)
            # fig = self.roc_curves.draw(formatter)
            # print(fig)
            # fig.savefig('xyza.png')
            # print("Saving to png")
            # pdf_pid.savefig(figure=fig)

        if 'resolution_fo_true_energy':
            fig = self.resolution_fo_true_energy.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'resolution_fo_local_shower_fraction':
            fig = self.resolution_fo_local_shower_energy_fraction.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'resolution_fo_eta':
            fig = self.resolution_fo_true_eta.draw(formatter)
            pdf_resolution.savefig(fig)


        if 'resolution_sum_fo_true_energy':
            fig = self.resolution_sum_fo_true_energy.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'resolution_sum_fo_local_shower_fraction':
            fig = self.resolution_sum_fo_local_shower_energy_fraction.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'resolution_sum_fo_eta':
            fig = self.resolution_sum_fo_true_eta.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'efficiency_fo_pt' in self.plots:
            fig = self.efficiency_fo_pt.draw(formatter)
            pdf_efficiency.savefig(fig)

        if 'fake_rate_fo_pt' in self.plots:
            fig = self.fake_rate_fo_pt.draw(formatter)
            pdf_fake_rate.savefig(fig)

        if 'response_fo_pt' in self.plots:
            fig = self.response_fo_pt.draw(formatter)
            pdf_response.savefig(fig)

        if 'resolution_fo_pt' in self.plots:
            fig = self.resolution_fo_pt.draw(formatter)
            pdf_resolution.savefig(fig)

        if 'response_histogram' in self.plots:
            fig = self.response_histogam.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_histogram_divided' in self.plots:
            fig = self.response_histogam_divided.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_pt_histogram' in self.plots:
            fig = self.response_pt_histogam.draw(formatter)
            pdf_response_histos.savefig(fig)

        if 'response_pt_histogram_divided' in self.plots:
            fig = self.response_pt_histogam_divided.draw(formatter)
            pdf_response_histos.savefig(fig)

        # if 'energy_found_fo_truth' in self.plots:
        #     self.energy_found_fo_truth_plot.draw(formatter)
        #     pdf.savefig()
        # if 'energy_found_fo_pred' in self.plots:
        #     self.energy_found_fo_pred_plot.draw(formatter)
        #     pdf.savefig()
        #
        # if 'energy_found_fo_truth' in self.plots and 'energy_found_fo_pred' in self.plots and len(self.energy_found_fo_truth_plot.models_data)==1:
        #     EnergyFoundFoTruthEnergyPlot.draw_together_scalar_metrics(self.energy_found_fo_truth_plot, self.energy_found_fo_pred_plot)
        #     pdf.savefig()

        #
        pdf_efficiency.close()
        pdf_response.close()
        pdf_fake_rate.close()
        pdf_others.close()
        pdf_pid.close()
        pdf_resolution.close()
        pdf_response_histos.close()

        plt.close('all')
