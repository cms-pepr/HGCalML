from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot
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
from hplots.general_hist_plot import GeneralHistogramPlot
from hplots.general_graph_plot import GeneralGraphPlot
import hplots.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import experiment_database_reading_manager
import matching_and_analysis


class HGCalAnalysisPlotter:
    def __init__(self, plots = ['settings', 'efficiency_fo_truth', 'fake_rate_fo_pred', 'response_fo_truth',
                                'response_fo_pred', 'response_sum_fo_truth', 'energy_resolution_estimator','energy_resolution','energy_resolution_correlation',
                                'energy_found_fo_truth', 'energy_found_fo_pred','efficiency_fo_local_shower_energy_fraction','response_fo_local_shower_energy_fraction',
                                'efficiency_fo_truth_eta','fake_rate_fo_pred_eta', 'response_fo_truth_eta', 'response_fo_pred_eta',
                                'efficiency_fo_truth_pid', 'response_fo_truth_pid'],log_of_distributions=True):
        self.efficiency_plot = EfficiencyFoTruthEnergyPlot(histogram_log=log_of_distributions)
        self.fake_rate_plot = FakeRateFoPredEnergyPlot(histogram_log=log_of_distributions)
        self.response_plot = ResponseFoEnergyPlot(histogram_log=log_of_distributions)
        self.response_fo_pred_plot = ResponseFoEnergyPlot(x_label='Pred energy [GeV]', y_label='Response mean (pred energy/truth energy)', histogram_log=log_of_distributions)
        self.response_sum_plot = ResponseFoEnergyPlot(y_label='Response (sum/truth)', histogram_log=log_of_distributions)
        self.response_fo_local_shower_energy_fraction = ResponseFoLocalShowerEnergyFractionPlot()
        self.efficiency_fo_local_shower_energy_fraction = EfficiencyFoLocalShowerEnergyFractionPlot()
        self.efficiency_fo_truth_eta_plot = EfficiencyFoTruthEtaPlot(histogram_log=log_of_distributions)
        self.fake_rate_fo_pred_eta_plot = FakeRateFoPredEtaPlot(histogram_log=log_of_distributions)
        self.response_fo_truth_eta_plot = ResponseFoTruthEtaPlot(histogram_log=log_of_distributions)
        self.response_fo_pred_eta_plot = ResponseFoTruthEtaPlot(x_label='abs(Pred eta)', y_label='Response mean (pred energy/truth energy)', histogram_log=log_of_distributions)
        self.efficiency_fo_truth_pid_plot = EfficiencyFoTruthPIDPlot(histogram_log=log_of_distributions)
        self.response_fo_truth_pid_plot = ResponseFoTruthPIDPlot(histogram_log=log_of_distributions)


        self.energy_found_fo_truth_plot = EnergyFoundFoTruthEnergyPlot()
        self.energy_found_fo_pred_plot = EnergyFoundFoPredEnergyPlot()

        #self.resolution_histogram_plot = GeneralHistogramPlot(bins=np.linspace(0,2,50),x_label='Energy Resolution', y_label='Frequency', title='Energy resolution', histogram_log=False)
        self.resolution_histogram_plot = GeneralHistogramPlot(bins=np.linspace(-1,1,50),x_label=r'Energy Resolution $\frac{E_{true}-E_{pred}}{E_{true}}$', y_label='Counts', title='', histogram_log=False)
        self.resolution_estimator_histogram_plot = GeneralHistogramPlot(bins=np.linspace(0,0.3,50),x_label='Energy Resolution Estimator', y_label='Counts', title='', histogram_log=False)
        self.resolution_estimator_correlation_plot = GeneralGraphPlot(x_label='Average Energy Resolution Estimator', y_label=r'Energy Resolution $\frac{E_{true}-E_{pred}}{E_{true}}$', title='', histogram_log=False)

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
        self.resolution_estimator_histogram_plot.write_to_database(database_manager, table_prefix+'_resolution_estimator_histogram_plot')
        self.resolution_estimator_correlation_plot.write_to_database(database_manager, table_prefix+'_resolution_estimator_correlation_plot')
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

        self.response_sum_plot.read_from_database(database_reading_manager, table_prefix + '_response_sum_plot', experiment_name=experiment_name, condition=condition)
        self.resolution_histogram_plot.read_from_database(database_reading_manager, table_prefix+'_resolution_histogram_plot', experiment_name=experiment_name, condition=condition)
        self.resolution_estimator_histogram_plot.read_from_database(database_reading_manager, table_prefix+'_resolution_estimator_histogram_plot', experiment_name=experiment_name, condition=condition)
        self.resolution_estimator_correlation_plot.read_from_database(database_reading_manager, table_prefix+'_resolution_estimator_correlation_plot', experiment_name=experiment_name, condition=condition)

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

        if 'energy_resolution_estimator' in self.plots:
            # energy resolution estimator fromt the network evaluated for truth matched
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy_unc', numpy=True, not_found_value=-1, sum_multi=True)
            filter = y!=-1
            self.resolution_estimator_histogram_plot.add_raw_values(y[filter], tags) #plot predicted resolution for truth matched showers

        if 'energy_resolution' in self.plots:
            #real energy resolution when truth is available
            truth_energy,pred_energy = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            true_resolution = (truth_energy[filter]-pred_energy[filter])/truth_energy[filter] #proper resolution deffinition
            #true_resolution = (truth_energy[filter]-pred_energy[filter])/pred_energy[filter] #v2 just to compare
            #true_resolution = (truth_energy[filter])/pred_energy[filter] # v3 following the derivation of estimator
            self.resolution_histogram_plot.add_raw_values(true_resolution, tags) 


        if 'energy_resolution_correlation' in self.plots:
            # energy resolution estimator fromt the network evaluated for truth matched
            _,energy_res_pred = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy_unc', numpy=True, not_found_value=-1, sum_multi=True)
            filter = energy_res_pred!=-1
            energy_res_pred = energy_res_pred[filter]

            truth_energy,pred_energy = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            true_resolution = (truth_energy[filter]-pred_energy[filter])/truth_energy[filter] #proper resolution deffinition
            #true_resolution = (truth_energy[filter]-pred_energy[filter])/pred_energy[filter] #v2 just to compare
            #true_resolution = (truth_energy[filter])/pred_energy[filter] # v3 following the derivation of estimator
            res_estim_values, res_true_quantiles = utils.profile(true_resolution,energy_res_pred,bins=30,range=[0,0.3],moments=False,average=True,quantiles=np.array([0.25,0.75])) 
            res_true_values =  0.5*(res_true_quantiles[1]-res_true_quantiles[0])
            self.resolution_estimator_correlation_plot.add_raw_values(res_estim_values,res_true_values, tags) #plot predicted resolution for truth matched showers

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

        if 'efficiency_fo_local_shower_energy_fraction' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'local_shower_energy_fraction', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = y!=-1
            self.efficiency_fo_local_shower_energy_fraction.add_raw_values(x, y, tags)

        if 'efficiency_fo_truth_eta' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'eta', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            x = np.abs(x)
            #print("ZZZZZZ", np.min(x), np.max(x), np.mean(x))
            x[x>3] = 3.01
            y = y!=-1
            self.efficiency_fo_truth_eta_plot.add_raw_values(x, y, tags)

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
            x[x>=100] = 41
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
            l[l>=100] = 41
            filter = y!=-1
            self.response_fo_truth_pid_plot.add_raw_values(l[filter], y[filter] / x[filter], tags)


    def write_to_pdf(self, pdfpath, formatter=lambda x:''):
        pdf = PdfPages(pdfpath)

        # TODO: draw settings page here

        if 'settings' in self.plots:
            self._draw_numerics()
            pdf.savefig()
        if 'efficiency_fo_truth' in self.plots:
            self.efficiency_plot.draw(formatter)
            pdf.savefig()
        if 'fake_rate_fo_pred' in self.plots:
            self.fake_rate_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_truth' in self.plots:
            self.response_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_pred' in self.plots:
            self.response_fo_pred_plot.draw(formatter)
            pdf.savefig()

        if 'response_sum_fo_truth' in self.plots:
            self.response_sum_plot.draw(formatter)
            pdf.savefig()

        if 'energy_resolution' in self.plots:
            self.resolution_histogram_plot.draw(formatter)
            pdf.savefig()

        if 'energy_resolution_estimator' in self.plots:
            self.resolution_estimator_histogram_plot.draw(formatter)
            pdf.savefig()

        if 'energy_resolution_correlation' in self.plots:
            self.resolution_estimator_correlation_plot.draw(formatter)
            pdf.savefig()

        if 'energy_found_fo_truth' in self.plots:
            self.energy_found_fo_truth_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_local_shower_energy_fraction' in self.plots:
            self.response_fo_local_shower_energy_fraction.draw(formatter)
            pdf.savefig()

        if 'efficiency_fo_local_shower_energy_fraction' in self.plots:
            self.efficiency_fo_local_shower_energy_fraction.draw(formatter)
            pdf.savefig()

        if 'efficiency_fo_truth_eta' in self.plots:
            self.efficiency_fo_truth_eta_plot.draw(formatter)
            pdf.savefig()

        if 'fake_rate_fo_pred_eta' in self.plots:
            self.fake_rate_fo_pred_eta_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_truth_eta' in self.plots:
            self.response_fo_truth_eta_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_pred_eta' in self.plots:
            self.response_fo_pred_eta_plot.draw(formatter)
            pdf.savefig()

        if 'efficiency_fo_truth_pid' in self.plots:
            self.efficiency_fo_truth_pid_plot.draw(formatter)
            pdf.savefig()

        if 'response_fo_truth_pid' in self.plots:
            self.response_fo_truth_pid_plot.draw(formatter)
            pdf.savefig()

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


        pdf.close()
        plt.close('all')
