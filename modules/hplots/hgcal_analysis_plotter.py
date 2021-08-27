from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot
from hplots.general_2d_plot_extensions import FakeRateFoPredEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoTruthEnergyPlot
import numpy as np
import matplotlib.pyplot as plt

from hplots.general_hist_plot import GeneralHistogramPlot


def convert_dataset_dict_elements_to_numpy(dataset_dict):
    dataset_dict['beta_threshold'] = np.array(dataset_dict['beta_threshold'])
    dataset_dict['distance_threshold'] = np.array(dataset_dict['distance_threshold'])
    dataset_dict['iou_threshold'] = np.array(dataset_dict['iou_threshold'])

    dataset_dict['truth_shower_energy'] = np.array(dataset_dict['truth_shower_energy'])
    dataset_dict['truth_shower_eta'] = np.array(dataset_dict['truth_shower_eta'])
    dataset_dict['truth_shower_found_or_not'] = np.array(dataset_dict['truth_shower_found_or_not'])
    dataset_dict['truth_shower_found_or_not_ticl'] = np.array(dataset_dict['truth_shower_found_or_not_ticl'])
    dataset_dict['truth_shower_sid'] = np.array(dataset_dict['truth_shower_sid'])
    dataset_dict['truth_shower_sample_id'] = np.array(dataset_dict['truth_shower_sample_id'])

    dataset_dict['truth_shower_local_density'] = np.array(dataset_dict['truth_shower_local_density'])
    dataset_dict['truth_shower_closest_particle_distance'] = np.array(
        dataset_dict['truth_shower_closest_particle_distance'])

    dataset_dict['truth_shower_num_rechits'] = np.array(dataset_dict['truth_shower_num_rechits'])
    dataset_dict['endcap_num_rechits'] = np.array(dataset_dict['endcap_num_rechits'])

    dataset_dict['pred_shower_regressed_energy'] = np.array(dataset_dict['pred_shower_regressed_energy'])
    dataset_dict['pred_shower_matched_energy'] = np.array(dataset_dict['pred_shower_matched_energy'])
    dataset_dict['pred_shower_energy_sum'] = np.array(dataset_dict['pred_shower_energy_sum'])
    dataset_dict['pred_shower_matched_energy_sum'] = np.array(dataset_dict['pred_shower_matched_energy_sum'])
    dataset_dict['truth_shower_matched_iou_pred'] = np.array(dataset_dict['truth_shower_matched_iou_pred'])
    dataset_dict['truth_shower_matched_iou_ticl'] = np.array(dataset_dict['truth_shower_matched_iou_ticl'])


    dataset_dict['truth_shower_matched_energy_regressed'] = np.array(
        dataset_dict['truth_shower_matched_energy_regressed'])
    dataset_dict['truth_shower_matched_energy_regressed_ticl'] = np.array(
        dataset_dict['truth_shower_matched_energy_regressed_ticl'])

    dataset_dict['truth_shower_matched_energy_sum'] = np.array(dataset_dict['truth_shower_matched_energy_sum'])

    dataset_dict['pred_shower_regressed_phi'] = np.array(dataset_dict['pred_shower_regressed_phi'])
    dataset_dict['pred_shower_matched_phi'] = np.array(dataset_dict['pred_shower_matched_phi'])
    dataset_dict['pred_shower_regressed_eta'] = np.array(dataset_dict['pred_shower_regressed_eta'])
    dataset_dict['pred_shower_matched_eta'] = np.array(dataset_dict['pred_shower_matched_eta'])
    dataset_dict['pred_shower_matched_iou'] = np.array(dataset_dict['pred_shower_matched_iou'])
    dataset_dict['pred_shower_sid'] = np.array(dataset_dict['pred_shower_sid'])
    dataset_dict['pred_shower_sid_merged'] = np.array(dataset_dict['pred_shower_sid_merged'])
    dataset_dict['pred_shower_sample_id'] = np.array(dataset_dict['pred_shower_sample_id'])


    dataset_dict['ticl_shower_regressed_energy'] = np.array(dataset_dict['ticl_shower_regressed_energy'])
    dataset_dict['ticl_shower_matched_energy'] = np.array(dataset_dict['ticl_shower_matched_energy'])
    dataset_dict['ticl_shower_energy_sum'] = np.array(dataset_dict['ticl_shower_energy_sum'])
    dataset_dict['ticl_shower_matched_energy_sum'] = np.array(dataset_dict['ticl_shower_matched_energy_sum'])
    dataset_dict['ticl_shower_matched_iou'] = np.array(dataset_dict['ticl_shower_matched_iou'])

    dataset_dict['ticl_shower_regressed_phi'] = np.array(dataset_dict['ticl_shower_regressed_phi'])
    dataset_dict['ticl_shower_matched_phi'] = np.array(dataset_dict['ticl_shower_matched_phi'])
    dataset_dict['ticl_shower_regressed_eta'] = np.array(dataset_dict['ticl_shower_regressed_eta'])
    dataset_dict['ticl_shower_matched_eta'] = np.array(dataset_dict['ticl_shower_matched_eta'])
    dataset_dict['ticl_shower_sid'] = np.array(dataset_dict['ticl_shower_sid'])
    dataset_dict['ticl_shower_sid_merged'] = np.array(dataset_dict['ticl_shower_sid_merged'])
    dataset_dict['ticl_shower_sample_id'] = np.array(dataset_dict['ticl_shower_sample_id'])

    dataset_dict['endcap_num_truth_showers'] = np.array(dataset_dict['endcap_num_truth_showers'])
    dataset_dict['endcap_num_pred_showers'] = np.array(dataset_dict['endcap_num_pred_showers'])
    dataset_dict['endcap_num_found_showers'] = np.array(dataset_dict['endcap_num_found_showers'])
    dataset_dict['endcap_num_missed_showers'] = np.array(dataset_dict['endcap_num_missed_showers'])
    dataset_dict['endcap_num_fake_showers'] = np.array(dataset_dict['endcap_num_fake_showers'])

    dataset_dict['endcap_num_ticl_showers'] = np.array(dataset_dict['endcap_num_ticl_showers'])
    dataset_dict['endcap_num_found_showers_ticl'] = np.array(dataset_dict['endcap_num_found_showers_ticl'])
    dataset_dict['endcap_num_missed_showers_ticl'] = np.array(dataset_dict['endcap_num_missed_showers_ticl'])
    dataset_dict['endcap_num_fake_showers_ticl'] = np.array(dataset_dict['endcap_num_fake_showers_ticl'])

    dataset_dict['endcap_total_energy_pred'] = np.array(dataset_dict['endcap_total_energy_pred'])
    dataset_dict['endcap_total_energy_ticl'] = np.array(dataset_dict['endcap_total_energy_ticl'])
    dataset_dict['endcap_total_energy_truth'] = np.array(dataset_dict['endcap_total_energy_truth'])

    dataset_dict['visualized_segments'] = np.array(dataset_dict['visualized_segments'])

    return dataset_dict


class HGCalAnalysisPlotter:
    def __init__(self):
        self.efficiency_plot = EfficiencyFoTruthEnergyPlot()
        self.fake_rate_plot = FakeRateFoPredEnergyPlot()
        self.response_plot = ResponseFoTruthEnergyPlot()
        self.response_sum_plot = ResponseFoTruthEnergyPlot(y_label='Response (sum/truth)')

        # TODO: for Nadya
        self.resolution_histogram_plot = GeneralHistogramPlot(bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),x_label='Resolution', y_label='Frequency', title='Energy resolution', histogram_log=False)

        self.dist_thresholds = []
        self.beta_thresholds = []
        self.iou_thresholds = []
        self.soft = []

    def _draw_settings(self):
        text_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal',
                     'verticalalignment': 'bottom'}
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_visible(False)
        ax.axis('off')

        bs = ','.join(['%.2f'%x for x in self.beta_thresholds])
        ds = ','.join(['%.2f'%x for x in self.dist_thresholds])
        iss = ','.join(['%.2f'%x for x in self.iou_thresholds])
        softs = ','.join([str(x) for x in self.soft])

        s = 'Beta threshold: %s\nDist threshold: %s\niou  threshold: %s\n Is soft: %r' % (bs, ds, iss, softs)

        plt.text(0, 0.9, s, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                 fontdict=text_font)

    def write_data_to_database(self, database_manager, table_prefix):
        self.efficiency_plot.write_to_database(database_manager, table_prefix+'_efficiency_plot')
        self.fake_rate_plot.write_to_database(database_manager, table_prefix+'_fake_rate_plot')
        self.response_plot.write_to_database(database_manager, table_prefix+'_response_plot')
        self.response_sum_plot.write_to_database(database_manager, table_prefix+'_response_sum_plot')
        self.resolution_histogram_plot.write_to_database(database_manager, table_prefix+'_resolution_histogram_plot')
        database_manager.flush()

    def add_data_from_database(self, database_reading_manager, table_prefix, experiment_name=None):
        self.efficiency_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_plot', experiment_name=experiment_name)
        self.fake_rate_plot.read_from_database(database_reading_manager, table_prefix + '_fake_rate_plot', experiment_name=experiment_name)
        self.response_plot.read_from_database(database_reading_manager, table_prefix + '_response_plot', experiment_name=experiment_name)
        self.response_sum_plot.read_from_database(database_reading_manager, table_prefix + '_response_sum_plot', experiment_name=experiment_name)
        self.resolution_histogram_plot.read_from_database(database_reading_manager, table_prefix+'_resolution_histogram_plot', experiment_name=experiment_name)

        tags = self.efficiency_plot.get_tags()

        self.beta_thresholds += [x['beta_threshold'] for x in tags]
        self.dist_thresholds += [x['distance_threshold'] for x in tags]
        self.iou_thresholds += [x['iou_threshold'] for x in tags]
        self.soft += [x['soft'] for x in tags]

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()
        self.soft = np.unique(self.soft).tolist()

    def add_data_from_analysis_dict(self, dataset_analysis_dict, label='', additional_tags=dict()):
        self.dataset_analysis_dict = dataset_analysis_dict
        dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)

        tags = dict()
        tags['beta_threshold'] = float(dataset_analysis_dict['beta_threshold'])
        tags['distance_threshold'] = float(dataset_analysis_dict['distance_threshold'])
        tags['iou_threshold'] = float(dataset_analysis_dict['iou_threshold'])
        tags['soft'] = str(dataset_analysis_dict['soft'])
        tags['label'] = str(label)

        for key, value in additional_tags.items():
            tags[key] = value

        self.beta_thresholds.append(float(tags['beta_threshold']))
        self.dist_thresholds.append(float(tags['distance_threshold']))
        self.iou_thresholds.append(float(tags['iou_threshold']))
        self.soft.append(bool(tags['soft']))

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()
        self.soft = np.unique(self.soft).tolist()


        self.efficiency_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'],
                                       dataset_analysis_dict['truth_shower_found_or_not'], tags)

        self.fake_rate_plot.add_raw_values(dataset_analysis_dict['pred_shower_regressed_energy'],
                                      dataset_analysis_dict['pred_shower_matched_energy']==-1, tags)


        filter_truth_found = dataset_analysis_dict['truth_shower_found_or_not']

        self.response_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'][filter_truth_found],
                                     dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter_truth_found], tags)

        self.response_sum_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'][filter_truth_found],
                                         dataset_analysis_dict['truth_shower_matched_energy_sum'][filter_truth_found], tags)


        # TODO: Nadya Just adding a histogram of all the truth shower energy as a placeholder
        self.resolution_histogram_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'][filter_truth_found], tags)

    def write_to_pdf(self, pdfpath, formatter=lambda x:''):
        pdf = PdfPages(pdfpath)

        # TODO: draw settings page here

        self._draw_settings()
        pdf.savefig()
        self.efficiency_plot.draw(formatter)
        pdf.savefig()
        self.fake_rate_plot.draw(formatter)
        pdf.savefig()
        self.response_plot.draw(formatter)
        pdf.savefig()
        self.response_sum_plot.draw(formatter)
        pdf.savefig()
        self.resolution_histogram_plot.draw(formatter)
        pdf.savefig()

        pdf.close()
