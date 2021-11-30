from matplotlib.backends.backend_pdf import PdfPages

from hplots.general_2d_plot import General2dBinningPlot
from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot, EffFakeRatePlot
from hplots.general_2d_plot_extensions import FakeRateFoPredEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoEnergyPlot
from hplots.general_2d_plot_extensions import EnergyFoundFoPredEnergyPlot
from hplots.general_2d_plot_extensions import EnergyFoundFoTruthEnergyPlot
import numpy as np
import matplotlib.pyplot as plt
import experiment_database_reading_manager
from hplots.general_hist_plot import GeneralHistogramPlot
import matching_and_analysis




class EffFakeFoNumHitsPlot(EffFakeRatePlot):
    def __init__(self, bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
                 x_label='Num hits', y_label='Reconstruction efficiency', title='Efficiency comparison', y_label_hist='Histogram (fraction)',histogram_log=False):
        super().__init__(bins, x_label, y_label, title, y_label_hist, histogram_log=histogram_log)


class TrackMLPlotter:
    def __init__(self):
        # self.plots = set(['settings', 'efficiency_fo_truth','hist_num_hits_per_pred', 'hist_num_hits_per_truth',
        #                   'eff_fo_num_hits', 'fake_fo_num_hits'])

        self.plots = set(['settings', 'efficiency_fo_truth',
                          'eff_fo_num_hits', 'fake_fo_num_hits',
                          'eff_fo_pt_1_4','eff_fo_pt_4_10',
                          'eff_fo_pt_10_20'])
        self.efficiency_plot = EfficiencyFoTruthEnergyPlot(
            bins=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.5,3.0,4,5,6,7,8,9,10,11]),
            x_label='pT (GeV)',
            y_label='Reconstruction Efficiency',
            histogram_log=False
        )


        self.efficiency_fo_pt_plot_1_4 = EfficiencyFoTruthEnergyPlot(
            bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]),
            x_label='pT (GeV)',
            y_label='Reconstruction Efficiency (nhits 1-3)',
            title='nhits 1-3',
            histogram_log=False
        )
        self.efficiency_fo_pt_plot_4_10 = EfficiencyFoTruthEnergyPlot(
            bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]),
            x_label='pT (GeV)',
            y_label='Reconstruction Efficiency (nhits 4-10)',
            title='nhits 4-10',
            histogram_log=False)
        self.efficiency_fo_pt_plot_10_20 = EfficiencyFoTruthEnergyPlot(
            bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]),
            x_label='pT (GeV)',
            y_label='Reconstruction Efficiency (nhits 10-20)',
            title='nhits 10-20',
            histogram_log=False
        )

        # self.fake_fo_num_hits_plot_1_4 = EfficiencyFoTruthEnergyPlot(title='Fake rate', y_label='Fake rate',
        #     bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]))
        # self.fake_fo_num_hits_plot_4_10 = EfficiencyFoTruthEnergyPlot(title='Fake rate', y_label='Fake rate',
        #     bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]))
        # self.fake_fo_num_hits_plot_10_20 = EfficiencyFoTruthEnergyPlot(title='Fake rate', y_label='Fake rate',
        #     bins=np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10, 11]))


        # self.num_hits_hist_pred = GeneralHistogramPlot(bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        #                                                x_label='Num hits',
        #                                                y_label='Predicted number of hits per track'
        #                                                )
        #
        # self.num_hits_hist_truth = GeneralHistogramPlot(bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        #                                                x_label='Num hits',
        #                                                y_label='Truth number of hits per track'
        #                                                )

        self.efficiency_fo_num_hits_plot = EffFakeFoNumHitsPlot()
        self.fake_fo_num_hits_plot = EffFakeFoNumHitsPlot(title='Fake rate', y_label='Fake rate')

        self.dist_thresholds = []
        self.beta_thresholds = []
        self.iou_thresholds = []
        self.matching_types = []

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

        s = 'Beta threshold: %s\nDist threshold: %s\niou  threshold: %s\nMatching types: %s\n' % (bs, ds, iss, matching_types)

        plt.text(0, 1, s, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                 fontdict=text_font)

    def write_data_to_database(self, database_manager, table_prefix):
        self.efficiency_plot.write_to_database(database_manager, table_prefix+'_efficiency_plot_fo_pt')
        # self.num_hits_hist_pred.write_to_database(database_manager, table_prefix+'_efficiency_plot')
        # self.num_hits_hist_truth.write_to_database(database_manager, table_prefix+'_efficiency_plot')
        self.efficiency_fo_num_hits_plot.write_to_database(database_manager, table_prefix+'_efficiency_plot_fo_nhits')
        self.fake_fo_num_hits_plot.write_to_database(database_manager, table_prefix+'_fake_rate_plot_fo_nhits')

        database_manager.flush()

    def add_data_from_database(self, database_reading_manager, table_prefix, experiment_name=None, condition=None):
        self.efficiency_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_plot_fo_pt', experiment_name=experiment_name, condition=condition)
        self.efficiency_fo_num_hits_plot.read_from_database(database_reading_manager, table_prefix + '_efficiency_plot_fo_nhits', experiment_name=experiment_name, condition=condition)
        self.fake_fo_num_hits_plot.read_from_database(database_reading_manager, table_prefix + '_fake_rate_plot_fo_nhits', experiment_name=experiment_name, condition=condition)

        tags = self.efficiency_plot.get_tags()

        self.beta_thresholds += [x['beta_threshold'] for x in tags]
        self.dist_thresholds += [x['distance_threshold'] for x in tags]
        self.iou_thresholds += [x['iou_threshold'] for x in tags]
        # self.soft += [x['soft'] for x in tags]

        self.beta_thresholds = np.unique(self.beta_thresholds).tolist()
        self.dist_thresholds = np.unique(self.dist_thresholds).tolist()
        self.iou_thresholds = np.unique(self.iou_thresholds).tolist()
        # self.soft = np.unique(self.soft).tolist()

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

        if 'efficiency_fo_truth' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
            y = np.not_equal(y, -1)

            x[x>10.99] = 10.99
            self.efficiency_plot.add_raw_values(x, y, tags)

        # if 'hist_num_hits_per_pred' in self.plots:
        #     x,_ = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
        #                                                             numpy=True, not_found_value=-1, sum_multi=True)
        #     self.num_hits_hist_pred.add_raw_values(x)
        #
        # if 'hist_num_hits_per_truth' in self.plots:
        #     x, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
        #                                                             numpy=True, not_found_value=-1, sum_multi=True)
        #     self.num_hits_hist_truth.add_raw_values(x)

        if 'eff_fo_num_hits' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            y = np.not_equal(y, -1)
            self.efficiency_fo_num_hits_plot.add_raw_values(x,y, tags)

        if 'fake_fo_num_hits' in self.plots:
            x,y = matching_and_analysis.get_pred_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            y = np.equal(y, -1)
            self.fake_fo_num_hits_plot.add_raw_values(x,y, tags)

        if 'eff_fo_pt_1_4' in self.plots:
            x,y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            x2,_ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)

            filter = x < 4

            x2 = x2[filter]
            y = y[filter]

            y = np.not_equal(y, -1)
            self.efficiency_fo_pt_plot_1_4.add_raw_values(x2, y, tags)

        if 'eff_fo_pt_4_10' in self.plots:
            x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            x2, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)

            filter = np.logical_and(x >=4 , x<10)

            x2 = x2[filter]
            y = y[filter]

            y = np.not_equal(y, -1)
            self.efficiency_fo_pt_plot_4_10.add_raw_values(x2, y, tags)

        if 'eff_fo_pt_10_20' in self.plots:
            x, y = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'num_hits', 'num_hits',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            x2, _ = matching_and_analysis.get_truth_matched_attribute(analysed_graphs, 'energy', 'energy',
                                                                    numpy=True, not_found_value=-1, sum_multi=True)
            filter = x >= 10
            x2 = x2[filter]
            y = y[filter]

            y = np.not_equal(y, -1)
            self.efficiency_fo_pt_plot_10_20.add_raw_values(x2, y, tags)



    def write_to_pdf(self, pdfpath, formatter=lambda x:''):
        pdf = PdfPages(pdfpath)

        # TODO: draw settings page here

        if 'settings' in self.plots:
            self._draw_numerics()
            pdf.savefig()
        if 'efficiency_fo_truth' in self.plots:
            self.efficiency_plot.draw(formatter)
            pdf.savefig()

        # if 'hist_num_hits_per_pred' in self.plots:
        #     self.num_hits_hist_pred.draw(formatter)
        #     pdf.savefig()
        #
        # if 'hist_num_hits_per_truth' in self.plots:
        #     self.num_hits_hist_truth.draw(formatter)
        #     pdf.savefig()


        if 'eff_fo_num_hits' in self.plots:
            self.efficiency_fo_num_hits_plot.draw(formatter)
            pdf.savefig()

        if 'fake_fo_num_hits' in self.plots:
            self.fake_fo_num_hits_plot.draw(formatter)
            pdf.savefig()

        if 'eff_fo_pt_1_4' in self.plots:
            self.efficiency_fo_pt_plot_1_4.draw(formatter)
            pdf.savefig()

        if 'eff_fo_pt_4_10' in self.plots:
            self.efficiency_fo_pt_plot_4_10.draw(formatter)
            pdf.savefig()

        if 'eff_fo_pt_10_20' in self.plots:
            self.efficiency_fo_pt_plot_10_20.draw(formatter)
            pdf.savefig()

        pdf.close()
        plt.close('all')
