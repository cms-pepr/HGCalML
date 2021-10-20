import numpy as np
import matplotlib.pyplot as plt
from hplots.general_2d_plot import  General2dBinningPlot





class EfficiencyFoLocalFractionPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), x_label='Local shower fraction', y_label='Efficiency', title='Local shower fraction', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)


class ResponseFoIouPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), x_label='IOU', y_label='Response (pred/truth)', title='IOU response', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)

    def draw(self, name_tag_formatter=None, return_fig=False):
        fig = super().draw(name_tag_formatter, return_fig=True)
        axes = fig.axes
        axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes[0].axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig



class EfficiencyFoTruthEnergyPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),
                 x_label='Truth energy [GeV]', y_label='Reconstruction efficiency', title='Efficiency comparison', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)



class EnergyFoundFoTruthEnergyPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),
                 x_label='Truth energy [GeV]', y_label='% energy found', title='% truth energy matched', y_label_hist='Total truth energy / bin (fraction)',
                 histogram_fraction=True, histogram_log=False):
        super().__init__(bins, x_label, y_label, title, y_label_hist, histogram_fraction=histogram_fraction, histogram_log=histogram_log)

    def _compute(self, x_values, y_values):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []

        hist_values = []

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]


            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_x_values = x_values[filter].astype(float)

            hist_values.append(np.sum(filtered_x_values))


            m = np.sum(filtered_y_values) / np.sum(filtered_x_values)

            mean.append(m)
            lows.append(l)
            highs.append(h)


        # hist_values, _ = np.histogram(x_values, bins=e_bins)
        hist_values = np.array(hist_values)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)

        return processed_data


    @classmethod
    def draw_together_scalar_metrics(cls, truth_plot, pred_plot, return_fig=False):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

        max_of_hist_values = 0
        def draw(model_data, name, xlabel, ylabel):
            lows = model_data['bin_lower_energy']
            highs = model_data['bin_upper_energy']
            hist_values = model_data['hist_values']
            mean = model_data['mean']

            e_bins = truth_plot.e_bins
            e_bins_n = np.array(e_bins)
            e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


            mean = mean.tolist()
            print(mean)

            e_bins = np.concatenate(([lows[0]], highs), axis=0)

            ax1.set_title('score')

            ax1.step(e_bins, [mean[0]] + mean, label=name)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)

        draw(truth_plot.models_data[0], xlabel='Energy', ylabel='Metric (legend)', name='% matched truth')
        draw(pred_plot.models_data[0], xlabel='Energy', ylabel='Metric (legend)', name='% matched pred')
        model_data_f = truth_plot.models_data[0].copy()
        model_data_f['mean'] = 2 * truth_plot.models_data[0]['mean'] * pred_plot.models_data[0]['mean'] /\
                               (truth_plot.models_data[0]['mean'] + pred_plot.models_data[0]['mean'])
        draw(model_data_f, xlabel='Energy', ylabel='Metric (legend)', name='F-1')

        ax1.legend(loc='center right')

        # ax1.set_ylim(0, 1.04)
        # ax2.set_ylim(0, max_of_hist_values * 1.3)
        if return_fig:
            return fig


class EnergyFoundFoPredEnergyPlot(EnergyFoundFoTruthEnergyPlot):
    def __init__(self, bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),
                 x_label='Pred energy [GeV]', y_label='% energy found', title='% of pred energy matched', y_label_hist='Total predicted energy / bin (fraction)',
                 histogram_fraction=True, histogram_log=False):
        super().__init__(bins, x_label, y_label, title, y_label_hist, histogram_fraction=histogram_fraction, histogram_log=histogram_log)


# class EfficiencyFoEtaPlot(General2dBinningPlot):
#     def __init__(self, bins=np.array([]),
#                  x_label='Truth energy [GeV]', y_label='Reconstruction efficiency', title='Efficiency comparison', y_label_hist='Histogram (fraction)'):
#         super().__init__(bins, x_label, y_label, title, y_label_hist)



class FakeRateFoPredEnergyPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),
                 x_label='Pred energy [GeV]', y_label='Fake rate', title='Fake rate comparison', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)


class ResponseFoTruthEnergyPlot(General2dBinningPlot):
    def __init__(self,
                 bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80,
                       90, 100, 120, 140, 160, 180, 200]),
                 x_label='Truth energy [GeV]', y_label='Response', title='Response comparison',
                 y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)

    def draw(self, name_tag_formatter=None, return_fig=False):
        fig = super().draw(name_tag_formatter, return_fig=True)
        axes = fig.axes
        axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes[0].axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig


class ResolutionFoTruthEnergyPlot(General2dBinningPlot):
    def __init__(self,
                 bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80,
                       90, 100, 120, 140, 160, 180, 200]),
                 x_label='Truth energy [GeV]', y_label='Resolution', title='Resolution comparison',
                 y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)

    def _compute(self, x_values, y_values):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(x_values > l, x_values < h))
            filtered_y_values = y_values[filter].astype(np.float)

            m = np.mean(filtered_y_values)
            m = (np.std(filtered_y_values - m) / m)
            mean.append(m)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)


        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)

        return processed_data

