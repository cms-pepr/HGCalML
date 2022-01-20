import numpy as np
import matplotlib.pyplot as plt
from hplots.general_2d_plot import  General2dBinningPlot

import hplots.response_scale
hplots.response_scale.register()

class ResponsePlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(ResponsePlot, self).__init__(**kwargs, yscale='response_scale')

    def draw(self, name_tag_formatter=None, return_fig=False):
        fig = super().draw(name_tag_formatter, return_fig=True)
        axes = fig.axes
        axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes[0].axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        error = []

        lows = []
        highs = []

        bug_in_error_comptuation = True

        if weights is None:
            bug_in_error_comptuation = False
            weights = np.ones_like(y_values)


        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]


            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_weights = weights[filter].astype(float)

            m = np.sum(filtered_y_values*filtered_weights)/np.sum(filtered_weights)
            mean.append(m)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(m / np.sqrt(float(len(filtered_y_values))))

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data



class ResolutionPlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(ResolutionPlot, self).__init__(**kwargs, yscale='response_scale')

    def draw(self, name_tag_formatter=None, return_fig=False):
        fig = super().draw(name_tag_formatter, return_fig=True)
        axes = fig.axes
        # axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes[0].axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []
        error = []

        bug_in_error_comptuation=True

        if weights is None:
            bug_in_error_comptuation=False
            weights = np.ones_like(y_values)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(x_values > l, x_values < h))
            filtered_y_values = y_values[filter].astype(np.float)
            filtered_weights = weights[filter].astype(float)
            # filtered_y_values = filtered_y_values * filtered_weights

            m = np.mean(filtered_y_values)
            relvar = (filtered_y_values - m) / m
            rms = np.sqrt(np.sum(filtered_weights * relvar ** 2) / np.sum(filtered_weights))

            # m = (np.std(filtered_y_values - m) / m)
            mean.append(rms)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(rms / np.sqrt(float(len(filtered_y_values))))



        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data


class EfficiencyFakeRatePlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(EfficiencyFakeRatePlot, self).__init__(**kwargs)

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        error = []

        lows = []
        highs = []

        bug_in_error_comptuation = True

        if weights is None:
            bug_in_error_comptuation = False
            weights = np.ones_like(y_values)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]


            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_weights = weights[filter].astype(float)
            m = np.sum(filtered_y_values*filtered_weights)/np.sum(filtered_weights)
            mean.append(m)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(np.sqrt(m * (1 - m) / len(filtered_y_values)))

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data


