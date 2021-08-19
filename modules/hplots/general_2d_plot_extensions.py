import numpy as np
import matplotlib.pyplot as plt
from hplots.general_2d_plot import  General2dBinningPlot





class EfficiencyFoLocalFractionPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), x_label='Local shower fraction', y_label='Efficiency', title='Local shower fraction', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)


class ResponseFoIouPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), x_label='IOU', y_label='Response (pred/truth)', title='IOU response', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)



class EfficiencyFoTruthEnergyPlot(General2dBinningPlot):
    def __init__(self, bins=np.array([0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]),
                 x_label='Truth energy [GeV]', y_label='Reconstruction efficiency', title='Efficiency comparison', y_label_hist='Histogram (fraction)'):
        super().__init__(bins, x_label, y_label, title, y_label_hist)



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

