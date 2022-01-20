from hplots.general_hist_plot import GeneralHistogramPlot
import numpy as np
import matplotlib.pyplot as plt


class ResponseHisto(GeneralHistogramPlot):
    def __init__(self,
                 bins=np.array([0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.1]),
                 x_label='$E_{pred}/E_{true}$', y_label='Frequency', title='Response histogram', histogram_log=False):
        super().__init__(bins, x_label, y_label, title, histogram_log)



class Multi4HistEnergy():
    def __init__(self):
        hist_a = ResponseHisto(title='0 < $E_{true} < 5$')
        hist_b = ResponseHisto(title='5 < $E_{true} < 10$')
        hist_c = ResponseHisto(title='10 < $E_{true} < 50$')
        hist_d = ResponseHisto(title='50 < $E_{true} < 200$')

        self.all_histos = [hist_a, hist_b, hist_c, hist_d]
        self.all_his_limits = [(0,5), (5,10), (10,50), (50,200)]

    def add_raw_values(self, x_values, y_values, tags={}):
        if type(x_values) is not np.ndarray or type(y_values) is not np.ndarray:
            raise ValueError("x and y values has to be numpy array")

        for h, limits in zip(self.all_histos, self.all_his_limits):
            filter = np.logical_and(np.greater(x_values, limits[0]), np.less(x_values, limits[1]))
            values = y_values[filter]
            h.add_raw_values(values, tags)

    def draw(self, formatter=None):
        fig, ax1 = plt.subplots(2, 2, figsize=(9, 6))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        ax1 = [ax1[0][0],ax1[0][1],ax1[1][0],ax1[1][1]]

        for i, h in enumerate(self.all_histos):
            h.draw(formatter, ax1[i])

        return fig


class Multi4HistPt(Multi4HistEnergy):
    def __init__(self):
        hist_a = ResponseHisto(title='$0 < {p_T}_{true} < 5$',x_label='${p_T}_{true}/{p_T}_{pred}$')
        hist_b = ResponseHisto(title='$5 < {p_T}_{true} < 10$',x_label='${p_T}_{true}/{p_T}_{pred}$')
        hist_c = ResponseHisto(title='$10 < {p_T}_{true} < 20$',x_label='${p_T}_{true}/{p_T}_{pred}$')
        hist_d = ResponseHisto(title='$20 < {p_T}_{true} < 80$',x_label='${p_T}_{true}/{p_T}_{pred}$')

        self.all_histos = [hist_a, hist_b, hist_c, hist_d]
        self.all_his_limits = [(0,5), (5,10), (10,20), (20,80)]
