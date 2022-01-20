import numpy as np
import matplotlib.pyplot as plt


class GeneralHistogramPlot():
    def __init__(self, bins, x_label='Values', y_label='Frequency', title='', histogram_log=False, histogram_fractions=True):
        self.models_data = list()
        self.e_bins = bins

        if bins is not None:
            if type(bins) is not np.ndarray:
                raise ValueError("bins has to be numpy array")

        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.histogram_log=histogram_log
        self.histogram_fractions=True
        # self.e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]


    def _compute(self, values):

        if self.e_bins is not None:
            e_bins = self.e_bins
        else:
            _, e_bins = np.histogram(values)

        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)


        hist_values, _ = np.histogram(values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)

        return processed_data

    def add_raw_values(self, values, tags={}):
        if type(values) is not np.ndarray:
            raise ValueError("x values has to be numpy array")

        data = self._compute(values)
        data['tags'] = tags
        self.models_data.append(data)

    def add_processed_data(self, processed_data):
        self.models_data.append(processed_data)

    def draw(self, name_tag_formatter=None, axis=None):
        """

        :param name_tag_formatter: a function to which tags dict is given and it returns the name
        :return:
        """
        if axis is None:
            fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        else:
            ax1 = axis

        do_legend = False

        max_of_hist_values = 0
        for model_data in self.models_data:
            lows = model_data['bin_lower_energy']
            highs = model_data['bin_upper_energy']
            hist_values = model_data['hist_values']

            if self.e_bins is not None:
                e_bins = self.e_bins
            else:
                e_bins = np.array(lows.tolist() + [highs[-1]])

            e_bins_n = np.array(e_bins)
            e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

            if self.histogram_fractions:
                hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
                hist_values = (hist_values / np.sum(hist_values))

            tags = model_data['tags']

            if name_tag_formatter is None:
                name_of_plot = ''
            else:
                name_of_plot = name_tag_formatter(tags)

            do_legend = do_legend or len(name_of_plot) > 0

            hist_values = hist_values.tolist()

            e_bins = np.concatenate(([lows[0]], highs), axis=0)
            max_of_hist_values = max(max_of_hist_values, np.max(hist_values))

            ax1.step(e_bins, [hist_values[0]] + hist_values, alpha=0.7, label=name_of_plot)
            # ax1.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

            if self.histogram_log:
                ax1.set_yscale('log')
            ax1.set_title(self.title)

        ax1.set_xlabel(self.x_label)
        ax1.set_ylabel(self.y_label)
        if do_legend:
            ax1.legend(loc='center right')

        # ax1.set_ylim(0, 1.04)
        # ax2.set_ylim(0, max_of_hist_values * 1.3)

    @classmethod
    def draw_static(cls, x_values, y_values):
        plotter = GeneralHistogramPlot()
        plotter.add_raw_values(x_values, y_values, tags=None)
        plotter.draw()

    def write_to_database(self, database_manager, table_name):
        # print("Iterating")
        for model_data in self.models_data:
            # print("Iterating xyz")
            lows = model_data['bin_lower_energy']
            highs = model_data['bin_upper_energy']
            hist_values = model_data['hist_values']
            tags = model_data['tags']

            database_data = dict()
            for i in range(len(lows)):
                # database_data['bin_lower_energy'] = lows[i]
                # database_data['bin_upper_energy'] = highs[i]
                database_data['hist_values_%d'%i] = float(hist_values[i])

            for tag_name, tag_value in tags.items():
                database_data[tag_name] = tag_value

            # print("Inserting ", database_data, table_name)
            database_manager.insert_experiment_data(table_name, database_data)


    def get_tags(self):
        return [x['tags'] for x in self.models_data]

    def read_from_database(self, database_reading_manager, table_name, experiment_name=None, condition=None):
        results_dict = database_reading_manager.get_data(table_name, experiment_name, condition_string=condition)
        num_rows = len(results_dict['experiment_name'])

        results_dict_copy = results_dict.copy()

        for i in range(len(self.e_bins) - 1):
            results_dict_copy.pop('hist_values_%d' % i)

        tags_names = results_dict_copy.keys()

        # print(results_dict.keys())

        for row in range(num_rows):
            processed_data = dict()

            lows = []
            highs = []
            hist_values = []
            for i in range(len(self.e_bins) - 1):
                l = self.e_bins[i]
                h = self.e_bins[i + 1]
                lows.append(l)
                highs.append(h)
                hist_values.append(float(results_dict['hist_values_%d'%i][row]))

            processed_data['hist_values'] = np.array(hist_values)

            processed_data['bin_lower_energy'] = np.array(lows)
            processed_data['bin_upper_energy'] = np.array(highs)
            tags = dict()

            for tag_name in tags_names:
                tags[tag_name] = results_dict[tag_name][row]

            processed_data['tags'] = tags
            self.add_processed_data(processed_data)






