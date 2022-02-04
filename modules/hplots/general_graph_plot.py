import numpy as np
import matplotlib.pyplot as plt


class GeneralGraphPlot():
    def __init__(self, x_label='Values', y_label='Frequency', title='', histogram_log=False):
        self.models_data = list()

        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.histogram_log=histogram_log


    def _compute(self, x_values,y_values):
        processed_data = dict()
        processed_data['x_values'] = x_values
        processed_data['y_values'] = y_values
        return processed_data

    def add_raw_values(self, x_values,y_values, tags={}):
        if type(x_values) is not np.ndarray or type(y_values) is not np.ndarray:
            raise ValueError("x and y values has to be numpy array")

        data = self._compute(x_values,y_values)
        data['tags'] = tags
        self.models_data.append(data)

    def add_processed_data(self, processed_data):
        self.models_data.append(processed_data)

    def draw(self, name_tag_formatter=None):
        """

        :param name_tag_formatter: a function to which tags dict is given and it returns the name
        :return:
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

        max_of_hist_values = 0
        for model_data in self.models_data:
            x_values = model_data['x_values']
            y_values = model_data['y_values']

            tags = model_data['tags']

            if name_tag_formatter is None:
                name_of_plot = ''
            else:
                name_of_plot = name_tag_formatter(tags)

            print(name_of_plot)

            ax1.plot(x_values, y_values, color='black',linestyle='None', alpha=1,marker='o')

            if self.histogram_log:
                ax1.set_yscale('log')
            ax1.set_title(self.title, fontsize=14)

            ax1.set_xlabel(self.x_label, fontsize=14)
            ax1.set_ylabel(self.y_label, fontsize=14)
            #ax1.legend(loc='center right')
            plt.subplots_adjust(left=0.15)

        # ax1.set_ylim(0, 1.04)
        # ax2.set_ylim(0, max_of_hist_values * 1.3)

    @classmethod
    def draw_static(cls, x_values, y_values):
        plotter = GeneralGraphPlot()
        plotter.add_raw_values(x_values, y_values, tags=None)
        plotter.draw()

    def write_to_database(self, database_manager, table_name):
        pass


    def get_tags(self):
        return [x['tags'] for x in self.models_data]

    def read_from_database(self, database_reading_manager, table_name, experiment_name=None, condition=None):
        pass





