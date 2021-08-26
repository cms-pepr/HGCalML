import os

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.general_hist_plot import GeneralHistogramPlot
import matplotlib.pyplot as plt
import numpy as np
import sql_credentials
import unittest



class DatabasePlotsTestCases(unittest.TestCase):
    def write_to_database(self, database_manager):
        database_manager.set_experiment('database_plots_test_case_1')

        histogram_plot = GeneralHistogramPlot(bins=np.array([0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]), histogram_log=True)

        values = np.random.normal(loc=8, scale=3, size=(100))

        #
        # print(type(x_values))
        histogram_plot.add_raw_values(values=values,
                                       tags={'beta_threshold': 0.1, 'dist_threshold': 0.9})
        histogram_plot.draw()
        histogram_plot.write_to_database(database_manager,table_name='database_plots_test_case_1')
        #
        plt.savefig('output/test-original-plot.png')

    def read_from_database(self, database_reading_manager):
        histogram_plot = GeneralHistogramPlot(bins=np.array([0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]), histogram_log=True)
        histogram_plot.read_from_database(database_reading_manager, 'database_plots_test_case_1')
        histogram_plot.draw()

        plt.savefig('output/test-reproduced-plot.png')
    def test_read_write(self):
        print("Writing to server")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        self.write_to_database(database_manager)
        database_manager.close()

        print("Reading back from server")
        database_reading_manager = ExperimentDatabaseReadingManager(sql_credentials.credentials)
        self.read_from_database(database_reading_manager)

        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.close()

    def test_read_write_file(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')
        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        print("Writing to file")
        self.write_to_database(database_manager)
        database_manager.close()

        print("Reading back from file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='sample.db')
        self.read_from_database(database_reading_manager)

        if os.path.exists('sample.db'):
            os.unlink('sample.db')


if __name__ == '__main__':
    unittest.main()