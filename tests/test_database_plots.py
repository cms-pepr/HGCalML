import os

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.general_2d_plot import General2dBinningPlot
import matplotlib.pyplot as plt
import numpy as np
import sql_credentials
import unittest
from experiment_database_tools import download_experiment_to_file


class DatabasePlotsTestCases(unittest.TestCase):
    def write_to_database(self, database_manager):
        efficiency_plot = General2dBinningPlot(bins=np.array([0, 1, 2, 3, 4]), histogram_log=False,
                                               histogram_fraction=False)
        x_values = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        y_values = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        #
        # print(type(x_values))
        efficiency_plot.add_raw_values(x_values=x_values, y_values=y_values,
                                       tags={'beta_threshold': 0.1, 'dist_threshold': 0.9})
        efficiency_plot.draw()
        efficiency_plot.write_to_database(database_manager,table_name='database_plots_test_case_1')

        plt.savefig('output/test-original-plot.png')

    def read_from_database(self,database_reading_manager):
        efficiency_plot = General2dBinningPlot(bins=np.array([0, 1, 2, 3, 4]), histogram_log=False,
                                               histogram_fraction=False)
        efficiency_plot.read_from_database(database_reading_manager, 'database_plots_test_case_1')
        efficiency_plot.draw()

        plt.savefig('output/test-reproduced-plot.png')

    def test_read_write(self):
        print("Writing to server")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        print("Deleting experiment")
        database_manager.delete_experiment('database_plots_test_case_1')
        print("Setting experiment")
        database_manager.set_experiment('database_plots_test_case_1')

        self.write_to_database(database_manager)
        print("Here, closing")
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
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.set_experiment('database_plots_test_case_1')

        print("Writing to file")
        self.write_to_database(database_manager)
        database_manager.close()

        print("Reading back from file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='sample.db')
        self.read_from_database(database_reading_manager)

        if os.path.exists('sample.db'):
            os.unlink('sample.db')

    def test_write_to_network_and_read_from_file(self):
        print("Writing to server")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.set_experiment('database_plots_test_case_1')

        self.write_to_database(database_manager)
        database_manager.close()

        download_experiment_to_file(experiment_name='database_plots_test_case_1', file_path='output/downloaded_database.db')

        print("Reading back from downloaded file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='output/downloaded_database.db')
        self.read_from_database(database_reading_manager)

        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.close()



    def test_write_to_network_and_file_and_read_from_both(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')


        print("Writing to server")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, file='sample.db', cache_size=40)
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.set_experiment('database_plots_test_case_1')

        self.write_to_database(database_manager)
        database_manager.close()

        download_experiment_to_file(experiment_name='database_plots_test_case_1', file_path='output/downloaded_database.db')

        print("Reading back from downloaded file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='output/downloaded_database.db')
        self.read_from_database(database_reading_manager)

        print("Reading back from cache file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='sample.db')
        self.read_from_database(database_reading_manager)


        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.delete_experiment('database_plots_test_case_1')
        database_manager.close()




if __name__ == '__main__':
    unittest.main()