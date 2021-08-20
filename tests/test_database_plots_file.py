import os

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.general_2d_plot import General2dBinningPlot
import matplotlib.pyplot as plt
import numpy as np
import sql_credentials
import unittest



class DatabasePlotsTestCases(unittest.TestCase):
    def write_to_database(self):
        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.set_experiment('database_plots_test_case_1')

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
        database_manager.close()

    def read_from_database(self):
        database_reading_manager = ExperimentDatabaseReadingManager(file=os.path.join('sample.db'))
        efficiency_plot = General2dBinningPlot(bins=np.array([0, 1, 2, 3, 4]), histogram_log=False,
                                               histogram_fraction=False)
        efficiency_plot.read_from_database(database_reading_manager, 'database_plots_test_case_1')
        efficiency_plot.draw()

        plt.savefig('output/test-reproduced-plot.png')

    def test_read_write(self):
        print("Writing")
        if os.path.exists('sample.db'):
            os.unlink('sample.db')
        self.write_to_database()
        #
        print("Reading back")
        self.read_from_database()
        #
        # database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        # database_manager.delete_experiment('database_plots_test_case_1')
        # database_manager.close()



if __name__ == '__main__':
    unittest.main()