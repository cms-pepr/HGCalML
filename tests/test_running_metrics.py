import os

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.general_2d_plot import General2dBinningPlot
import matplotlib.pyplot as plt
import numpy as np
import sql_credentials
import unittest
from experiment_database_tools import download_experiment_to_file
from training_metrics_plots import TrainingMetricPlots


class DatabasePlotsTestCases(unittest.TestCase):
    def write_to_database(self, database_manager):
        for i in range(100):
            dic = dict()
            dic['efficiency'] = min(1, max(0, np.random.normal(0,1)))
            dic['fake_rate'] = min(1, max(0, np.random.normal(0,1)))
            dic['response'] = np.random.normal(1,0.5)
            dic['sum_response'] = np.random.normal(1,0.5)


            dic['beta_threshold'] = 0.5
            dic['distance_threshold'] = 0.5
            dic['iteration'] = i
            dic['loss'] = np.random.normal(1,0.5)

            dic['f_score_energy'] = min(1, max(0, np.random.normal(0,1)))

            database_manager.insert_experiment_data('training_performance_metrics', dic)

            dic = dic.copy()
            dic['precision_without_energy'] = min(1, max(0, np.random.normal(0,1)))
            dic['recall_without_energy'] = min(1, max(0, np.random.normal(0,1)))
            dic['f_score_without_energy'] = min(1, max(0, np.random.normal(0,1)))
            dic['precision_energy'] = min(1, max(0, np.random.normal(0,1)))
            dic['recall_energy'] = min(1, max(0, np.random.normal(0,1)))
            dic['num_truth_showers'] = max(0, np.random.normal(100,10))
            dic['num_pred_showers'] = max(0, np.random.normal(100,10))

            database_manager.insert_experiment_data('training_performance_metrics_extended', dic)

    def read_from_database(self, database_manager):
        # Just to see if over 100 averaging works fine
        plotter = TrainingMetricPlots(database_manager, 'test_case_running_metrics_1', ignore_cache=True)
        plotter.do_plot_to_html('output/plots_not_av.html', average_over=int(1000))


        plotter = TrainingMetricPlots(database_manager, 'test_case_running_metrics_1', ignore_cache=True)
        plotter.do_plot_to_html('output/plots_av.html', average_over=int(10))

        # This experiment does not exist
        plotter = TrainingMetricPlots(database_manager, 'test_case_running_metrics_1_exp_does_not_exist', ignore_cache=True)
        test_case_passed = False
        try:
            plotter.do_plot_to_html('output/plots.html', average_over=int(1000))
        except TrainingMetricPlots.ExperimentNotFoundError as e:
            test_case_passed = True

        assert test_case_passed is True




    def test_insertion_and_reading(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        print("Writing to server")
        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, file='sample.db', cache_size=40)
        database_manager.delete_experiment('test_case_running_metrics_1')
        database_manager.delete_experiment('test_case_running_metrics_1_exp_does_not_exist')
        database_manager.set_experiment('test_case_running_metrics_1')

        self.write_to_database(database_manager)
        database_manager.close()

        download_experiment_to_file(experiment_name='test_case_running_metrics_1', file_path='output/downloaded_database.db')

        print("Reading back from downloaded file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='output/downloaded_database.db')
        self.read_from_database(database_reading_manager)

        print("Reading back from cache file")
        database_reading_manager = ExperimentDatabaseReadingManager(file='sample.db')
        self.read_from_database(database_reading_manager)


        database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, cache_size=40)
        database_manager.delete_experiment('test_case_running_metrics_1')
        database_manager.close()



if __name__ == '__main__':
    unittest.main()