import os

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.general_2d_plot import General2dBinningPlot
import matplotlib.pyplot as plt
import numpy as np
import sql_credentials
import unittest




class DatabasePlotsTestCases(unittest.TestCase):
    def test_read_write_1(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.delete_experiment('writing_numerical_data_test_case_1')
        database_manager.set_experiment('writing_numerical_data_test_case_1')

        inserted_data = dict()
        inserted_data['var_1'] = 19
        inserted_data['var_2'] = 109
        inserted_data['var_3'] = 54.1

        database_manager.insert_experiment_data('writing_numerical_data_test_case_1', inserted_data)
        database_manager.flush()

        database_manager_2 = ExperimentDatabaseReadingManager(file='sample.db')
        read_back_data = database_manager_2.get_data('writing_numerical_data_test_case_1','writing_numerical_data_test_case_1')

        assert read_back_data['var_1'][0] == inserted_data['var_1'] # Always returns list
        assert read_back_data['var_2'][0] == inserted_data['var_2'] # Always returns list
        assert read_back_data['var_3'][0] == inserted_data['var_3'] # Always returns list

        # Doing it at the end to check if it flushing works properly
        database_manager.delete_experiment('writing_numerical_data_test_case_1')
        database_manager.close()


    def test_read_write_2(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.delete_experiment('writing_numerical_data_test_case_2')
        database_manager.set_experiment('writing_numerical_data_test_case_2')

        inserted_data = dict()
        inserted_data['var_1'] = [19,20]
        inserted_data['var_2'] = [109,110]
        inserted_data['var_3'] = [54.1,43]

        database_manager.insert_experiment_data('writing_numerical_data_test_case_2', inserted_data)
        database_manager.flush()

        database_manager_2 = ExperimentDatabaseReadingManager(file='sample.db')
        read_back_data = database_manager_2.get_data('writing_numerical_data_test_case_2','writing_numerical_data_test_case_2')

        assert read_back_data['var_1'][0] == inserted_data['var_1'][0]
        assert read_back_data['var_1'][1] == inserted_data['var_1'][1]

        assert read_back_data['var_2'][0] == inserted_data['var_2'][0]
        assert read_back_data['var_2'][1] == inserted_data['var_2'][1]

        assert read_back_data['var_3'][0] == inserted_data['var_3'][0]
        assert read_back_data['var_3'][1] == inserted_data['var_3'][1]

        # Doing it at the end to check if it flushing works properly
        database_manager.delete_experiment('writing_numerical_data_test_case_2')
        database_manager.close()

    def test_read_write_3(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.delete_experiment('writing_numerical_data_test_case_3')
        database_manager.set_experiment('writing_numerical_data_test_case_3')

        inserted_data = dict()
        inserted_data['var_1'] = np.array([19,20])
        inserted_data['var_2'] = np.array([109,110])
        inserted_data['var_3'] = np.array([54.1,43])
        inserted_data['var_4'] = np.array(['hello','world'])

        database_manager.insert_experiment_data('writing_numerical_data_test_case_3', inserted_data)
        database_manager.flush()

        database_manager_2 = ExperimentDatabaseReadingManager(file='sample.db')
        read_back_data = database_manager_2.get_data('writing_numerical_data_test_case_3','writing_numerical_data_test_case_3')

        assert read_back_data['var_1'][0] == inserted_data['var_1'][0]
        assert read_back_data['var_1'][1] == inserted_data['var_1'][1]

        assert read_back_data['var_2'][0] == inserted_data['var_2'][0]
        assert read_back_data['var_2'][1] == inserted_data['var_2'][1]

        assert read_back_data['var_3'][0] == inserted_data['var_3'][0]
        assert read_back_data['var_3'][1] == inserted_data['var_3'][1]

        assert read_back_data['var_4'][0] == inserted_data['var_4'][0]
        assert read_back_data['var_4'][1] == inserted_data['var_4'][1]

        # Doing it at the end to check if it flushing works properly
        # database_manager.delete_experiment('writing_numerical_data_test_case_3')
        database_manager.close()

    def test_read_write_4(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.delete_experiment('writing_numerical_data_test_case_4')
        database_manager.set_experiment('writing_numerical_data_test_case_4')

        inserted_data = dict()
        inserted_data['var_1'] = 19
        inserted_data['var_2'] = 109
        inserted_data['var_3'] = np.nan

        database_manager.insert_experiment_data('writing_numerical_data_test_case_4', inserted_data)
        database_manager.flush()

        database_manager_2 = ExperimentDatabaseReadingManager(file='sample.db')
        read_back_data = database_manager_2.get_data('writing_numerical_data_test_case_4','writing_numerical_data_test_case_4')

        assert read_back_data['var_1'][0] == inserted_data['var_1'] # Always returns list
        assert read_back_data['var_2'][0] == inserted_data['var_2'] # Always returns list
        assert read_back_data['var_3'][0] == 0 # Always returns list

        # Doing it at the end to check if it flushing works properly
        database_manager.delete_experiment('writing_numerical_data_test_case_4')
        database_manager.close()



    def test_read_write_5(self):
        if os.path.exists('sample.db'):
            os.unlink('sample.db')

        database_manager = ExperimentDatabaseManager(file='sample.db', cache_size=40)
        database_manager.delete_experiment('writing_numerical_data_test_case_5')
        database_manager.set_experiment('writing_numerical_data_test_case_5')

        inserted_data = dict()
        inserted_data['var_1'] = 19
        inserted_data['var_2'] = 109
        inserted_data['var_3'] = np.nan

        database_manager.insert_experiment_data('writing_numerical_data_test_case_5', inserted_data)
        inserted_data = inserted_data.copy()
        inserted_data['var_4'] = 98.1
        database_manager.insert_experiment_data('writing_numerical_data_test_case_5_2', inserted_data)

        database_manager.flush()

        database_manager_2 = ExperimentDatabaseReadingManager(file='sample.db')
        read_back_data = database_manager_2.get_data('writing_numerical_data_test_case_5','writing_numerical_data_test_case_5')

        assert read_back_data['var_1'][0] == inserted_data['var_1'] # Always returns list
        assert read_back_data['var_2'][0] == inserted_data['var_2'] # Always returns list
        assert read_back_data['var_3'][0] == 0 # Always returns list

        # Doing it at the end to check if it flushing works properly
        database_manager.delete_experiment('writing_numerical_data_test_case_5')
        database_manager.close()




if __name__ == '__main__':
    unittest.main()