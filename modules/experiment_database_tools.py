import os

from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from sql_credentials import credentials
from experiment_database_manager import ExperimentDatabaseManager



def download_experiment_to_file(experiment_name, file_path):
    if os.path.exists(file_path):
        os.unlink(file_path)

    file_database = ExperimentDatabaseManager(file=file_path, cache_size=100000)
    file_database.set_experiment(experiment_name)


    reading_manager = ExperimentDatabaseReadingManager(mysql_credentials=credentials)

    query = '''SELECT (table_name) FROM information_schema.columns WHERE column_name = 'experiment_name' AND table_schema = '%s';''' % credentials['database']
    data = reading_manager.get_data_from_query(query)
    tables = [x[0] for x in data]
    tables.remove('experiments')

    print("Gotten list of all tables:")
    print('\n'.join(tables))

    for table in tables:
        print("Downloading data from ", table)
        table_data = reading_manager.get_data(table, experiment_name)

        if table_data is None:
            print("No data found in table for this experiment, skipping")
            continue

        print("Gotten keys", table_data.keys())
        for key in table_data.keys():
            print('\t%s is %s' % (key, str(type(table_data[key][0]))))
        table_data.pop('experiment_name')

        file_database.insert_experiment_data(table, table_data)

    print("Finishing up writing to file...")
    file_database.close()

