#!/usr/bin/env python3

import sql_credentials
import argparse
import experiment_database_manager

parser = argparse.ArgumentParser(
    'Deletes an experiment from the database')
parser.add_argument('experiment_name',
                    help='Experiment name on the server (normally exists in model_train_output/unique_id.txt')


args = parser.parse_args()

database_manager = experiment_database_manager.ExperimentDatabaseManager(sql_credentials.credentials)
database_manager.delete_experiment(args.experiment_name)
database_manager.close()