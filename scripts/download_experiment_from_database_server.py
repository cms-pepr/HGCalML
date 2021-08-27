#!/usr/bin/env python3

import sql_credentials
import argparse
import experiment_database_manager
from experiment_database_tools import download_experiment_to_file

parser = argparse.ArgumentParser(
    'Deletes an experiment from the database')
parser.add_argument('experiment_name',
                    help='Experiment name on the server (normally exists in model_train_output/unique_id.txt')
parser.add_argument('file',
                    help='The database file to which to save to. It will be sqlite format. You can open using sqlite db browser '
                         'or any other of your favorite tools. You can also read back data from it using this repository.')


args = parser.parse_args()

download_experiment_to_file(experiment_name=args.experiment_name, file_path=args.file)