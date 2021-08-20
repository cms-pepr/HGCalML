#!/usr/bin/env python3

import sql_credentials
import experiment_database_reading_manager
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def running_mean(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


parser = argparse.ArgumentParser(
    'Produce running metrics plot (loss, efficiency and more)')
parser.add_argument('experiment_name',
                    help='Experiment name on the server (normally exists in model_train_output/unique_id.txt')
parser.add_argument('--is_database_file', help='If you want to plot from a database file instead of server \nThe database file is normally located in model_train_output/training_metrics.db', action='store_true')
parser.add_argument('--plot_all',
                    help='Plot only a few metrics for faster performance or all of them?',action='store_true')
parser.add_argument('--running_average',
                    help='N running average elements, 1 means no running average', default='10')
parser.add_argument('output',
                    help='PDF file')


args = parser.parse_args()


if not args.is_database_file:
    print("Gonna get data from the server, using experiment_name %s" % args.experiment_name)
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
    training_performance_metrics = manager.get_data('training_performance_metrics_extended', experiment_name=args.experiment_name)
else:
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(file=args.experiment_name)
    training_performance_metrics = manager.get_data('training_performance_metrics_extended')



average_over = int(args.running_average)

if average_over<=0:
    print("Error in running average, should be 1 or more (and less than number of iterations)")


def plot_metric(metric):
    global training_performance_metrics, average_over
    training_performance_metrics[metric] = [float(x) for x in training_performance_metrics[metric]]
    plt.figure()
    y_values = training_performance_metrics[metric]
    if average_over > len(y_values):
        print("Running over can't be greater than number of iterations...")
        raise RuntimeError("Running over can't be greater than number of iterations...")
    if average_over > 1:
        y_values = running_mean(y_values, average_over)
    plt.plot(training_performance_metrics['iteration'], y_values)
    plt.xlabel('iteration')
    plt.ylabel(metric)
    pdf.savefig()

pdf = PdfPages(args.output)

plot_metric('loss')
plot_metric('efficiency')
plot_metric('fake_rate')
plot_metric('sum_response')
plot_metric('response')

if bool(args.plot_all):
    plot_metric('f_score_energy')
    plot_metric('num_pred_showers')
    plot_metric('num_truth_showers')

pdf.close()
