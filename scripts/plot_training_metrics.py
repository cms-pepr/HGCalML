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
parser.add_argument('-experiment_name',
                    help='Experiment name',default='')
parser.add_argument('-database_file',
                    help='Database output file with training metrics (normally train_path/training_metrics.db',default='')
parser.add_argument('-plot_all',
                    help='Plot only a few metrics for faster performance or all of them?',default='False')
parser.add_argument('output',
                    help='PDF file')


args = parser.parse_args()


if len(args.experiment_name) != 0:
    print("Gonna get data from the server, using experiment_name %s" % args.experiment_name)
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
    training_performance_metrics = manager.get_data('training_performance_metrics_extended', experiment_name=args.experiment_name)
else:
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(file=args.database_file)
    training_performance_metrics = manager.get_data('training_performance_metrics_extended')



average_over = 10


def plot_metric(metric):
    global training_performance_metrics
    training_performance_metrics[metric] = [float(x) for x in training_performance_metrics[metric]]
    plt.figure()
    eff = training_performance_metrics[metric]
    eff = running_mean(eff, average_over)
    plt.plot(training_performance_metrics['iteration'], eff)
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
