#!/usr/bin/env python3
import sql_credentials
import experiment_database_reading_manager
import argparse
from training_metrics_plots import TrainingMetricPlots

parser = argparse.ArgumentParser(
    'Produce running metrics plot (loss, efficiency and more)')
parser.add_argument('experiment_name',
                    help='Experiment name on the server (normally exists in model_train_output/unique_id.txt) or database file path (check is_database_file argument)')
parser.add_argument('--is_database_file', help='If you want to plot from a database file instead of server \nThe database file is normally located in model_train_output/training_metrics.db', action='store_true')
parser.add_argument('--plot_all',
                    help='Plot only a few metrics for faster performance or all of them?',action='store_true')
parser.add_argument('--running_average',
                    help='N running average elements, 1 means no running average', default='10')
parser.add_argument('output',
                    help='HTML file where to produce output')
parser.add_argument('--ignore_cache',
                    help='''Normally this script caches data so it doesn't have to pull everything
                    again and again but this option will ignore the cache''',action='store_true')
parser.add_argument('--trackml',
                    help='''For trackml problem, will plot different metrics''',action='store_true')


args = parser.parse_args()

if not args.is_database_file:
    print("Gonna get data from the server, using experiment_name %s" % args.experiment_name)
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
else:
    manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(file=args.experiment_name)


if args.trackml:
    plotter = TrainingMetricPlots(manager, args.experiment_name, ignore_cache=args.ignore_cache, cache_path='training_metrics_plotter_trackml.cache',
                                  metrics=['beta_threshold', 'distance_threshold', 'loss', 'trackml_score', 'num_truth_particles', 'num_reco_tracks'],
                                  titles=['Beta threshold', 'Distance threshold', 'loss', 'trackml score', 'Num truth particles', 'Num reco tracks'],
                                  database_table_name='training_performance_metrics_trackml'
                                  )
else:
    plotter = TrainingMetricPlots(manager, args.experiment_name, ignore_cache=args.ignore_cache)

plotter.do_plot_to_html(args.output, average_over=int(args.running_average))