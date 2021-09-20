import argparse

from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
import sql_credentials

parser = argparse.ArgumentParser(
    'Analyse predictions from object condensation and plot relevant results')
parser.add_argument('table_prefix',
                    help='Output directory with .bin.gz files or a txt file with full paths of the bin gz files')
parser.add_argument('output',
                    help='PDF file')
parser.add_argument('--condition_string', default='',
                    help='Condition sql string')


args = parser.parse_args()

condition_string = None
if len(args.condition_string)!=0:
    condition_string = args.condition_string



plotter = HGCalAnalysisPlotter(['settings', 'efficiency_fo_truth', 'fake_rate_fo_pred', 'response_fo_truth',
                                'response_fo_pred', 'response_sum_fo_truth', 'energy_resolution'])
reading_manager = ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
plotter.add_data_from_database(reading_manager, table_prefix=args.table_prefix, condition=condition_string)
# plotter.add_data_from_database(reading_manager, table_prefix='alpha_plots_a2')
# plotter.write_to_pdf(args.output, formatter=lambda x: 'Optimized f1 score\n$\\alpha$ param=$%.2f$ \n$\\beta$ param$=%.2f$ \n$\\beta=%.4f$\n$d=%.4f$\n'%(x['beta_param'],x['alpha_param'],x['beta_threshold'],x['distance_threshold']))


plotter.write_to_pdf(args.output, formatter=lambda x: 'IOU threshold %f'%x['iou_threshold'])
