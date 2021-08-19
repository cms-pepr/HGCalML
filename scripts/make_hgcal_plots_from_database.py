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


args = parser.parse_args()


plotter = HGCalAnalysisPlotter()
reading_manager = ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
plotter.add_data_from_database(reading_manager, table_prefix=args.table_prefix)
# plotter.add_data_from_database(reading_manager, table_prefix='alpha_plots_a2')
plotter.write_to_pdf(args.output, formatter=lambda x: '$\\beta=%.2f$'%(x['beta_threshold']))