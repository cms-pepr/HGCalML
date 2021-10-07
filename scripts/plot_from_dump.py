#!/usr/bin/env python3
from experiment_database_manager import ExperimentDatabaseManager
import gzip

import numpy as np
import pickle
import sys

from matplotlib.backends.backend_pdf import PdfPages

import sql_credentials
import hplots.hgcal_analysis_plotter as hp

with gzip.open(sys.argv[1], 'rb') as f:
    graphs, metadata = pickle.load(f)


plotter = hp.HGCalAnalysisPlotter()
plotter.add_data_from_analysed_graph_list(graphs, metadata)
plotter.write_to_pdf('/Users/shahrukhqasim/Downloads/tmp/sept19/y/result.pdf')

pdfpath = sys.argv[2]
pdf = PdfPages(pdfpath)
plotter = hp.HGCalAnalysisPlotter()
plotter.add_data_from_analysed_graph_list(graphs, metadata)
plotter.write_to_pdf(pdfpath)
