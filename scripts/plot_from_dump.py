#!/usr/bin/env python3
from experiment_database_manager import ExperimentDatabaseManager
import gzip

import numpy as np
import pickle
import sys

import sql_credentials
import hplots.hgcal_analysis_plotter as hp
import hplots.trackml_plotter as tp

with gzip.open(sys.argv[1], 'rb') as f:
    graphs, metadata = pickle.load(f)


type = 'hgcal'
if len(sys.argv) == 4:
    type = sys.argv[3]

if type == 'hgcal':
    plotter = hp.HGCalAnalysisPlotter()
elif type =='trackml':
    plotter = tp.TrackMLPlotter()
else:
    raise NotImplementedError("Error")

pdfpath = sys.argv[2]
plotter.add_data_from_analysed_graph_list(graphs, metadata)
plotter.write_to_pdf(pdfpath)
