#!/usr/bin/env python3
import gzip

import pickle
import sys

import hplots.hgcal_analysis_plotter as hp

with gzip.open(sys.argv[1], 'rb') as f:
    read_data = pickle.load(f)



type = 'hgcal'
if len(sys.argv) == 4:
    type = sys.argv[3]

if type == 'hgcal':
    plotter = hp.HGCalAnalysisPlotter()
elif type =='trackml':
    raise NotImplementedError('ERROR')
    # plotter = tp.TrackMLPlotter()
else:
    raise NotImplementedError("Error")

pdfpath = sys.argv[2]
plotter.set_data(read_data['showers_dataframe'], read_data['events_dataframe'], '', pdfpath, scalar_variables=read_data['scalar_variables'])
plotter.process()
