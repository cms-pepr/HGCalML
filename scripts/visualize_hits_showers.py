#!/bin/env python

#all the reading stuff
#import matplotlib
#matplotlib.use('Agg') 
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
import numpy as np
from DeepJetCore.preprocessing import readListArray
from DeepJetCore.TrainData import TrainData
from datastructures import TrainData_NanoML
from argparse import ArgumentParser
import ROOT
import os
import plotly.express as px

parser = ArgumentParser('')
parser.add_argument('--inputFile')
parser.add_argument('--outputDir')
args = parser.parse_args()


infile = args.inputFile
outdir=args.outputDir+"/"
events_max=1
os.system('mkdir -p '+outdir)
        
td=TrainData_NanoML()
td.readFromFile(infile)
df = (td.createPandasDataFrame(1)) #[:1000] just looking at some 1000 hits
front_face_z = 323
noise_filter = (df['truthHitAssignementIdx'] > -1)
#hgcal_front_face_filter = (abs(df['truthHitAssignedZ']) < front_face_z) # < - on front, > not on front
hgcal_front_face_filter = (df['truthHitFullyContainedFlag'] > 0) 
spectator_filter = (df['truthHitSpectatorFlag'] < 1) # 1 - spectator, 0 - normal
filt = noise_filter & hgcal_front_face_filter & spectator_filter
df = df[filt]

fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", 
                    color="truthHitAssignementIdx", size="recHitLogEnergy",
                    template='plotly_white',
    color_continuous_scale='Inferno')
fig.update_traces(marker=dict(line=dict(width=0)))
fig.write_html(outdir+ "recHits_3D.html")

