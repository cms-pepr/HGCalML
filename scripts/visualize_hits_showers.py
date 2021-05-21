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
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm
import matplotlib._color_data as mcd
import random
random.seed(0)
colors_ = list(mcd.XKCD_COLORS.values())
random.shuffle(colors_)


def hitSize(energy):
    scale = 100/np.average(energy)
    maxsize = 3
    loge = np.log(energy*scale)
    return [max(0, min(x, maxsize)) for x in loge]

def mapColors(vals):
    return [mapColor(i) for i in vals]

def mapColor(i):
    i = int(i)
    if i < 0:
        return "#c8cbcc"
    cmap = matplotlib.cm.get_cmap('inferno') #inferno or Viridis
    all_colors = [matplotlib.colors.rgb2hex(cmap(c)) for c in range(cmap.N)]
    all_colors.extend(colors_)
    if i >= len(all_colors):
        i = np.random.randint(0, len(all_colors))
    # Avoid too "smooth" of a transition for close by values
    return all_colors[i]

parser = ArgumentParser('')
parser.add_argument('--inputFile')
parser.add_argument('--outputDir')
#parser.add_argument('--outName', default='recHits_3D.html',type=str)
args = parser.parse_args()


infile = args.inputFile
outdir=args.outputDir+"/"
events_max=1
os.system('mkdir -p '+outdir)
#outfile = args.outName
#if not outfile[-5:] == ".html":
#    outfile+=".html"
        
td=TrainData_NanoML()
td.readFromFile(infile)

for event_num in range(1,17,2): #looking at jsut one half of event
    df = (td.createPandasDataFrame(event_num)) #[:1000] just looking at some 1000 hits
    front_face_z = 323
    noise_filter = (df['truthHitAssignementIdx'] > -1)
    #hgcal_front_face_filter = (abs(df['truthHitAssignedZ']) < front_face_z) # < - on front, > not on front
    hgcal_front_face_filter = (df['truthHitFullyContainedFlag'] > 0) 
    filt = noise_filter & hgcal_front_face_filter
    df = df[filt]
    spectator_filter = (df['truthHitSpectatorFlag'] == 1) # 1 - spectator, 0 - normal
    df_spectators_only = df[filt & spectator_filter]
    showers_with_spectators = np.unique(df_spectators_only['truthHitAssignementIdx'])
    showers_spec_filt = np.isin(df['truthHitAssignementIdx'], showers_with_spectators)
    df_spec_filt = df[showers_spec_filt]

    fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", 
                        color="truthHitAssignementIdx", size="recHitLogEnergy",
                        template='plotly_white',
                        color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html(outdir+ 'recHits_3D_AllfrontFace_event%i.html'%event_num)


    fig = px.scatter_3d(df_spec_filt, x="recHitX", y="recHitZ", z="recHitY", 
                        color="truthHitAssignementIdx", size="recHitLogEnergy",
                        template='plotly_white',
                        color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html(outdir+ 'recHits_3D_AllfrontFace_spectatorShowers_event%i.html'%event_num)


    fig = px.scatter_3d(df_spectators_only, x="recHitX", y="recHitZ", z="recHitY", 
                        color="truthHitAssignementIdx", size="recHitLogEnergy",
                        template='plotly_white',
                        color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html(outdir+ 'recHits_3D_AllfrontFace_spectators_event%i.html'%event_num)



    fig2 = go.Figure([
        go.Scatter3d(
            name='Showers',
            x=df_spec_filt['recHitX'],
            y=df_spec_filt['recHitZ'],
            z=df_spec_filt['recHitY'],
            mode='markers',
            #marker=dict(color=mapColors(df_spec_filt['truthHitAssignementIdx'].to_numpy()),
            #            size=hitSize(df_spec_filt["recHitEnergy"]), line=dict(width=0)),
            marker=dict(color=df_spec_filt['truthHitAssignementIdx'],colorscale='Viridis',
                        size=hitSize(df_spec_filt["recHitEnergy"]), line=dict(width=0)),
            text=["Cluster Idx %i<br>RecHit Energy: %.4f<br>PDG Id: %i<br>" % (idx,e,pdgid)
                  for (idx,e,pdgid) in zip(df_spec_filt['truthHitAssignementIdx'],
                                           df_spec_filt['recHitEnergy'],df_spec_filt['truthHitAssignedPIDs'])],
            hovertemplate="x: %{x:0.2f}<br>y: %{z:0.2f}<br>z: %{y:0.2f}<br>%{text}<br>",
            showlegend=True
            ),
        go.Scatter3d(
            name='Spectator Hits',
            x=df_spectators_only['recHitX'],
            y=df_spectators_only['recHitZ'],
            z=df_spectators_only['recHitY'],
            mode='markers',
            # marker=dict(color=mapColors(df_spec_filt['truthHitAssignementIdx'].to_numpy()),
            marker=dict(color='red',
                        symbol='x',size=3,line=dict(width=3,
                                                    color=df_spectators_only['truthHitAssignementIdx'],colorscale='Viridis')),
            text=["Cluster Idx %i<br>RecHit Energy: %.4f<br>PDG Id: %i<br>" % (idx,e,pdgid)
                  for (idx,e,pdgid) in zip(df_spectators_only['truthHitAssignementIdx'],
                                           df_spectators_only['recHitEnergy'],df_spectators_only['truthHitAssignedPIDs'])],
            showlegend=True
            )
        ])
    fig2.write_html(outdir+ 'recHits_3D_AllfrontFaceWithSpectators_event%i.html'%event_num)

