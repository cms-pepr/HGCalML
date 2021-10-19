#!/usr/bin/env python3

from DeepJetCore import DataCollection
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Process
import random
from datastructures import TrainData_NanoML
import plotly.express as px
import pandas as pd
import tqdm
from DeepJetCore.dataPipeline import TrainDataGenerator


parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
args = parser.parse_args()

outdir = args.outputDir+'/'
### rewrite!
os.system('mkdir -p '+outdir)
#read a file
def invokeGen(infile):
    if infile[-6:] == '.djcdc':
        dc = DataCollection(infile)
        td = dc.dataclass()
        dc.setBatchSize(1)
        gen = dc.invokeGenerator()
    elif infile[-6:] == '.djctd':
        td = TrainData_NanoML()
        td.readFromFile(infile)
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setBuffer(td)
    elif infile[-5:] == '.root':
        td = TrainData_NanoML()
        td.convertFromSourceFile(infile,{},True)
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setBuffer(td)
        
    gen.setSkipTooLargeBatches(False)
    nevents = gen.getNBatches()
    return gen.feedNumpyData,nevents,td

gen,nevents,td = invokeGen(args.inputFile)

def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx"):
    ta = df[qualifier]
    unta = np.unique(ta)
    unta = unta[unta>-0.1]
    np.random.shuffle(unta)
    out = ta.copy()
    dfo = df.copy()
    for i in range(len(unta)):
        out[ta ==unta[i]]=i
    dfo[qualifier] = out
    return dfo
    
def toDataFrame(thegen, thetd):
    
    def popRSAndSqueeze(df):
        for k in df.keys():
            if "_rowsplits" in k:
                df.pop(k)
            else:
                df[k] = np.squeeze(df[k])
        return df
    
    data,_ = next(thegen())#this is a dict, row splits can be ignored, this is per event
    df = thetd.createFeatureDict(data,False)
    dftruth = thetd.createTruthDict(data)
    df = popRSAndSqueeze(df)
    dftruth = popRSAndSqueeze(dftruth)
    df['recHitLogEnergy'] = np.log(df['recHitEnergy']+1)
    dffeat = pd.DataFrame.from_dict(df)
    dftruth = pd.DataFrame.from_dict(dftruth)
    df.update(dftruth)
    dfall = pd.DataFrame.from_dict(df)
    return dffeat, dftruth, dfall

def compressShowerFeatures(df):
    dfout = df.drop_duplicates(subset = ["truthHitAssignementIdx"])
    return dfout[dfout["truthHitAssignementIdx"]>=0]



hitdf = pd.DataFrame()
showerdf = pd.DataFrame()
eventdf = pd.DataFrame()

print(nevents,'events')
#3D plots
for i in tqdm.tqdm(range(nevents)):
    
    dffeat,dftruth,df  = toDataFrame(gen,td)
    
    dfshowers = compressShowerFeatures(dftruth)
    showerhits = dffeat[dftruth["truthHitAssignementIdx"]>=0]
    #depvstruthenergy.append(np.sum(showerhits['recHitEnergy'])/(np.sum(dfshowers['truthHitAssignedEnergies'])+1.))
    
    #plot the first 20 as 3D plots
    if i < 20:
        #makes a copy
        df3d = shuffle_truth_colors(df)
        df3d['invTruthHitSpectatorFlag'] = np.where(df3d['truthHitAssignementIdx']<0,0.,
                                    np.ones_like(df3d['truthHitSpectatorFlag'])) * 1./(df3d['truthHitSpectatorFlag']+1e-1)
        
        hover_data=['truthHitAssignedEnergies',
                    'truthHitAssignedT',
                    'truthHitAssignedX',
                    'truthHitAssignedY',
                    'truthHitAssignementIdx',
                    'truthHitAssignedPIDs',
                    'truthHitSpectatorFlag']
        
        fig = px.scatter_3d(df3d, x="recHitX", y="recHitZ", z="recHitY", 
                                    color="truthHitAssignementIdx", size="recHitLogEnergy",
                                    symbol = "recHitID",
                                    hover_data=hover_data,
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = outdir + str(i) + "_event.html"
        fig.write_html(ccfile)
        
        fig = px.scatter_3d(df3d, x="recHitX", y="recHitZ", z="recHitY", 
                                    color="truthHitAssignementIdx", size="invTruthHitSpectatorFlag",
                                    hover_data=hover_data,
                                    symbol = "recHitID",
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = outdir + str(i) + "_spect.html"
        fig.write_html(ccfile)
    

#plt.hist(depvstruthenergy)
#plt.savefig(outdir +'depvstruth.pdf')

