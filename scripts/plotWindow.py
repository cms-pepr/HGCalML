#!/usr/bin/env python3

from DeepJetCore import DataCollection
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Process
import random
from datastructures import TrainData_NanoML, TrainData_NanoMLPF
import plotly.express as px
import pandas as pd
import tqdm
from DeepJetCore.dataPipeline import TrainDataGenerator


parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--hipsearch',action='store_true')
parser.add_argument('--plots',action='store_true')
parser.add_argument('--pf',action='store_true')
args = parser.parse_args()

outdir = args.outputDir+'/'
### rewrite!
os.system('mkdir -p '+outdir)

td_class = TrainData_NanoML
if args.pf:
    td_class = TrainData_NanoMLPF

#read a file
def invokeGen(infile):
    if infile[-6:] == '.djcdc':
        dc = DataCollection(infile)
        td = dc.dataclass()
        tdclass = dc.dataclass
        dc.setBatchSize(1)
        gen = dc.invokeGenerator()
    elif infile[-6:] == '.djctd':
        td = td_class()
        tdclass = td_class
        td.readFromFile(infile)
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setBuffer(td)
    elif infile[-5:] == '.root':
        print('reading from root file, converting...')
        td = td_class()
        tdclass = td_class
        td.readFromSourceFile(infile,{},True)
        td.writeToFile(infile+'.djctd')
        td.readFromFile(infile+'.djctd')
        print('conversion done')
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setBuffer(td)
        
    gen.setSkipTooLargeBatches(False)
    nevents = gen.getNBatches()
    gen.cast_to = tdclass
    return gen.feedTrainData,nevents,td

gen,nevents,td = invokeGen(args.inputFile)

def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx"):
    ta = df[qualifier]
    unta = np.unique(ta)
    unta = unta[unta>-0.1]
    np.random.seed(42)
    np.random.shuffle(unta)
    out = ta.copy()
    dfo = df.copy()
    for i in range(len(unta)):
        out[ta ==unta[i]]=i
    dfo[qualifier] = out
    return dfo
    
def toDataFrame(thegen, thetd):
    data = next(thegen())#this is a dict, row splits can be ignored, this is per event
    return data.createPandasDataFrame(0)
    

def compressShowerFeatures(df):
    dfout = df.drop_duplicates(subset = ["truthHitAssignementIdx"])
    return dfout[dfout["truthHitAssignementIdx"]>=0]

def quickplotshower(df,out):
    fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", 
                                color="hitratio", size="recHitLogEnergy",
                                symbol = "marker",
                                hover_data=['rel_std','totruthHitAssignedEnergies_ratio','marker','hitratio','nhits','corratio'],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html(out)
        
def hipsearch(df3d, i, outdir, makeplots=False):
    truthHitAssignementIdx = df3d['truthHitAssignementIdx']
    utidx = np.unique(truthHitAssignementIdx)
    counter=0
    Er_dep=[]
    Er_corr_dep=[]
    E =[]
    for t in utidx:
        if t < 0:
            continue
        seldf = df3d[df3d['truthHitAssignementIdx']==t]
        
        depsum = np.ones_like(seldf['recHitEnergy'])*np.sum(seldf['recHitEnergy'])
        nhits = float(len(seldf['recHitEnergy']))
        seldf['energy_ratio'] = seldf['recHitEnergy']/seldf['truthHitAssignedEnergies']
        
        seldf['totruthHitAssignedEnergies_ratio'] = depsum/seldf['truthHitAssignedEnergies']
        E.append(np.mean(seldf['truthHitAssignedEnergies']))
        
        Er_dep.append(np.mean(seldf['totruthHitAssignedEnergies_ratio']))
        
        hitratio = seldf['recHitEnergy']/depsum
        seldf['nhits'] = nhits
        hitratio *= nhits #1. on average for uniform etc.
        seldf['hitratio'] = hitratio
        
        m = np.mean(seldf['hitratio'])
        s = np.std(seldf['hitratio']-m)
        
        seldf['rel_std']= (hitratio-m)/s
        seldf['marker'] = np.array(seldf['rel_std'] > 5.,dtype='int32')
        ewithout = np.sum((1.-seldf['marker'])*seldf['recHitEnergy'])
        seldf['corratio'] = ewithout/seldf['truthHitAssignedEnergies']
        
        Er_corr_dep.append(np.mean(seldf['corratio']))
        if makeplots and np.all(depsum < seldf['truthHitAssignedEnergies']*1.1):
            quickplotshower(seldf,outdir+str(i)+'_'+str(counter)+'.html')
        counter+=1
        
    return Er_dep,  Er_corr_dep , E  
    


hitdf = pd.DataFrame()
showerdf = pd.DataFrame()
eventdf = pd.DataFrame()

print(nevents,'events')
#3D plots
Er_dep,  Er_corr_dep, E   = [], [], []
for i in tqdm.tqdm(range(nevents)):
    
    df  = toDataFrame(gen,td)
    
    #print(df.columns)
    
    dfshowers = compressShowerFeatures(df)
    showerhits = df[df["truthHitAssignementIdx"]>=0]
    #depvstruthenergy.append(np.sum(showerhits['recHitEnergy'])/(np.sum(dfshowers['truthHitAssignedEnergies'])+1.))
    
    from globals import pu
    #df["recHitLogEnergy"]*= (1. - (1.-1e-2)*(df["truthHitAssignementIdx"]>=pu.t_idx_offset))
    df3d = shuffle_truth_colors(df)
    df3d['orig_truthHitAssignementIdx']=df['truthHitAssignementIdx']
    df3d['t_inv_spec'] = np.where(df3d['truthHitAssignementIdx']<0,
                     0.,np.ones_like(df3d['truthHitSpectatorFlag'])) * 1./(df3d['truthHitSpectatorFlag']+1e-1)
    #plot the first 20 as 3D plots
    if i < 20 and args.plots:
        #makes a copy
        
        hover_data=['recHitEnergy',
                    'recHitHitR',
                    'truthHitAssignedEnergies',
                    'truthHitAssignedT',
                    'truthHitAssignedX',
                    'truthHitAssignedY',
                    'truthHitAssignedZ',
                    'truthHitAssignementIdx',
                    'orig_truthHitAssignementIdx',
                    'truthHitAssignedPIDs',
                    'truthHitSpectatorFlag']
        
        print('N hits', len(df3d))
        
        fig = px.scatter_3d(df3d, x="recHitX", y="recHitZ", z="recHitY", 
                                    color="truthHitAssignementIdx", size="recHitLogEnergy",
                                    symbol = "recHitID",
                                    hover_data=hover_data,
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = outdir + str(i) + "_event.html"
        fig.write_html(ccfile)
        
        continue
        
        fig = px.scatter_3d(df3d, x="recHitX", y="recHitZ", z="recHitY", 
                                    color="truthHitAssignementIdx", size="recHitHitR",
                                    symbol = "recHitID",
                                    hover_data=hover_data,
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = outdir + str(i) + "_event_hitsize.html"
        fig.write_html(ccfile)
        
        fig = px.scatter_3d(df3d, x="recHitX", y="recHitZ", z="recHitY", 
                                    color="truthHitAssignementIdx", size="t_inv_spec",
                                    hover_data=hover_data,
                                    symbol = "recHitID",
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = outdir + str(i) + "_spect.html"
        fig.write_html(ccfile)
        
    if args.hipsearch:
        iEr_dep,  iEr_corr_dep, iE = hipsearch(df3d, i, outdir, args.plots)
        Er_dep+=iEr_dep
        Er_corr_dep+=iEr_corr_dep
        E+=iE
    
plt.hist(Er_dep,bins=31,label='uncorr',alpha=0.5)
plt.hist(Er_corr_dep,bins=31,label='corr',alpha=0.5)
plt.legend()
plt.savefig(outdir +'hipcorr.pdf')


dfout = pd.DataFrame(zip(E,Er_dep,Er_corr_dep), columns = ['E','Er','Er_corr'])
dfout.to_pickle("df.pkl")