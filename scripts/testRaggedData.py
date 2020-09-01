#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np


parser = ArgumentParser('')
parser.add_argument('inputFile', help="Input DataCollection file")
parser.add_argument('-n', help='Number of randomly picked events', default="10" )
parser.add_argument('-b', help='Batch size', default="8000" )
parser.add_argument('--min', help='Minumum number of batches', default="200" ) #min random batches
parser.add_argument('--max', help='Maximum number of batches', default="1200" ) #min random batches
args = parser.parse_args()

minbatch = int(args.min)
maxbatch = int(args.max)
n_plots = int(args.n)
infile = args.inputFile
batchsize = int(args.b)



from DeepJetCore.DataCollection import DataCollection
from index_dicts import create_truth_dict, create_feature_dict
from ragged_plotting_tools import make_original_truth_shower_plot, createRandomizedColors
import matplotlib
import matplotlib.pyplot as plt
import random

dc = DataCollection(infile)
dc.setBatchSize(batchsize)
gen = dc.invokeGenerator()
nbatches = gen.getNBatches()

if maxbatch >= nbatches:
    raise ValueError("maxbatch >= nbatches in sample")
if minbatch >= maxbatch:
    raise ValueError("minbatch >= maxbatch")

events = random.sample(range(minbatch,maxbatch), n_plots)
lastev = -1
n_plots_done=0
print('scanning...')
for i in range(nbatches):
    f,t = next(gen.feedNumpyData())
    rs = f[1]
    f = f[0]
    t = t[0]
    if i in events:
        print('plotting ',i)
        rs = np.array(rs[0:int(rs[-1])], dtype='int')
        for subbatch in [1,-1]:
            start = int(rs[subbatch-1])
            end   = int(rs[subbatch])
            
            feat = f[start:end]
            truth = t[start:end]
            
            feat = create_feature_dict(feat)
            truth = create_truth_dict(truth)
            
            t_idx = truth['truthHitAssignementIdx']
            t_idx_nospec = np.where(truth['truthIsSpectator']>0.1, -1, t_idx )
            #print(truth['truthIsSpectator'])
            
            for k in range(4):
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111, projection='3d')
                cmap = createRandomizedColors('jet',seed=k)
                make_original_truth_shower_plot(plt, ax,
                                            t_idx,                      
                                            feat['recHitEnergy'], 
                                            feat['recHitX'],
                                            feat['recHitY'],
                                            feat['recHitZ'],
                                            cmap=cmap)
                plt.tight_layout()
                fig.savefig("event_"+str(i)+"_batch_"+str(subbatch)+"_plotit_"+str(k)+".pdf")
                fig.clear()
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111, projection='3d')
                
                make_original_truth_shower_plot(plt, ax,
                                            t_idx_nospec,                      
                                            feat['recHitEnergy'], 
                                            feat['recHitX'],
                                            feat['recHitY'],
                                            feat['recHitZ'],
                                            cmap=cmap)
                plt.tight_layout()
                fig.savefig("event_"+str(i)+"_batch_"+str(subbatch)+"_plotit_"+str(k)+"_nospec.pdf")
                
                plt.close(fig)
                plt.clf()
                plt.cla()
                plt.close() 
            if len(rs) < 3:
                break
        print('scanning...')
        
        n_plots_done += 1
    

    if n_plots_done == n_plots:
        break
        break