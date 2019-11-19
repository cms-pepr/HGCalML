#!/usr/bin/env python

import DeepJetCore
import os
from argparse import ArgumentParser
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
from DeepJetCore.TrainData import TrainData
import numpy as np
import matplotlib.pyplot as plt

def selectEvent(rs, feat, truth, event):
    rs = np.array(rs , dtype='int')
    rs = rs[:rs[-1]]
    print rs
    
    print(feat.shape)
    #get event
    feat = feat[rs[event]:rs[event+1],...]
    
    print(feat.shape)
    return feat, truth[rs[event]:rs[event+1],...]

def makePlot(pfeat, ptruth, outfile, movie=False):
    
    if e < 19: return
    
    fig = plt.figure(figsize=(10,4))
    ax = [fig.add_subplot(1,2,1, projection='3d'), fig.add_subplot(1,2,2, projection='3d')]
    

    rechit_e = pfeat[:,0]
    rechit_eta = pfeat[:,1]
    rechit_x = pfeat[:,5]
    rechit_y = pfeat[:,6]
    rechit_z = pfeat[:,7]
    
    truthasso = ptruth[:,1]
    not_assigned = ptruth[:,1] < 0
    print('event', e)
    print('not assigned ',np.count_nonzero(not_assigned, axis=-1))
    
    rechit_nonoise = np.array(rechit_e)
    rechit_nonoise[not_assigned] = 0
    totalenergy = np.sum(rechit_e)
    nonoise_energy = np.sum(rechit_nonoise)
    noisefraction = 1. - nonoise_energy/totalenergy
    print('totalenergy ',totalenergy)
    print('noisefraction ',noisefraction)
    print('noise energy ',totalenergy - nonoise_energy)
    print('mean eta', np.mean(rechit_eta))
    
    
    
    pl = plotter_3d(output_file=outdir+"/plot")
    
    pl.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_e , c =truthasso)
    pl.marker_scale=2.
    pl.plot3d(ax=ax[0])
    
    
    pl.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_nonoise , c =truthasso)
    pl.plot3d(ax=ax[1])
    
    fig.savefig(outfile+".pdf")
    plt.close()
    
    if movie:
        pl2 = plotter_3d(output_file=outdir+"/plot_noise")
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_e , c =truthasso)
    
        mm = movie_maker(pl2, output_file=outfile+"_mm_noise", silent=True)
        mm.make_movie()
        
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_nonoise , c =truthasso)
        mm2 = movie_maker(pl2, output_file=outfile+"_mm_nonoise", silent=True)
        mm2.make_movie()
    

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--movie', help='Also create a movie', action='store_true' , default=False )
parser.add_argument('-n', help='Event', default='0' )
args = parser.parse_args()


infile = args.inputFile
outdir=args.outputDir+"/"
nevents=int(args.n)
make_movie = args.movie
os.system('mkdir -p '+outdir)



td = TrainData()
td.readFromFile(infile)
#td.skim(event)
rs = td.x[0]
feat = td.x[1]
truth = td.y[0]

for e in range(nevents):
    pfeat, ptruth = selectEvent(rs,feat,truth,e)
    
    makePlot(pfeat, ptruth, outdir+str(e), make_movie)