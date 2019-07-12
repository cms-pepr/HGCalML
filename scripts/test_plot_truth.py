#!/bin/env python

#all the reading stuff
#import matplotlib
#matplotlib.use('Agg') 
from root_numpy import tree2array, root2array
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
import numpy as np
from DeepJetCore.preprocessing import readListArray
from argparse import ArgumentParser
import ROOT
import os

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--layerclusters', help='Plot layer clusters instead of rechits', action='store_true' , default=False )
args = parser.parse_args()


infile = args.inputFile
outdir=args.outputDir+"/"
plotRechits= not args.layerclusters

os.system('mkdir -p '+outdir)

features=None
truth=None

add_to_out="lc_"
if plotRechits:
    add_to_out="rh_"

rfile = ROOT.TFile(infile)
tree = rfile.Get("Delphes")
nentries=tree.GetEntries()
        

if plotRechits:
    print('reading features')
    
    features = readListArray(filename=infile, treename="Delphes", branchname="rechit_features", 
                             nevents=nentries, list_size=50000, n_feat_per_element=20,
                    zeropad=True)
    
    print('reading truth')
    
    truth = readListArray(filename=infile, treename="Delphes", branchname="rechit_simcluster_fractions", 
                             nevents=nentries, list_size=50000, n_feat_per_element=20,
                    zeropad=True)

else:
    print('reading features')
    
    features = readListArray(filename=infile, treename="Delphes", branchname="layercluster_features", 
                             nevents=nentries, list_size=10000, n_feat_per_element=30,
                    zeropad=True)
    
    print('reading truth')
    
    truth = readListArray(filename=infile, treename="Delphes", branchname="layercluster_simcluster_fractions", 
                             nevents=nentries, list_size=10000, n_feat_per_element=20,
                    zeropad=True)

#B x V x F

name = "brg"



print('plotting')

def remove_zero_energy(a, e):
    return a[e>0]

#clean up

print(truth.shape)

def plot_event(hgc_event):
    
    print(hgc_event)
    
    rechit_e = features[hgc_event,:, 0]
    rechit_x = features[hgc_event,:, 5]
    rechit_y = features[hgc_event,:, 6]
    rechit_z = features[hgc_event,:, 7]
    
    simcluster_truth = truth[hgc_event]
    
    simcluster_truth = remove_zero_energy(simcluster_truth,rechit_e)
    rechit_x = remove_zero_energy(rechit_x,rechit_e)
    rechit_y = remove_zero_energy(rechit_y,rechit_e)
    rechit_z = remove_zero_energy(rechit_z,rechit_e)
    rechit_e = remove_zero_energy(rechit_e,rechit_e)
    
    simcluster_truth = remove_zero_energy(simcluster_truth,rechit_z)
    rechit_x = remove_zero_energy(rechit_x,rechit_z)
    rechit_y = remove_zero_energy(rechit_y,rechit_z)
    rechit_e = remove_zero_energy(rechit_e,rechit_z)
    rechit_z = remove_zero_energy(rechit_z,rechit_z)
    
    print('n simclusters: ',truth.shape[1])
    
    
    pl = plotter_fraction_colors( output_file=outdir+str(hgc_event)+add_to_out+"simcl", interactive=False)
    pl.marker_scale=2.
    #pl.interactive=True
    pl.set_data(rechit_x,rechit_y,rechit_z,rechit_e, simcluster_truth)#np.logical_and(rechit_z<0, rechit_e>1e-6))
    pl.plot3d()
    pl.save_image()
    
    #var = raw_input('reshuffle colours and plot again? "yY/yes"\n')
    #if not(var=='y' or var =='Y' or var =='yes'):
    #hgc_event+=1
    #return
    

    mm = movie_maker(pl, output_file=outdir+add_to_out+"mm"+str(hgc_event), silent=True)
    mm.make_movie()
    pl.reset()
    
maxevents= nentries
if maxevents>30:
    maxevents=30
    
allevents = [i for i in range(maxevents)]

from multiprocessing import Pool
p = Pool()
print(p.map(plot_event, allevents))



