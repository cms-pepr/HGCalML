#!/bin/env python

#all the reading stuff
#import matplotlib
#matplotlib.use('Agg') 
from root_numpy import tree2array, root2array
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
import numpy as np
from DeepJetCore.preprocessing import readListArray
## get the data


indir = "/eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/LocalRun/NtupTask/dev_CloseByParticleGun_fixedTimeOffset_DeltaR"
DR="0p3"
DRcut=0.4
infile = indir+DR+"_trialLocal/ntuple_converted_0_ntuple.root"

outdir="/eos/home-j/jkiesele/HGCal/HGCalML_data/"

print('reading features')

features = readListArray(filename=infile, treename="Delphes", branchname="rechit_features", 
                         nevents=50, list_size=50000, n_feat_per_element=10,
                zeropad=True)

print('reading truth')

truth = readListArray(filename=infile, treename="Delphes", branchname="simcluster_fractions", 
                         nevents=50, list_size=50000, n_feat_per_element=20,
                zeropad=True)

#B x V x F

name = "brg"



print('plotting')

def remove_zero_energy(a, e):
    return a[e>0]

def plot_event(hgc_event):
    
    print(hgc_event)
    
    rechit_e = features[hgc_event,:, 0]
    rechit_x = features[hgc_event,:, 5]
    rechit_y = features[hgc_event,:, 6]
    rechit_z = features[hgc_event,:, 7]

    simcluster_truth = remove_zero_energy(truth[hgc_event],rechit_e)

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
    
    
    pl = plotter_fraction_colors( output_file=outdir+str(hgc_event)+"_2sim.pdf", interactive=True)
    pl.marker_scale=2.
    pl.interactive=True
    pl.set_data(rechit_x,rechit_y,rechit_z,rechit_e, simcluster_truth)#np.logical_and(rechit_z<0, rechit_e>1e-6))
    pl.plot3d()
    #pl.save_image()
    
    #var = raw_input('reshuffle colours and plot again? "yY/yes"\n')
    #if not(var=='y' or var =='Y' or var =='yes'):
    #hgc_event+=1
    

    mm = movie_maker(pl, output_file=outdir+"mm3_sim_"+str(hgc_event), silent=False)
    mm.make_movie()
    pl.reset()
    
    
plot_event(0)
exit()
allevents = [i for i in range(1)]

from multiprocessing import Pool
p = Pool()
print(p.map(plot_event, allevents))



