#!/bin/env python

#all the reading stuff
from root_numpy import tree2array, root2array
from plotting_tools import plotter_3d, movie_maker
import numpy as np
## get the data


indir = "/eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/LocalRun/NtupTask/dev_CloseByParticleGun_fixedTimeOffset_DeltaR"
DR="0p2"
DRcut=0.4
infile = indir+DR+"_trialLocal/ntup_0_n50.root"

usebranches = ['rechit_x','rechit_y','rechit_z','rechit_energy',
               'simcluster_eta','simcluster_phi',
               'rechit_eta','rechit_phi']


def read_vector(Tuple,branch):
    return np.array(Tuple[branch])



Tuple = root2array(infile, 
                treename = 'ana/hgc', 
                branches = usebranches)

for hgc_event in range(0,50):
    rechit_x = read_vector(Tuple,'rechit_x')[hgc_event]
    rechit_y = read_vector(Tuple,'rechit_y')[hgc_event]
    rechit_z = read_vector(Tuple,'rechit_z')[hgc_event]
    rechit_e = read_vector(Tuple,'rechit_energy')[hgc_event]
    
    
    rechit_eta = read_vector(Tuple,'rechit_eta')[hgc_event]
    rechit_phi = read_vector(Tuple,'rechit_phi')[hgc_event]
    simcluster_eta = read_vector(Tuple,'simcluster_eta')[hgc_event]
    simcluster_phi = read_vector(Tuple,'simcluster_phi')[hgc_event]
    
    average_eta = np.mean(simcluster_eta)
    average_phi = np.mean(simcluster_phi)
    
    selection_eta = np.abs(rechit_eta - average_eta) < DRcut
    selection_phi = np.logical_or(np.abs(rechit_phi - average_phi) < DRcut,
                                  np.abs(rechit_phi - average_phi -2.*3.1415) < DRcut)
    
    selection = np.logical_and(selection_eta,selection_phi)
    selection = np.logical_and(selection,rechit_z>0)
    
    rechit_x = rechit_x[selection]
    rechit_y = rechit_y[selection]
    rechit_e = rechit_e[selection]
    rechit_z = rechit_z[selection]
    
    print('selected rechits ', rechit_x.shape, 'eta', average_eta)
    
    #rechit_x = rechit_x[rechit_e>1e-2]
    #rechit_y = rechit_y[rechit_e>1e-2]
    #rechit_z = rechit_z[rechit_e>1e-2]
    #rechit_e = rechit_e[rechit_e>1e-2]
    
    pl = plotter_3d(str(hgc_event)+".pdf", interactive=False)
    pl.set_data(rechit_x,rechit_y,rechit_z,rechit_e)#np.logical_and(rechit_z<0, rechit_e>1e-6))
    pl.plot3d()
    pl.save_binary("testfile_"+ str(hgc_event) +".bin")
    pl.save_image()
    
    mm = movie_maker(pl, outputDir="mm_"+str(hgc_event), silent=False)
    mm.make_movie()
    pl.reset()
    
    


