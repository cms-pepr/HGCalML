#!/bin/env python

#all the reading stuff
from root_numpy import tree2array, root2array
from plotting_tools import plotter_3d
import numpy as np
## get the data


infile = "/afs/cern.ch/work/j/jkiesele/HGCal/data_test/hgc_tuple/ntup_0_n20.root"

usebranches = ['rechit_x','rechit_y','rechit_z','rechit_energy',
               'simcluster_eta','simcluster_phi',
               'rechit_eta','rechit_phi']


def read_vector(Tuple,branch):
    return np.array(Tuple[branch])



Tuple = root2array(infile, 
                treename = 'ana/hgc', 
                branches = usebranches)

for event in range(4,20):
    rechit_x = read_vector(Tuple,'rechit_x')[event]
    rechit_y = read_vector(Tuple,'rechit_y')[event]
    rechit_z = read_vector(Tuple,'rechit_z')[event]
    rechit_e = read_vector(Tuple,'rechit_energy')[event]
    
    
    rechit_eta = read_vector(Tuple,'rechit_eta')[event]
    rechit_phi = read_vector(Tuple,'rechit_phi')[event]
    simcluster_eta = read_vector(Tuple,'simcluster_eta')[event]
    simcluster_phi = read_vector(Tuple,'simcluster_phi')[event]
    
    average_eta = np.mean(simcluster_eta)
    average_phi = np.mean(simcluster_phi)
    
    selection_eta = np.abs(rechit_eta - average_eta) < 0.5
    selection_phi = np.logical_or(np.abs(rechit_phi - average_phi) < 0.5, 
                                  np.abs(rechit_phi - average_phi -2.*3.1415) < 0.5)
    
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
    
    pl = plotter_3d("outfile.pdf", interactive=True)
    pl.set_data(rechit_x,rechit_y,rechit_z,rechit_e)#np.logical_and(rechit_z<0, rechit_e>1e-6))
    pl.plot3d()
    pl.save_binary("testfile_"+ str(event) +".bin")


