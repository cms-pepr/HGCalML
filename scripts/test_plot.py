#!/bin/env python

#all the reading stuff
from root_numpy import tree2array, root2array
from plotting_tools import plotter_3d
import numpy as np
## get the data


infile = "/afs/cern.ch/work/j/jkiesele/HGCal/data_test/hgc_tuple/hgcal_example.root"

usebranches = ['rechit_x','rechit_y','rechit_z','rechit_energy']

event=0

def read_vector(Tuple,branch):
    return np.array(Tuple[branch])



Tuple = root2array(infile, 
                treename = 'ana/hgc', 
                branches = usebranches)


rechit_x = read_vector(Tuple,'rechit_x')[event]
rechit_y = read_vector(Tuple,'rechit_y')[event]
rechit_z = read_vector(Tuple,'rechit_z')[event]
rechit_e = read_vector(Tuple,'rechit_energy')[event]


rechit_x = rechit_x[rechit_z>0]
rechit_y = rechit_y[rechit_z>0]
rechit_e = rechit_e[rechit_z>0]
rechit_z = rechit_z[rechit_z>0]

rechit_x = rechit_x[rechit_e>5e-2]
rechit_y = rechit_y[rechit_e>5e-2]
rechit_z = rechit_z[rechit_e>5e-2]
rechit_e = rechit_e[rechit_e>5e-2]

pl = plotter_3d("outfile.pdf", interactive=True)
pl.set_data(rechit_x,rechit_y,rechit_z,rechit_e)#np.logical_and(rechit_z<0, rechit_e>1e-6))

pl.save_binary("testfile.bin")

pl2 = plotter_3d()
pl2.load_binary("testfile.bin")
pl2.plot3d()





