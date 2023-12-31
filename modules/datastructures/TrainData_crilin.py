



from DeepJetCore import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot

import os
import pickle
import gzip
import pandas as pd

from datastructures.TrainData_NanoML import TrainData_NanoML

class TrainData_crilin(TrainData_NanoML):
   
    def branchToFlatArray(self, b, return_row_splits=False, dtype='float32'):
        
        a = b.array()
        nevents = a.shape[0]
        rowsplits = [0]
        
        for i in range(nevents):
            rowsplits.append(rowsplits[-1] + a[i].shape[0])
        
        rowsplits = np.array(rowsplits, dtype='int64')
        
        if return_row_splits:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1),np.array(rowsplits, dtype='int64') 
        else:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1)

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="converted_photons"):
        
        #fileTimeOut(filename, 10)#wait 10 seconds for file in case there are hiccups
        tree = uproot.open(filename)[treename]
        
        '''
        
        hit_x, hit_y, hit_z: the spatial coordinates of the voxel centroids that registered the hit
        hit_dE: the energy registered in the voxel (signal + BIB noise)
        recHit_dE: the 'reconstructed' hit energy, i.e. the energy deposited by signal only
        evt_dE: the total energy deposited by the signal photon in the calorimeter
        evt_ID: an int label for each event -only for bookkeeping, should not be needed
        isSignal: a flag, -1 if only BIB noise, 0 if there is also signal hit deposition

        '''
        
        hit_x, rs = self.branchToFlatArray(tree["hit_x"], True)
        hit_y = self.branchToFlatArray(tree["hit_y"])
        hit_z = self.branchToFlatArray(tree["hit_z"])
        hit_dE = self.branchToFlatArray(tree["hit_dE"])
        
        zerosf = 0.*hit_dE
        
        print(hit_x.shape, rs)
        
        #truth
        evt_dE = self.branchToFlatArray(tree["evt_dE"])
        evt_trueE = evt_dE #this need to be updated
        isSignal = self.branchToFlatArray(tree["isSignal"], dtype='int32')
        
        zerosi = 0 * isSignal
        ### now we build the same structure as NanoML
        
        farr = SimpleArray(np.concatenate([
            hit_dE,
            zerosf,
            zerosf, #indicator if it is track or not
            zerosf,
            zerosf,
            hit_x,
            hit_y,
            hit_z,
            zerosf,
            zerosf
            ], axis=-1), rs,name="recHitFeatures")
        
        t = {
            't_idx' : SimpleArray(isSignal, rs), #names are optional
            't_energy' : SimpleArray(evt_trueE, rs),
            't_pos' : SimpleArray(np.concatenate(3*[zerosf],axis=-1), rs), #three coordinates
            't_time' : SimpleArray(isSignal, rs)  ,
            't_pid' : SimpleArray(np.concatenate( [1+zerosi]+5*[zerosi],axis=-1 ), rs) , #6 truth classes
            't_spectator' : SimpleArray(zerosf, rs),
            't_fully_contained' : SimpleArray(zerosf + 1., rs),
            't_rec_energy' : SimpleArray(evt_dE, rs),
            't_is_unique' : SimpleArray(zerosi, rs) 
            }
        
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ],[], []
        
        
        
        
        
        
        
    
    
    
 