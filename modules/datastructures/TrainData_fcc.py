


from djcdata import TrainData
from djcdata.TrainData import fileTimeOut
from djcdata import SimpleArray
import numpy as np
import uproot3 as uproot
import awkward as ak1
from numba import jit
import gzip
import os
import pickle

#@jit(nopython=False)
def truth_loop(link_list :list, 
               t_dict:dict,
               part_p_list :list,
               ):
    
    nevts = len(link_list)
    for ie in range(nevts):#event
        nhits  = len(link_list[ie])
        for ih in range(nhits):
            idx = -1
            mom = 0.
            if link_list[ie][ih] >= 0:
                idx = link_list[ie][ih]
                mom = part_p_list[ie][idx]
                
            t_dict['t_idx'].append([idx])
            t_dict['t_energy'].append([mom])
            
            t_dict['t_pos'].append([0.,0.,0.])
            t_dict['t_time'].append([0.])
            t_dict['t_pid'].append([0.,0.,0.,0.,0.,0.])
            t_dict['t_spectator'].append([0.])
            t_dict['t_fully_contained'].append([1.])
            t_dict['t_rec_energy'].append([mom]) # THIS WILL NEED TO BE ADJUSTED
            t_dict['t_is_unique'].append([1]) #does not matter really
    
    
    return t_dict
    

class TrainData_fcc(TrainData):
   
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

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="events"):
        
        fileTimeOut(filename, 10)#wait 10 seconds for file in case there are hiccups
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
        hit_t = self.branchToFlatArray(tree["hit_t"])
        hit_e = self.branchToFlatArray(tree["hit_e"])
        hit_theta = self.branchToFlatArray(tree["hit_theta"])
        

        zerosf = 0.*hit_e
        
        print('hit_e',hit_e)
        hit_e = np.where(hit_e<0., 0., hit_e)
        
        
        farr = SimpleArray(np.concatenate([
            hit_e,
            zerosf,
            zerosf, #indicator if it is track or not
            zerosf,
            hit_theta,
            hit_x,
            hit_y,
            hit_z,
            zerosf,
            hit_t
            ], axis=-1), rs,name="recHitFeatures")
        
        
        
        # create truth
        hit_genlink = tree["hit_genlink0"].array()
        part_p = tree["part_p"].array()
        
        t = {
            't_idx' : [], #names are optional
            't_energy' :  [],
            't_pos' :  [], #three coordinates
            't_time' : []  ,
            't_pid' :  [] , #6 truth classes
            't_spectator' :  [],
            't_fully_contained' :  [],
            't_rec_energy' :  [],
            't_is_unique' :  []
            }
        
        #do this with numba
        t = truth_loop(hit_genlink.tolist(), 
                       t,
               part_p.tolist(),
               )
        
        for k in t.keys():
            if k == 't_idx' or k == 't_is_unique':
                t[k] = np.array(t[k], dtype='int32')
            else:
                t[k] = np.array(t[k], dtype='float32')
            t[k] = SimpleArray(t[k],  rs,name=k)
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ],[], []
        
        
    
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
        # print("hello", outfilename, inputfile)

        outdict = dict()
        outdict['predicted'] = predicted
        outdict['features'] = features
        outdict['truth'] = truth

        print("Writing to ", outfilename)
        with gzip.open(outfilename, "wb") as mypicklefile:
            pickle.dump(outdict, mypicklefile)
        print("Done")

    def writeOutPredictionDict(self, dumping_data, outfilename):
        '''
        this function should not be necessary... why break with DJC standards?
        '''
        if not str(outfilename).endswith('.bin.gz'):
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'

        with gzip.open(outfilename, 'wb') as f2:
            pickle.dump(dumping_data, f2)

    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)
        
    
    
