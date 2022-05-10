



from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot

import os
import pickle
import gzip
import pandas as pd



class TrainData_ild(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        #don't use for now
        #self.input_names=["input_hits","input_row_splits"]


    
    ######### helper functions for ragged interface
    ##### might be moved to DJC soon?, for now lives here
    
    def createSelection(self, jaggedarr):
        # create/read a jagged array 
        # with selects for every event
        pass
    
    def branchToFlatArray(self, b, returnRowSplits=False, selectmask=None, is3d=None):
        
        a = b.array()
        nbatch = a.shape[0]
        
        if is3d:
            allba=[]
            for b in range(nbatch):
                ba = np.array(a[b])
                allba.append(ba)
            
            a = np.concatenate(allba,axis=0)
            print(a.shape)
            
        if selectmask is not None:
            if is3d:
                a = a[selectmask.flatten()]
            else:
                a = a[selectmask]
        #use select(flattened) to select
        contentarr=None
        if  is3d is  None:
            contentarr = a.content
            contentarr = np.expand_dims(contentarr, axis=1)
        else:
            contentarr=a
        
        if not returnRowSplits:
            return np.array(contentarr,dtype='float32')
        
        nevents = a.shape[0]
        rowsplits = [0]
        
        max_per_rs=0
        #not super fast but ok given there aren't many events per file
        for i in range(nevents):
            #make a[i] np array
            #get select[i] -> the truth mask
            #apply, then fill RS
            if selectmask is None:
                rowsplits.append(rowsplits[-1] + a[i].shape[0])
            else:
                select = selectmask[i]
                nonzero = np.count_nonzero(select)
                if nonzero > max_per_rs:
                    max_per_rs=nonzero
                rowsplits.append(rowsplits[-1] + nonzero)
                
        rowsplits = np.array(rowsplits, dtype='int64')
        print('mean hits per rs', contentarr.shape[0]/rowsplits.shape[0], ' max hits per rs: ',max_per_rs)
        return np.expand_dims(a.content, axis=1),np.array(rowsplits, dtype='int64') 

    def fileIsValid(self, filename):
        import ROOT
        try:
            fileTimeOut(filename, 2)
            tree = uproot.open(filename)["SLCIOConverted"]
            f=ROOT.TFile.Open(filename)
            t=f.Get("SLCIOConverted")
            if t.GetEntries() < 1:
                raise ValueError("")
        except Exception as e:
            print(e)
            return False
        return True

    
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="SLCIOConverted"):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)[treename]
        nevents = tree.numentries
        selection=None
        
        hit_energy , rs = self.branchToFlatArray(tree["energy"], True,selection)
        hit_x  = self.branchToFlatArray(tree["positionX"], False,selection)
        hit_y  = self.branchToFlatArray(tree["positionY"], False,selection)
        hit_z  = self.branchToFlatArray(tree["positionZ"], False,selection)
        
        
        hit_ass_truth_idx = self.branchToFlatArray(tree["maxE_particle_index"], False,selection)
        hit_ass_truth_energy = self.branchToFlatArray(tree["maxE_particle_energy"], False,selection)
        
        #not used right now
        hit_ass_truth_pX = self.branchToFlatArray(tree["maxE_particle_pX"], False,selection)
        hit_ass_truth_pY = self.branchToFlatArray(tree["maxE_particle_pY"], False,selection)
        hit_ass_truth_pZ = self.branchToFlatArray(tree["maxE_particle_pZ"], False,selection)
        
        
        
        features = np.concatenate([
            hit_energy,
            hit_x   ,
            hit_y, 
            hit_z 
            ], axis=-1)
        
        farr = SimpleArray(features,rs,name="features")
        
        t_idxarr = SimpleArray(hit_ass_truth_idx,rs,name="t_idx")
        t_energyarr = SimpleArray(hit_ass_truth_energy,rs,name="t_energy")
        
        zeros = np.zeros_like(hit_ass_truth_energy)
        #just for compatibility
        t_posarr = SimpleArray(zeros,rs,name="t_pos")
        t_time = SimpleArray(zeros,rs,name="t_time")
        t_pid = SimpleArray(zeros,rs,name="t_pid") #this would need some massaging so we can't use the PID directly
        t_spectator = SimpleArray(zeros,rs,name="t_spectator")
        t_fully_contained = SimpleArray(zeros,rs,name="t_fully_contained")
        
        t_rest = SimpleArray(zeros,rs,name="t_rest") #breaks with old plotting but needs to be done at some point
                
        return [farr, t_idxarr, t_energyarr, t_posarr, t_time, t_pid, t_spectator, t_fully_contained],[t_rest], []
    
    def createFullDict(self, f_arraylist):
        farr, rs, t_idxarr,_, t_energyarr,_, t_posarr,_, t_time,_, t_pid,_, t_spectator,_, t_fully_contained,_ = f_arraylist
        d={
            'hit_energy': farr[:,0:1],
            'hit_x': farr[:,1:2],
            'hit_y': farr[:,2:3],
            'hit_z': farr[:,3:4],
            
            't_idx': t_idxarr,
            't_energy': t_energyarr
            
            }
        
        return d, rs
    
    
    def createPandasDataFrame(self, eventno=-1):
        #since this is only needed occationally
        
        if self.nElements() <= eventno:
            raise IndexError("Event wrongly selected")
        
        tdc = self.copy()
        if eventno>=0:
            tdc.skim(eventno)
        
        d, rs = self.createFullDict(tdc.transferFeatureListToNumpy(False))
        
        d['hit_log_energy'] = np.log(d['hit_energy']+1)
        
        #and a continuous truth index
        
        
        allarr = []
        for k in d:
            allarr.append(d[k])
        allarr = np.concatenate(allarr,axis=1)
        
        frame = pd.DataFrame (allarr, columns = [k for k in d])
        if eventno>=0:
            return frame
        else:
            return frame, rs
    
    def interpretAllModelInputs(self, ilist):
        '''
        input: the full list of keras inputs
        returns: 
         - rechit feature array
         - t_idxarr
         - t_energyarr
         - t_posarr
         - t_time
         - t_pid
         - t_spectator
         - t_fully_contained
         - row_splits
         
        (for copy-paste: feat,  t_idx, t_energy, t_pos, t_time, t_pid, t_spectator ,t_fully_contained, row_splits)
        '''
        return ilist[0], ilist[2], ilist[4], ilist[6], ilist[8], ilist[10], ilist[12], ilist[14], ilist[1] 
     
    
      
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
    
    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)
        
    
    
 