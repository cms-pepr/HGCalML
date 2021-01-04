


from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore.compiled.c_simpleArray import simpleArray
import numpy as np
import uproot3 as uproot
from numba import jit
import ROOT
import os
import pickle
import gzip

    
class TrainData_OC(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
    
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
        try:
            fileTimeOut(filename, 2)
            tree = uproot.open(filename)["WindowNTupler/tree"]
            f=ROOT.TFile.Open(filename)
            t=f.Get("WindowNTupler/tree")
            if t.GetEntries() < 1:
                raise ValueError("")
        except Exception as e:
            print(e)
            return False
        return True

    
    
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)[treename]
        nevents = tree.numentries
        
        print("n entries: ",nevents )
        
        selection = (tree["recHitEnergy"]).array() > 0
        
        if removeTracks:
            selection = np.logical_and(selection, (tree["recHitID"]).array() > -0.5)
            
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True,selection)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False,selection)
        #recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False,selection)
        recHitTheta              = self.branchToFlatArray(tree["recHitTheta"], False,selection)
        recHitR                  = self.branchToFlatArray(tree["recHitR"], False,selection)
        recHitX                  = self.branchToFlatArray(tree["recHitX"], False,selection)
        recHitY                  = self.branchToFlatArray(tree["recHitY"], False,selection)
        recHitZ                  = self.branchToFlatArray(tree["recHitZ"], False,selection)
        recHitDetID              = self.branchToFlatArray(tree["recHitDetID"], False,selection)
        recHitTime               = self.branchToFlatArray(tree["recHitTime"], False,selection)
        recHitID                 = self.branchToFlatArray(tree["recHitID"], False,selection)
        recHitPad                = self.branchToFlatArray(tree["recHitPad"], False,selection)
        
        ## weird shape for this truthHitFractions        = self.branchToFlatArray(tree["truthHitFractions"], False)
        truthHitAssignementIdx   = self.branchToFlatArray(tree["truthHitAssignementIdx"], False,selection)   #0 
        truthHitAssignedEnergies = self.branchToFlatArray(tree["truthHitAssignedEnergies"], False,selection)  #1
        truthHitAssignedX     = self.branchToFlatArray(tree["truthHitAssignedX"], False,selection)  #2
        truthHitAssignedY     = self.branchToFlatArray(tree["truthHitAssignedY"], False,selection)  #3
        truthHitAssignedZ     = self.branchToFlatArray(tree["truthHitAssignedZ"], False,selection)  #3
        truthHitAssignedDirX   = self.branchToFlatArray(tree["truthHitAssignedDirX"], False,selection)  #4
        truthHitAssignedDirY   = self.branchToFlatArray(tree["truthHitAssignedDirY"], False,selection)  #4
        truthHitAssignedDirZ   = self.branchToFlatArray(tree["truthHitAssignedDirZ"], False,selection)  #4
        truthHitAssignedT      = self.branchToFlatArray(tree["truthHitAssignedT"], False,selection) 
        truthHitAssignedEta     = self.branchToFlatArray(tree["truthHitAssignedEta"], False,selection)  #2
        truthHitAssignedPhi     = self.branchToFlatArray(tree["truthHitAssignedPhi"], False,selection)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False,selection)  #4
        truthHitAssignedDepEnergies   = self.branchToFlatArray(tree["truthHitAssignedDepEnergies"], False,selection)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False,selection)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        
        
        truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False,selection,
                                                          is3d = True)
        
        truthHitAssignedPIDs = np.expand_dims(np.argmax(truthHitAssignedPIDs, axis=-1),axis=1) #no one-hot encoding
        truthHitAssignedPIDs = np.array(truthHitAssignedPIDs,dtype='float32')
        
        ticlHitAssignementIdx    = self.branchToFlatArray(tree["ticlHitAssignementIdx"], False,selection)  #4
        ticlHitAssignedEnergies    = self.branchToFlatArray(tree["ticlHitAssignedEnergies"], False,selection)  #4


        #for now!
        truthHitAssignedEnergies = truthHitAssignedDepEnergies #for now rechitsSum
        
        
        #object weighted energy (1.0 for highest energy hit per object)
        
            
        
        
        features = np.concatenate([
            recHitEnergy,
            recHitEta   ,
            recHitID, #indicator if it is track or not
            recHitTheta ,
            recHitR   ,
            recHitX     ,
            recHitY     ,
            recHitZ     ,
            recHitTime  
            ], axis=-1)
        
        

        farr = simpleArray()
        farr.createFromNumpy(features, rs)
        del features
        
        
        
        truth = np.concatenate([
            truthHitAssignementIdx  , # 0
            truthHitAssignedEnergies ,
            truthHitAssignedX     ,
            truthHitAssignedY,
            truthHitAssignedZ,  #4
            truthHitAssignedDirX,
            truthHitAssignedDirY, #6
            truthHitAssignedDirZ,
            truthHitAssignedEta     ,
            truthHitAssignedPhi,
            truthHitAssignedT,  #10
            truthHitAssignedDirEta,
            truthHitAssignedDirR,
            truthHitAssignedDepEnergies, #16
            
            ticlHitAssignementIdx  , #17
            ticlHitAssignedEnergies, #18
            truthHitAssignedPIDs    #19 - 19+n_classes #won't be used anymore
            
            ], axis=-1)
        
        
        
        t_idxarr = simpleArray()
        t_idxarr.createFromNumpy(truthHitAssignementIdx, rs)
        
        t_energyarr = simpleArray()
        t_energyarr.createFromNumpy(truthHitAssignedEnergies,rs)
        
        t_posarr = simpleArray()
        t_posarr.createFromNumpy(np.concatenate([truthHitAssignedX, truthHitAssignedY],axis=-1),rs)
        
        t_time = simpleArray()
        t_time.createFromNumpy(truthHitAssignedT,rs)
        
        t_pid = simpleArray()
        t_pid.createFromNumpy(truthHitAssignedPIDs,rs)
        
        #remaining truth is mostly for consistency in the plotting tools
        t_rest = simpleArray()
        t_rest.createFromNumpy(truth,rs)
        
        return [farr, t_idxarr, t_energyarr, t_posarr, t_time, t_pid],[t_rest], []
    
    def createFeatureDict(self,feat,addxycomb=True):
        d = {
        'recHitEnergy': feat[:,0:1] ,          #recHitEnergy,
        'recHitEta'   : feat[:,1:2] ,          #recHitEta   ,
        'recHitID'    : feat[:,2:3] ,          #recHitID, #indicator if it is track or not
        'recHitTheta' : feat[:,3:4] ,          #recHitTheta ,
        'recHitR'     : feat[:,4:5] ,          #recHitR   ,
        'recHitX'     : feat[:,5:6] ,          #recHitX     ,
        'recHitY'     : feat[:,6:7] ,          #recHitY     ,
        'recHitZ'     : feat[:,7:8] ,          #recHitZ     ,
        'recHitTime'  : feat[:,8:9] ,            #recHitTime  
        }
        if addxycomb:
            d['recHitXY']  = feat[:,5:7]    
            
        return d
    
    def createTruthDict(self, truth):
        
        out = {}
        keys = ['truthHitAssignementIdx',
                'truthHitAssignedEnergies', 
                'truthHitAssignedX',    
                'truthHitAssignedY',
                'truthHitAssignedZ',  
                'truthHitAssignedDirX',
                'truthHitAssignedDirY', 
                'truthHitAssignedDirZ',
                'truthHitAssignedEta'     
                'truthHitAssignedPhi',
                'truthHitAssignedT',  
                'truthHitAssignedDirEta',
                'truthHitAssignedDirR',
                'truthHitAssignedDepEnergies', 
                
                'ticlHitAssignementIdx'  , #17
                'ticlHitAssignedEnergies', #18
                
                'truthHitAssignedPIDs']
        
        for key, i in zip(keys, range(len(keys))):
            out[key] = truth[:,i:i+1]
        
        return out
    
    def createPandasDataFrame(self, eventno):
        #since this is only needed occationally
        import pandas as pd
        
        if self.nElements() <= eventno:
            raise IndexError("Event wrongly selected")
        tdc = self.copy()
        tdc.skim(eventno)
        
        f = tdc.transferFeatureListToNumpy()
        featd = self.createFeatureDict(f[0])
        t = tdc.transferTruthListToNumpy()
        truthd = self.createTruthDict(t[0])
        
        featd.update(truthd)
        
        del featd['recHitXY'] #so that it's flat
        
        featd['recHitLogEnergy'] = np.log(featd['recHitEnergy']+1)
        
        allarr = []
        for k in featd:
            allarr.append(featd[k])
        allarr = np.concatenate(allarr,axis=1)
        
        
        frame = pd.DataFrame (allarr, columns = [k for k in featd])
        return frame
        
    
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
         - row_splits
         
        (for copy-paste: feat,  t_idx, t_energy, t_pos, t_time, t_pid, row_splits)
        '''
        return ilist[0], ilist[2], ilist[4], ilist[6], ilist[8], ilist[10], ilist[1]
        
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename)
      
      
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
        
    
    
    
class TrainData_OC_tracks(TrainData_OC):
    def __init__(self):
        TrainData_OC.__init__(self)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename,
                                               removeTracks=False)
 