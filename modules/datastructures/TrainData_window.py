



from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore.compiled.c_simpleArray import simpleArray
import numpy as np
import uproot
from numba import jit
import ROOT
import os
import pickle
import gzip

#@jit(nopython=True)   
def _findRechitsSum(showerIdx, recHitEnergy, rs):

    rechitEnergySums = []

    for i in range(len(rs)-1):
        ishowerIdx = showerIdx[rs[i]:rs[i+1]]
        ishowerIdx = np.where(ishowerIdx<0,ishowerIdx-0.2,ishowerIdx)
        ishowerIdx += 0.1
        ishowerIdx = np.array(ishowerIdx,dtype='int64')
        irechitEnergy = recHitEnergy[rs[i]:rs[i+1]]

        uniques = np.unique(ishowerIdx)

        irechitEnergySums = np.zeros_like(irechitEnergy)
        for j in range(len(uniques)):
            s=uniques[j]
            energySum = np.sum(irechitEnergy[ishowerIdx==s])
            irechitEnergySums[ishowerIdx==s] = energySum

        rechitEnergySums.append(irechitEnergySums)
    return rechitEnergySums
    

def findRechitsSum(showerIdx, recHitEnergy, rs):
    return np.concatenate(_findRechitsSum(showerIdx, recHitEnergy, rs), axis=0)
    
class TrainData_window(TrainData):
    def __init__(self):
        TrainData.__init__(self)


    
    ######### helper functions for ragged interface
    ##### might be moved to DJC soon?, for now lives here
    
    def createSelection(self, jaggedarr):
        # create/read a jagged array 
        # with selects for every event
        pass
    
    def branchToFlatArray(self, b, returnRowSplits=False, selectmask=None):
        
        a = b.array()
        nbatch = a.shape[0]
        
        if selectmask is not None:
            a = a[selectmask]
        #use select(flattened) to select
        contentarr = a.content
        contentarr = np.expand_dims(contentarr, axis=1)
        
        if not returnRowSplits:
            return contentarr
        
        nevents = a.shape[0]
        rowsplits = [0]
        
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
                rowsplits.append(rowsplits[-1] + nonzero)
            
        return np.expand_dims(a.content, axis=1), np.array(rowsplits, dtype='int64')

    def fileIsValid(self, filename):
        try:
            fileTimeOut(filename, 2)
            tree = uproot.open(filename)["WindowNTupler/tree"]
            f=ROOT.TFile.Open(filename)
            t=f.Get("WindowNTupler/tree")
            if t.GetEntries() < 1:
                raise ValueError("")
        except Exception as e:
            return False
        return True

    
    
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, onlytruth, treename="WindowNTupler/tree"):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)[treename]
        nevents = tree.numentries
        
        print("n entries: ",nevents )
        select_truth = None
        if onlytruth:
            select_truth   = (tree["truthHitAssignementIdx"]).array()>-0.1  #0 
        
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True,select_truth)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False,select_truth)
        recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False,select_truth)
        recHitTheta              = self.branchToFlatArray(tree["recHitTheta"], False,select_truth)
        recHitR                = self.branchToFlatArray(tree["recHitR"], False,select_truth)
        recHitX                  = self.branchToFlatArray(tree["recHitX"], False,select_truth)
        recHitY                  = self.branchToFlatArray(tree["recHitY"], False,select_truth)
        recHitZ                  = self.branchToFlatArray(tree["recHitZ"], False,select_truth)
        recHitDetID              = self.branchToFlatArray(tree["recHitDetID"], False,select_truth)
        recHitTime               = self.branchToFlatArray(tree["recHitTime"], False,select_truth)
        recHitID                 = self.branchToFlatArray(tree["recHitID"], False,select_truth)
        recHitPad                = self.branchToFlatArray(tree["recHitPad"], False,select_truth)
        
        ## weird shape for this truthHitFractions        = self.branchToFlatArray(tree["truthHitFractions"], False)
        truthHitAssignementIdx   = self.branchToFlatArray(tree["truthHitAssignementIdx"], False,select_truth)   #0 
        truthHitAssignedEnergies = self.branchToFlatArray(tree["truthHitAssignedEnergies"], False,select_truth)  #1
        truthHitAssignedX     = self.branchToFlatArray(tree["truthHitAssignedX"], False,select_truth)  #2
        truthHitAssignedY     = self.branchToFlatArray(tree["truthHitAssignedY"], False,select_truth)  #3
        truthHitAssignedZ     = self.branchToFlatArray(tree["truthHitAssignedZ"], False,select_truth)  #3
        truthHitAssignedDirX   = self.branchToFlatArray(tree["truthHitAssignedDirX"], False,select_truth)  #4
        truthHitAssignedDirY   = self.branchToFlatArray(tree["truthHitAssignedDirY"], False,select_truth)  #4
        truthHitAssignedDirZ   = self.branchToFlatArray(tree["truthHitAssignedDirZ"], False,select_truth)  #4
        truthHitAssignedEta     = self.branchToFlatArray(tree["truthHitAssignedEta"], False,select_truth)  #2
        truthHitAssignedPhi     = self.branchToFlatArray(tree["truthHitAssignedPhi"], False,select_truth)  #3
        truthHitAssignedR       = self.branchToFlatArray(tree["truthHitAssignedR"], False,select_truth)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False,select_truth)  #4
        truthHitAssignedDirPhi   = self.branchToFlatArray(tree["truthHitAssignedDirPhi"], False,select_truth)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False,select_truth)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        #windowEta                =
        #windowPhi                =


        # For calculating spectators
        rechitsSum = findRechitsSum(truthHitAssignementIdx, recHitEnergy, rs)
        notSpectators = np.logical_or(np.greater(recHitEnergy, 0.01 * rechitsSum), np.less(np.abs(recHitZ), 330))

        # If truth shower energy < 5% of sum of rechits, assign rechits sum to it instead
        truthShowerEnergies  = truthHitAssignedEnergies.copy()
        truthShowerEnergies[rechitsSum<0.25*truthHitAssignedEnergies] = rechitsSum[rechitsSum<0.25*truthHitAssignedEnergies]


        #rs = self.padRowsplits(rs, recHitEnergy.shape[0], nevents)
        
        features = np.concatenate([
            recHitEnergy,
            recHitEta   ,
            recHitRelPhi,
            recHitTheta ,
            recHitR   ,
            recHitX     ,
            recHitY     ,
            recHitZ     ,
            recHitTime  
            ], axis=-1)
        
        farr = simpleArray()
        farr.createFromNumpy(features, rs)
        #farr.cout()
        print("features",features.shape)
        
        del features

        
        truth = np.concatenate([
        #    np.expand_dims(frs,axis=1),
        #    truthHitFractions        ,
            np.array(truthHitAssignementIdx, dtype='float32')   , # 0
            truthShowerEnergies ,
            truthHitAssignedX     ,
            truthHitAssignedY,
            truthHitAssignedZ,  #4
            truthHitAssignedDirX,
            truthHitAssignedDirY, #6
            truthHitAssignedDirZ,
            truthHitAssignedEta     ,
            truthHitAssignedPhi,
            truthHitAssignedR,  #10
            truthHitAssignedDirEta,
            truthHitAssignedDirPhi, #12
            truthHitAssignedDirR,
            np.logical_not(notSpectators),#14
            truthHitAssignedEnergies,
            rechitsSum,
        #    truthHitAssignedPIDs    
            ], axis=-1)
        
        tarr = simpleArray()
        tarr.createFromNumpy(truth, rs)
        
        print("truth",truth.shape)
                
        return [farr],[tarr],[]
    
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, onlytruth=False, treename=treename)
      
      
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
    
    def bla(self):
        print("hello")
    


class TrainData_window_onlytruth(TrainData_window):
    def __init__(self):
        TrainData_window.__init__(self)



    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, onlytruth=True, treename=treename)
    
    
    
    
