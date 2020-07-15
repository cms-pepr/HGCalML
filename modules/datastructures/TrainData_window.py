



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
        
        selection = None
        
        if onlytruth:
            selection   = (tree["truthHitAssignementIdx"]).array()>-0.1  #0 
        
        
        
        #remove zero energy hits from removing of bad simclusters
        if selection is None:
            selection = (tree["recHitEnergy"]).array() > 0
        else:
            selection = np.logical_and(selection, (tree["recHitEnergy"]).array() > 0)
            
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True,selection)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False,selection)
        recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False,selection)
        recHitTheta              = self.branchToFlatArray(tree["recHitTheta"], False,selection)
        recHitR                = self.branchToFlatArray(tree["recHitR"], False,selection)
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
        truthHitAssignedEta     = self.branchToFlatArray(tree["truthHitAssignedEta"], False,selection)  #2
        truthHitAssignedPhi     = self.branchToFlatArray(tree["truthHitAssignedPhi"], False,selection)  #3
        truthHitAssignedR       = self.branchToFlatArray(tree["truthHitAssignedR"], False,selection)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False,selection)  #4
        truthHitAssignedDirPhi   = self.branchToFlatArray(tree["truthHitAssignedDirPhi"], False,selection)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False,selection)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        #windowEta                =
        #windowPhi                =
        
        ticlHitAssignementIdx    = self.branchToFlatArray(tree["ticlHitAssignementIdx"], False,selection)  #4
        ticlHitAssignedEnergies    = self.branchToFlatArray(tree["ticlHitAssignedEnergies"], False,selection)  #4


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
            np.array(ticlHitAssignementIdx, dtype='float32')   , #17
            ticlHitAssignedEnergies #18
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
    
    
    
    
