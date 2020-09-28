



from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore.compiled.c_simpleArray import simpleArray
import numpy as np
import uproot
from numba import jit
import ROOT
import os
import pickle
import gzip
from index_dicts import n_classes

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
            if s < 0:
                irechitEnergySums[ishowerIdx==s] = 0

        rechitEnergySums.append(irechitEnergySums)
    return rechitEnergySums
    

def findRechitsSum(showerIdx, recHitEnergy, rs):
    return np.concatenate(_findRechitsSum(showerIdx, recHitEnergy, rs), axis=0)
    
class TrainData_window(TrainData):
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

    
    
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, onlytruth, treename="WindowNTupler/tree",
                                   removeTracks=True):
        
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
            
        
        if removeTracks:
            selection = np.logical_and(selection, (tree["recHitID"]).array() > -0.5)
            
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True,selection)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False,selection)
        #recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False,selection)
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
        #truthHitAssignedR       = self.branchToFlatArray(tree["truthHitAssignedR"], False,selection)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False,selection)  #4
        truthHitAssignedDepEnergies   = self.branchToFlatArray(tree["truthHitAssignedDepEnergies"], False,selection)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False,selection)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        
        
        truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False,selection,
                                                          is3d = True)
        #print('truthHitAssignedPIDs',truthHitAssignedPIDs.shape)
        #print('truthHitAssignedEnergies',truthHitAssignedEnergies.shape)
        
        #print(truthHitAssignedPIDs)
        
        #truthHitAssignedPIDs = np.zeros_like(truthHitAssignedEnergies)
        #truthHitAssignedPIDs = np.tile(truthHitAssignedPIDs, [1, n_classes])
        
        
        #type_ambiguous,
        #type_electron,
        #type_photon,
        #type_mip,
        #type_charged_hadron,
        #type_neutral_hadron,
        
        # For tracks
        #
        #     *(data++) = t->obj->p();                  *(data++) = recHit->hit->energy();
        ####  *(data++) = t->pos.eta();                 *(data++) = recHit->pos.eta();
        ####  *(data++) = t->pos.phi();                 *(data++) = recHit->pos.phi();
        ####  *(data++) = t->pos.theta();               *(data++) = recHit->pos.theta();
        ####  *(data++) = t->pos.mag();                 *(data++) = recHit->pos.mag();
        ####  *(data++) = t->pos.x();                   *(data++) = recHit->pos.x();
        ####  *(data++) = t->pos.y();                   *(data++) = recHit->pos.y();
        ####  *(data++) = t->pos.z();                   *(data++) = recHit->pos.z();
        ####  *(data++) = t->obj->charge();             *(data++) = (float)recHit->hit->detid();
        ####  *(data++) = t->obj->chi2();               *(data++) = recHit->hit->time();
        ####  *(data++) = -1.; //track ID bit           *(data++) = 0.; //rechit ID bit
        ####  *(data++) = 0.; //pad                     *(data++) = 0.; //pad
        #
        
        #
        #
        
        
        #make these the only spectators, and set 'energy' to zero
        
        

        
        truthHitAssignedT   = self.branchToFlatArray(tree["truthHitAssignedT"], False,selection) 
        
        ticlHitAssignementIdx    = self.branchToFlatArray(tree["ticlHitAssignementIdx"], False,selection)  #4
        ticlHitAssignedEnergies    = self.branchToFlatArray(tree["ticlHitAssignedEnergies"], False,selection)  #4


        # For calculating spectators
        #rechitsSum = findRechitsSum(truthHitAssignementIdx, recHitEnergy, rs)
        #spectator = np.where(recHitEnergy < 0.0005 * rechitsSum, np.ones_like(recHitEnergy), np.zeros_like(recHitEnergy))
        
        
        ############ special track treatment for now
        #make tracks spectators
        isTrack = recHitID < 0
        spectator = np.where(isTrack, np.ones_like(isTrack), np.zeros_like(isTrack))
        recHitEnergy[isTrack] = 0. #don't use track momenta just use as seeds
        ##############
        
    
        
        # If truth shower energy < 5% of sum of rechits, assign rechits sum to it instead
        truthShowerEnergies  = truthHitAssignedEnergies.copy()
        
        #take them as is
        #truthShowerEnergies[rechitsSum<0.25*truthHitAssignedEnergies] = rechitsSum[rechitsSum<0.25*truthHitAssignedEnergies]

        #for now!
        truthShowerEnergies = truthHitAssignedDepEnergies #for now rechitsSum
        
        
        
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
        
        
        np.savetxt("textarr.txt",features[0:rs[1]])
        farr = simpleArray()
        farr.createFromNumpy(features, rs)
        #farr.cout()
        print("features",features.shape)
        
        del features
        
        
        
        truth = np.concatenate([
            truthHitAssignementIdx  , # 0
            truthShowerEnergies ,
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
            np.zeros_like(truthHitAssignedDirEta), #12
            truthHitAssignedDirR,
            spectator,#14
            truthHitAssignedEnergies,#15
            truthHitAssignedDepEnergies, #16
            ticlHitAssignementIdx  , #17
            ticlHitAssignedEnergies, #18
            truthHitAssignedPIDs    #19 - 19+n_classes
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
    
    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)
        
    
    
    
class TrainData_window_tracks(TrainData_window):
    def __init__(self):
        TrainData_window.__init__(self)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, onlytruth=False, treename=treename,
                                               removeTracks=False)
 
     
class TrainData_window_defaulttruth(TrainData_window):
    def __init__(self):
        TrainData_window.__init__(self)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTuplerDefaultTruth/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, onlytruth=False, treename=treename)
    

class TrainData_window_onlytruth(TrainData_window):
    def __init__(self):
        TrainData_window.__init__(self)



    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, onlytruth=True, treename=treename)
    
    
class TrainData_window_truthinjected  (TrainData_window):
    def __init__(self):
        TrainData_window.__init__(self)
        
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
        #recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False,selection)
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
        #truthHitAssignedR       = self.branchToFlatArray(tree["truthHitAssignedR"], False,selection)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False,selection)  #4
        truthHitAssignedDirPhi   = self.branchToFlatArray(tree["truthHitAssignedDirPhi"], False,selection)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False,selection)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        
        
        truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False,selection,
                                                          is3d = True)
        #print('truthHitAssignedPIDs',truthHitAssignedPIDs.shape)
        #print('truthHitAssignedEnergies',truthHitAssignedEnergies.shape)
        
        #print(truthHitAssignedPIDs)
        
        #truthHitAssignedPIDs = np.zeros_like(truthHitAssignedEnergies)
        #truthHitAssignedPIDs = np.tile(truthHitAssignedPIDs, [1, n_classes])
        
        
        #type_ambiguous,
        #type_electron,
        #type_photon,
        #type_mip,
        #type_charged_hadron,
        #type_neutral_hadron,
        

        
        truthHitAssignedT   = self.branchToFlatArray(tree["truthHitAssignedT"], False,selection) 
        
        ticlHitAssignementIdx    = self.branchToFlatArray(tree["ticlHitAssignementIdx"], False,selection)  #4
        ticlHitAssignedEnergies    = self.branchToFlatArray(tree["ticlHitAssignedEnergies"], False,selection)  #4


        # For calculating spectators
        rechitsSum = findRechitsSum(truthHitAssignementIdx, recHitEnergy, rs)
        spectator = np.where(recHitEnergy < 0.0005 * rechitsSum, np.ones_like(recHitEnergy), np.zeros_like(recHitEnergy))
        
        # If truth shower energy < 5% of sum of rechits, assign rechits sum to it instead
        truthShowerEnergies  = truthHitAssignedEnergies.copy()
        
        #take them as is
        #truthShowerEnergies[rechitsSum<0.25*truthHitAssignedEnergies] = rechitsSum[rechitsSum<0.25*truthHitAssignedEnergies]

        #for now!
        truthShowerEnergies = rechitsSum

        
        features = np.concatenate([
            recHitEnergy,
            recHitEta   ,
            truthHitAssignementIdx, #no phi anymore!
            truthShowerEnergies ,
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
            truthHitAssignementIdx  , # 0
            truthShowerEnergies ,
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
            truthHitAssignedDirPhi, #12
            truthHitAssignedDirR,
            spectator,#14
            truthHitAssignedEnergies,#15
            rechitsSum, #16
            ticlHitAssignementIdx  , #17
            ticlHitAssignedEnergies, #18
            truthHitAssignedPIDs    #19 - 19+n_classes
            ], axis=-1)
        
        tarr = simpleArray()
        tarr.createFromNumpy(truth, rs)
        
        print("truth",truth.shape)
                
        return [farr],[tarr],[]
    
      
    
