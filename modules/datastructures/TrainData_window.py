



from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore.compiled.c_simpleArray import simpleArray
import numpy as np
import uproot


class TrainData_window(TrainData):
    def __init__(self):
        TrainData.__init__(self)


    
    ######### helper functions for ragged interface
    ##### might be moved to DJC soon?, for now lives here
    
    def branchToFlatArray(self, b, returnRowSplits=False):
        
        a = b.array()
        nbatch = a.shape[0]
        if not returnRowSplits:
            return np.expand_dims(a.content, axis=1)
        
        nevents = a.shape[0]
        rowsplits = [0]
        
        #not super fast but ok fiven there aren't many events per file
        for i in range(nevents):
            rowsplits.append(rowsplits[-1] + a[i].shape[0])
            
        return np.expand_dims(a.content, axis=1), np.array(rowsplits, dtype='int64')


        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="WindowNTupler/tree"):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)[treename]
        nevents = tree.numentries
        
        print("n entries: ",nevents )
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False)
        recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False)
        recHitTheta              = self.branchToFlatArray(tree["recHitTheta"], False)
        recHitR                = self.branchToFlatArray(tree["recHitR"], False)
        recHitX                  = self.branchToFlatArray(tree["recHitX"], False)
        recHitY                  = self.branchToFlatArray(tree["recHitY"], False)
        recHitZ                  = self.branchToFlatArray(tree["recHitZ"], False)
        recHitDetID              = self.branchToFlatArray(tree["recHitDetID"], False)
        recHitTime               = self.branchToFlatArray(tree["recHitTime"], False)
        recHitID                 = self.branchToFlatArray(tree["recHitID"], False)
        recHitPad                = self.branchToFlatArray(tree["recHitPad"], False)
        
        ## weird shape for this truthHitFractions        = self.branchToFlatArray(tree["truthHitFractions"], False)
        truthHitAssignementIdx   = self.branchToFlatArray(tree["truthHitAssignementIdx"], False)   #0 
        truthHitAssignedEnergies = self.branchToFlatArray(tree["truthHitAssignedEnergies"], False)  #1
        truthHitAssignedX     = self.branchToFlatArray(tree["truthHitAssignedX"], False)  #2
        truthHitAssignedY     = self.branchToFlatArray(tree["truthHitAssignedY"], False)  #3
        truthHitAssignedZ     = self.branchToFlatArray(tree["truthHitAssignedZ"], False)  #3
        truthHitAssignedDirX   = self.branchToFlatArray(tree["truthHitAssignedDirX"], False)  #4
        truthHitAssignedDirY   = self.branchToFlatArray(tree["truthHitAssignedDirY"], False)  #4
        truthHitAssignedDirZ   = self.branchToFlatArray(tree["truthHitAssignedDirZ"], False)  #4
        truthHitAssignedEta     = self.branchToFlatArray(tree["truthHitAssignedEta"], False)  #2
        truthHitAssignedPhi     = self.branchToFlatArray(tree["truthHitAssignedPhi"], False)  #3
        truthHitAssignedR       = self.branchToFlatArray(tree["truthHitAssignedR"], False)  #3
        truthHitAssignedDirEta   = self.branchToFlatArray(tree["truthHitAssignedDirEta"], False)  #4
        truthHitAssignedDirPhi   = self.branchToFlatArray(tree["truthHitAssignedDirPhi"], False)  #4
        truthHitAssignedDirR    = self.branchToFlatArray(tree["truthHitAssignedDirR"], False)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        #windowEta                =
        #windowPhi                =
        
        
        
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
            truthHitAssignedEnergies ,
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
            truthHitAssignedDirR
        #    truthHitAssignedPIDs    
            ], axis=-1)
        
        tarr = simpleArray()
        tarr.createFromNumpy(truth, rs)
        
        print("truth",truth.shape)
                
        return [farr],[tarr],[]
    
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        pass
    

