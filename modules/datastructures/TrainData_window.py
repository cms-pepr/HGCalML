



from DeepJetCore.TrainData import TrainData
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
        if not returnRowSplits:
            return np.expand_dims(a.content, axis=1)
        
        nevents = a.shape[0]
        rowsplits = [0]
        
        #not super fast but ok fiven there aren't many events per file
        for i in range(nevents):
            rowsplits.append(rowsplits[-1] + a[i].shape[0])
        
        return np.expand_dims(a.content, axis=1), np.array(rowsplits)


        
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        tree = uproot.open(filename)["WindowNTupler/tree"]
        nevents = tree.numentries
        
        recHitEnergy , rs        = self.branchToFlatArray(tree["recHitEnergy"], True)
        recHitEta                = self.branchToFlatArray(tree["recHitEta"], False)
        recHitRelPhi             = self.branchToFlatArray(tree["recHitRelPhi"], False)
        recHitTheta              = self.branchToFlatArray(tree["recHitTheta"], False)
        recHitMag                = self.branchToFlatArray(tree["recHitMag"], False)
        recHitX                  = self.branchToFlatArray(tree["recHitX"], False)
        recHitY                  = self.branchToFlatArray(tree["recHitY"], False)
        recHitZ                  = self.branchToFlatArray(tree["recHitZ"], False)
        recHitDetID              = self.branchToFlatArray(tree["recHitDetID"], False)
        recHitTime               = self.branchToFlatArray(tree["recHitTime"], False)
        recHitID                 = self.branchToFlatArray(tree["recHitID"], False)
        recHitPad                = self.branchToFlatArray(tree["recHitPad"], False)
        
        ## weird shape for this truthHitFractions        = self.branchToFlatArray(tree["truthHitFractions"], False)
        truthHitAssignementIdx   = self.branchToFlatArray(tree["truthHitAssignementIdx"], False)   #1 (first is row splits)
        truthHitAssignedEnergies = self.branchToFlatArray(tree["truthHitAssignedEnergies"], False)  #2
        truthHitAssignedEtas     = self.branchToFlatArray(tree["truthHitAssignedEtas"], False)  #3
        truthHitAssignedPhis     = self.branchToFlatArray(tree["truthHitAssignedPhis"], False)  #4
        ## weird shape for this truthHitAssignedPIDs     = self.branchToFlatArray(tree["truthHitAssignedPIDs"], False)
        #windowEta                =
        #windowPhi                =
        
        
        
        #rs = self.padRowsplits(rs, recHitEnergy.shape[0], nevents)
        
        features = np.concatenate([
            recHitEnergy,
            recHitEta   ,
            recHitRelPhi,
            recHitTheta ,
            recHitMag   ,
            recHitX     ,
            recHitY     ,
            recHitZ     ,
            recHitTime  
            ], axis=-1)
        
        
        farr = simpleArray()
        farr.createFromNumpy(features, rs)
        del features
        
        truth = np.concatenate([
        #    np.expand_dims(frs,axis=1),
        #    truthHitFractions        ,
            np.array(truthHitAssignementIdx, dtype='float32')   , # 1
            truthHitAssignedEnergies ,
            truthHitAssignedEtas     ,
            truthHitAssignedPhis     
        #    truthHitAssignedPIDs    
            ], axis=-1)
        
        tarr = simpleArray()
        tarr.createFromNumpy(truth, rs)
        
        return [farr],[tarr],[]
    
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        pass
    

