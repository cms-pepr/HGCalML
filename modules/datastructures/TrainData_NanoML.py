from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore.compiled.c_simpleArray import simpleArray
import uproot3 as uproot
import awkward0 as ak
from numba import jit
import ROOT
import os
import pickle
import gzip
import tensorflow as tf
import numpy as np

    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)

    def hitObservable(self, tree, hitTypes, label, flatten=True, offsets=False):
        obs = map(lambda x: tree[x+"_"+label].array(), hitTypes)
        # For awkward1
        # jagged = np.concatenate([x for x in obs], axis=1)
        # off = np.cumsum(ak.to_numpy(ak.num(jagged)))
        # off = np.insert(off, 0, 0)
        jagged = ak.JaggedArray.concatenate([x for x in obs], axis=1)
        off = jagged.offsets
        if flatten:
            jagged = jagged.flatten()
        return (jagged, off) if offsets else jagged

    def truthObjects(self, sc, indices, null):
        return np.where(indices.flatten() < 0, null, sc[indices].flatten())
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename)
      
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]

        hits = ["RecHitHGC"+x for x in ["EE", "HEF", "HEB"]]
        recHitEnergy, offsets = self.hitObservable(tree, hits, "energy", offsets=True)
        recHitSimClusIdx = self.hitObservable(tree, hits, "MergedSimClusterIdx", flatten=False)
        recHitX = self.hitObservable(tree, hits, "x")
        recHitY = self.hitObservable(tree, hits, "y")
        recHitZ = self.hitObservable(tree, hits, "z")
        recHitDetaId = self.hitObservable(tree, hits, "detId")
        recHitTime = self.hitObservable(tree, hits, "time")
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))

        simClusterEnergy = tree["MergedSimCluster_boundaryEnergy"].array()
        simClusterX = tree["MergedSimCluster_impactPoint_x"].array()
        simClusterY = tree["MergedSimCluster_impactPoint_y"].array()
        simClusterZ = tree["MergedSimCluster_impactPoint_z"].array()
        simClusterTime = tree["MergedSimCluster_impactPoint_t"].array()
        simClusterPdgId = tree["MergedSimCluster_pdgId"].array()

        print(simClusterPdgId, len(simClusterPdgId))
        print(recHitSimClusIdx, len(recHitSimClusIdx))
        recHitTruthPID = self.truthObjects(simClusterPdgId, recHitSimClusIdx, 0)
        recHitTruthEnergy = self.truthObjects(simClusterEnergy, recHitSimClusIdx, -1)
        recHetTruthEnergyMu = np.where(np.abs(recHitTruthPID) == 13, recHitEnergy, recHitTruthEnergy)
        recHitTruthX = self.truthObjects(simClusterX, recHitSimClusIdx, 0)
        recHitTruthY = self.truthObjects(simClusterY, recHitSimClusIdx, 0)
        recHitTruthZ = self.truthObjects(simClusterZ, recHitSimClusIdx, 0)
        recHitTruthTime = self.truthObjects(simClusterZ, recHitSimClusIdx, -1)
        recHitTruthR = np.sqrt(recHitTruthX*recHitTruthX+recHitTruthY*recHitTruthY+recHitTruthZ*recHitTruthZ)
        recHitTruthTheta = np.arccos(recHitTruthZ/recHitTruthR)
        recHitTruthPhi = np.arctan(recHitTruthY/recHitTruthX)
        recHitTruthEta = -np.log(np.tan(recHitTruthTheta/2))

        features = np.array(np.concatenate([
            recHitEnergy,
            recHitEta,
            np.zeros(len(recHitEnergy)), #indicator if it is track or not
            recHitTheta,
            recHitR,
            recHitX,
            recHitY,
            recHitZ,
            recHitTime,
            ], axis=-1), dtype='float32')

        farr = simpleArray()
        farr.createFromNumpy(features, np.array(offsets, dtype='int64'))
        del features  

        truth = np.concatenate([
            recHitSimClusIdx.flatten(), # 0
            recHitTruthEnergy,
            recHitTruthX,
            recHitTruthY,
            recHitTruthZ,  #4
            #truthHitAssignedDirX,
            #truthHitAssignedDirY, #6
            #truthHitAssignedDirZ,
            recHitTruthEta     ,
            recHitTruthPhi,
            recHitTruthTime,  #10
            #truthHitAssignedDirEta,
            #truthHitAssignedDirR,
            #truthHitAssignedDepEnergies, #16
            
            ticlHitAssignementIdx  , #17
            ticlHitAssignedEnergies, #18
            recHitTruthPID #19 - 19+n_classes #won't be used anymore
            
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

def main():
    data = TrainData_NanoML()
    info = data.convertFromSourceFile("/eos/cms/store/cmst3/group/hgcal/CMG_studies/kelong/GeantTruthStudy/SimClusterNtuples/partGun_PDGid22_x96_Pt1.0To100.0_nano.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
