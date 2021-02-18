from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
import awkward0 as ak
import pickle
import gzip
import numpy as np
from numba import jit
from IPython import embed
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)

    def buildObs(self, tree, hitType, label, ext=None):
        obs = tree["_".join([hitType, label])].array()
        if ext:
            # If index is -1, take -1, not the last entry
            newobs = tree[ext].array()[obs]
            newobs[obs < 0] = -1
            obs = newobs
        return obs

    def splitJaggedArray(self, jagged, splitIdx):
        split1 = jagged[splitIdx]
        split2 = jagged[~splitIdx]
        pairEvents = []
        for x in zip(split1, split2):
            pairEvents.extend(x)
        return ak.JaggedArray.fromiter(pairEvents)

    def hitObservable(self, tree, hitTypes, label, ext=None, flatten=True, splitIdx=None):
        obs = map(lambda x: self.buildObs(tree, x, label, ext), hitTypes)
        # For awkward1
        # jagged = np.concatenate([x for x in obs], axis=1)
        # off = np.cumsum(ak.to_numpy(ak.num(jagged)))
        # off = np.insert(off, 0, 0)
        jagged = ak.JaggedArray.concatenate([x for x in obs], axis=1)
        if splitIdx is not None:
            jagged = self.splitJaggedArray(jagged, splitIdx)

        return jagged.flatten() if flatten else jagged

    def truthObjects(self, sc, indices, null, splitIdx=None):
        vals = sc[indices]
        #vals[indices < 0] = -1
        if splitIdx is not None:
            vals = self.splitJaggedArray(vals, splitIdx)
        return np.array(vals.flatten(), dtype='float32')
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename)

    # Sum up deposited energy from all the associated unmerged simclusters
    # Unfortunately I'm not skilled enough to do this without loops
    def depositedEnergyMergedSC(self, unmergedDepEnergy, mergedSimClusterIdx):
        # Keeping track of the indices in case they're needed somewhere else later
        groups = []
        energies = []
        nev = len(unmergedDepEnergy)
        for i in range(nev):
            nsc = len(unmergedDepEnergy[i])
            entries = []
            evt_energies = []
            for j in range(nsc):
                matches = (mergedSimClusterIdx[i] == j).flatten().nonzero()[0]
                entries.append(matches)
                ienergy = unmergedDepEnergy[i][np.array(matches, dtype='int32')]
                evt_energies.append(ienergy)
            groups.append(entries)
            energies.append(evt_energies)

        # Not used currently, but could be useful, so leaving here
        # scIndices = ak.JaggedArray.fromiter(groups)
        return ak.JaggedArray.fromiter(energies).sum()

    def removeMuonEnergy(self, mergedSC, unmergedSCIdx, unmergedSC, unmergedSCId, unmergedDepE):
        energies = []
        for i, msc in enumerate(mergedSC):
            energy = np.zeros(len(msc), dtype='float32')
            for sci in unmergedSCIdx:
                if abs(unmergedSCId[sci]) == 13:
                    nomu = msc - unmergedSC[sci]
                    energy.append(nomu.energy())
            energies.append(energy)

      
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]

        hits = ["RecHitHGC"+x for x in ["EE", "HEF", "HEB"]]
        recHitZUnsplit = self.hitObservable(tree, hits, "z", flatten=False)
        splitBy = recHitZUnsplit < 0
        recHitZ = self.splitJaggedArray(recHitZUnsplit, splitBy)
        offsets = recHitZ.offsets
        recHitZ = recHitZ.flatten()

        recHitX = self.hitObservable(tree, hits, "x", splitIdx=splitBy)
        recHitY = self.hitObservable(tree, hits, "y", splitIdx=splitBy)
        recHitEnergy = self.hitObservable(tree, hits, "energy", splitIdx=splitBy)
        recHitDetaId = self.hitObservable(tree, hits, "detId", splitIdx=splitBy)
        recHitTime = self.hitObservable(tree, hits, "time", splitIdx=splitBy)
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))

        recHitSimClusIdx = self.hitObservable(tree, hits, "SimClusterIdx", ext="SimCluster_MergedSimClusterIdx", flatten=False)
        # TODO: Filter out simclusters that are off the boundary or don't have many hits

        simClusterEnergy = tree["MergedSimCluster_boundaryEnergy"].array()
        unmergedSimClusterRecEnergy = tree["SimCluster_recEnergy"].array()
        mergedSimClusterIdx = tree["SimCluster_MergedSimClusterIdx"].array()
        simClusterDepEnergy = self.depositedEnergyMergedSC(unmergedSimClusterRecEnergy, mergedSimClusterIdx)

        simClusterX = tree["MergedSimCluster_impactPoint_x"].array()
        simClusterY = tree["MergedSimCluster_impactPoint_y"].array()
        simClusterZ = tree["MergedSimCluster_impactPoint_z"].array()
        simClusterTime = tree["MergedSimCluster_impactPoint_t"].array()
        simClusterPdgId = tree["MergedSimCluster_pdgId"].array()

        recHitTruthPID = self.truthObjects(simClusterPdgId, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthEnergy = self.truthObjects(simClusterEnergy, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthDepEnergy = self.truthObjects(simClusterDepEnergy, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthEnergyNoMu = np.where(np.abs(recHitTruthPID) == 13, recHitTruthDepEnergy, recHitTruthEnergy)
        recHitTruthX = self.truthObjects(simClusterX, recHitSimClusIdx, 1., splitIdx=splitBy)
        recHitTruthY = self.truthObjects(simClusterY, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthZ = self.truthObjects(simClusterZ, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthTime = self.truthObjects(simClusterZ, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthR = np.sqrt(recHitTruthX*recHitTruthX+recHitTruthY*recHitTruthY+recHitTruthZ*recHitTruthZ)
        recHitTruthTheta = np.arccos(np.divide(recHitTruthZ, recHitTruthR, out=np.zeros_like(recHitTruthZ), where=recHitTruthR!=0))
        recHitTruthPhi = np.arctan(np.divide(recHitTruthY, recHitTruthX, out=np.zeros_like(recHitTruthY), where=recHitTruthX!=0))
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

        farr = SimpleArray()
        farr.createFromNumpy(features, offsets)
        del features  

        recHitSimClusIdx = np.array(recHitSimClusIdx.flatten(), dtype='float32')
        truth = np.concatenate([
            recHitSimClusIdx, # 0
            recHitTruthEnergyNoMu,
            recHitTruthX,
            recHitTruthY,
            recHitTruthZ,  #4
            np.zeros(len(recHitEnergy), dtype='float32'), #truthHitAssignedDirX,
            np.zeros(len(recHitEnergy), dtype='float32'), #truthHitAssignedDirY, #6
            np.zeros(len(recHitEnergy), dtype='float32'), #truthHitAssignedDirZ,
            recHitTruthEta     ,
            recHitTruthPhi,
            recHitTruthTime,  #10
            np.zeros(len(recHitEnergy), dtype='float32'), #truthHitAssignedDirEta,
            np.zeros(len(recHitEnergy), dtype='float32'), #truthHitAssignedDirR,
            np.zeros(len(recHitEnergy), dtype='float32'), # recHitTruthDepEnergy, #16
            
            np.zeros(len(recHitEnergy), dtype='float32'), #ticlHitAssignementIdx  , #17
            np.zeros(len(recHitEnergy), dtype='float32'), #ticlHitAssignedEnergies, #18
            recHitTruthPID #19 - 19+n_classes #won't be used anymore
            
            ], axis=-1)
        
        
        
        t_idxarr = SimpleArray()
        t_idxarr.createFromNumpy(recHitSimClusIdx, offsets)
        
        t_energyarr = SimpleArray()
        t_energyarr.createFromNumpy(recHitTruthEnergy, offsets)
        
        t_posarr = SimpleArray()
        t_posarr.createFromNumpy(np.concatenate([recHitTruthX, recHitTruthY],axis=-1), offsets)
        
        t_time = SimpleArray()
        t_time.createFromNumpy(recHitTruthTime, offsets)
        
        t_pid = SimpleArray()
        t_pid.createFromNumpy(recHitTruthPID, offsets)
        
        #remaining truth is mostly for consistency in the plotting tools
        t_rest = SimpleArray()
        t_rest.createFromNumpy(truth, offsets)
        
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
    info = data.convertFromSourceFile("/eos/cms/store/cmst3/group/hgcal/CMG_studies/kelong/GeantTruthStudy/SimClusterNtuples/testNanoML.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
