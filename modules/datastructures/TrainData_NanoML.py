from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
import awkward0 as ak
import pickle
import gzip
import numpy as np
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
    
    def isValid(self):
        return True #needs to be filled

    def buildObs(self, tree, hitType, label, ext=None):
        obs = tree["_".join([hitType, label])].array()
        if ext:
            obs = tree[ext].array()[obs]
        return obs

    def hitObservable(self, tree, hitTypes, label, ext=None, flatten=True, offsets=False):
        obs = map(lambda x: self.buildObs(tree, x, label, ext), hitTypes)
        # For awkward1
        # jagged = np.concatenate([x for x in obs], axis=1)
        # off = np.cumsum(ak.to_numpy(ak.num(jagged)))
        # off = np.insert(off, 0, 0)
        jagged = ak.JaggedArray.concatenate([x for x in obs], axis=1)
        off = np.array(jagged.offsets, dtype='int64')
        if flatten:
            jagged = np.array(jagged.flatten(), dtype='float32')
        return (jagged, off) if offsets else jagged

    def truthObjects(self, sc, indices, null):
        return np.array(np.where(indices.flatten() < 0, null, sc[indices].flatten()), dtype='float32')
        
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
      
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]

        hits = ["RecHitHGC"+x for x in ["EE", "HEF", "HEB"]]
        recHitEnergy, offsets = self.hitObservable(tree, hits, "energy", offsets=True)
        recHitSimClusIdx = self.hitObservable(tree, hits, "SimClusterIdx", ext="SimCluster_MergedSimClusterIdx", flatten=False)
        recHitX = self.hitObservable(tree, hits, "x")
        recHitY = self.hitObservable(tree, hits, "y")
        recHitZ = self.hitObservable(tree, hits, "z")
        recHitDetaId = self.hitObservable(tree, hits, "detId")
        recHitTime = self.hitObservable(tree, hits, "time")
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))

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

        recHitTruthPID = self.truthObjects(simClusterPdgId, recHitSimClusIdx, 0.)
        recHitTruthEnergy = self.truthObjects(simClusterEnergy, recHitSimClusIdx, -1)
        recHitTruthDepEnergy = self.truthObjects(simClusterDepEnergy, recHitSimClusIdx, -1)
        recHitTruthEnergyNoMu = np.where(np.abs(recHitTruthPID) == 13, recHitTruthDepEnergy, recHitTruthEnergy)
        recHitTruthX = self.truthObjects(simClusterX, recHitSimClusIdx, 1.)
        recHitTruthY = self.truthObjects(simClusterY, recHitSimClusIdx, 0.)
        recHitTruthZ = self.truthObjects(simClusterZ, recHitSimClusIdx, 0.)
        recHitTruthTime = self.truthObjects(simClusterZ, recHitSimClusIdx, -1)
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
                'truthHitAssignedEta',
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
    
    
    def createPandasDataFrame(self, eventno=-1):
        #since this is only needed occationally
        import pandas as pd
        
        if self.nElements() <= eventno:
            raise IndexError("Event wrongly selected")
        
        tdc = self.copy()
        if eventno>=0:
            tdc.skim(eventno)
        
        f = tdc.transferFeatureListToNumpy(False)
        featd = self.createFeatureDict(f[0])
        rs = f[1]
        t = tdc.transferTruthListToNumpy(False)
        truthd = self.createTruthDict(t[0])
        
        featd.update(truthd)
        
        del featd['recHitXY'] #so that it's flat
        
        featd['recHitLogEnergy'] = np.log(featd['recHitEnergy']+1)
        
        allarr = []
        for k in featd:
            allarr.append(featd[k])
        allarr = np.concatenate(allarr,axis=1)
        
        
        frame = pd.DataFrame (allarr, columns = [k for k in featd])
        if eventno>=0:
            return frame
        else:
            return frame, rs
    
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
