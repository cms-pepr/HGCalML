from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
from uproot3_methods import TLorentzVectorArray
import awkward0 as ak
import pickle
import gzip
import numpy as np
from numba import jit
#from IPython import embed


def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.)
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        self.splitIdx = None
        TrainData.__init__(self)

    def setSplitIdx(self, idx):
        self.splitIdx = idx
    
    def isValid(self):
        return True #needs to be filled

    def splitJaggedArray(self, jagged):
        split1 = jagged[self.splitIdx]
        split2 = jagged[~self.splitIdx]
        pairEvents = []
        for x in zip(split1, split2):
            pairEvents.extend(x)
        return ak.JaggedArray.fromiter(pairEvents)

    def hitObservable(self, tree, hitType, label, flatten=True, split=True):
        obs = tree["_".join([hitType, label])].array()
        if split:
            obs = self.splitJaggedArray(obs)

        return np.expand_dims(obs.content, axis=1) if flatten else obs

    def truthObjects(self, sc, indices, null, split=True, flatten=True):
        vals = sc[indices]
        offsets = vals.offsets
        vals[indices < 0] = null
        if split:
            vals = self.splitJaggedArray(vals)
        if not flatten:
            return vals
        return np.expand_dims(vals.content.astype(np.float32), axis=1)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename)

    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]

        hits = "RecHitHGC"
        recHitZUnsplit = self.hitObservable(tree, hits, "z", split=False, flatten=False)
        self.setSplitIdx(recHitZUnsplit < 0)

        recHitZ = self.splitJaggedArray(recHitZUnsplit)
        offsets = recHitZ.offsets
        recHitZ = np.expand_dims(recHitZ.content, axis=1)


        recHitX = self.hitObservable(tree, hits, "x")
        recHitY = self.hitObservable(tree, hits, "y")
        recHitEnergy = self.hitObservable(tree, hits, "energy")
        recHitDetaId = self.hitObservable(tree, hits, "detId")
        recHitTime = self.hitObservable(tree, hits, "time")
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))

        # Don't split this until the end, so it can be used to index the truth arrays
        recHitSimClusIdx = self.hitObservable(tree, hits, "BestMergedSimClusterIdx", split=False, flatten=False)

        simClusterDepEnergy = tree["MergedSimCluster_recEnergy"].array()
        simClusterEnergy = tree["MergedSimCluster_boundaryEnergy"].array()
        simClusterEnergyNoMu = tree["MergedSimCluster_boundaryEnergyNoMu"].array()
    
        # Remove muon energy, add back muon deposited energy
        unmergedId = tree["SimCluster_pdgId"].array()
        unmergedDepE = tree["SimCluster_recEnergy"].array()
        unmergedMatchIdx = tree["MergedSimCluster_SimCluster_MatchIdx"].array()
        unmergedMatches = tree["MergedSimCluster_SimClusterNumMatch"].array()
        unmergedDepEMuOnly = unmergedDepE
        unmergedDepEMuOnly[np.abs(unmergedId) != 13] = 0
        # Add another layer of nesting, then sum over all unmerged associated to merged
        unmergedDepEMuOnly = ak.JaggedArray.fromcounts(unmergedMatches.counts, 
            ak.JaggedArray.fromcounts(unmergedMatches.content, unmergedDepEMuOnly[unmergedMatchIdx].flatten()))
        depEMuOnly = unmergedDepEMuOnly.sum()

        simClusterEnergyMuCorr = simClusterEnergyNoMu + depEMuOnly

        simClusterX = tree["MergedSimCluster_impactPoint_x"].array()
        simClusterY = tree["MergedSimCluster_impactPoint_y"].array()
        simClusterZ = tree["MergedSimCluster_impactPoint_z"].array()
        simClusterTime = tree["MergedSimCluster_impactPoint_t"].array()
        simClusterEta = tree["MergedSimCluster_impactPoint_eta"].array()
        simClusterPhi = tree["MergedSimCluster_impactPoint_phi"].array()
        simClusterPdgId = tree["MergedSimCluster_pdgId"].array()

        # Mark simclusters outside of volume or with very few hits as noise
        # Maybe not a good idea if the merged SC pdgId is screwed up
        # Probably removing neutrons is a good idea though
        #noNeutrons = simClusterPdgId[recHitSimClusIdx] == 2112
        
        #filter non-boundary positions. Hopefully working?
        goodSimClus = tree["MergedSimCluster_isTrainable"].array()
        # Don't split by index here to keep same dimensions as SimClusIdx
        markNoise = self.truthObjects(~goodSimClus, recHitSimClusIdx, False, split=False, flatten=False).astype(np.bool_)
        
        nbefore = (recHitSimClusIdx < 0).sum().sum()
        recHitSimClusIdx[markNoise] = -1
        nafter = (recHitSimClusIdx < 0).sum().sum()

        print("Number of noise hits before", nbefore, "after", nafter)
        print('removed another factor of', nafter/nbefore, ' bad simclusters')

        recHitTruthPID = self.truthObjects(simClusterPdgId, recHitSimClusIdx, 0.)
        recHitTruthEnergy = self.truthObjects(simClusterEnergy, recHitSimClusIdx, 0)
        recHitTruthDepEnergy = self.truthObjects(simClusterDepEnergy, recHitSimClusIdx, 0)
        recHitTruthEnergyCorrMu = self.truthObjects(simClusterEnergyMuCorr, recHitSimClusIdx, 0)
        recHitTruthX = self.truthObjects(simClusterX, recHitSimClusIdx, 0)
        recHitTruthY = self.truthObjects(simClusterY, recHitSimClusIdx, 0)
        recHitTruthZ = self.truthObjects(simClusterZ, recHitSimClusIdx, 0)
        recHitTruthTime = self.truthObjects(simClusterTime, recHitSimClusIdx, 0)
        recHitTruthR = np.sqrt(recHitTruthX*recHitTruthX+recHitTruthY*recHitTruthY+recHitTruthZ*recHitTruthZ)
        recHitTruthTheta = np.arccos(np.divide(recHitTruthZ, recHitTruthR, out=np.zeros_like(recHitTruthZ), where=recHitTruthR!=0))
        recHitTruthPhi = np.arctan2(recHitTruthY, recHitTruthX)
        recHitTruthEta = self.truthObjects(simClusterEta, recHitSimClusIdx, 0)
        #print(recHitTruthPhi)
        #print(np.max(recHitTruthPhi))
        #print(np.min(recHitTruthPhi))

        # Placeholder 
        zeroFeature = np.zeros(shape=(len(recHitEnergy), 1), dtype='float32')

        features = np.concatenate([
            recHitEnergy,
            recHitEta,
            zeroFeature, #indicator if it is track or not
            recHitTheta,
            recHitR,
            recHitX,
            recHitY,
            recHitZ,
            recHitTime,
            ], axis=1)

        farr = SimpleArray(name="recHitFeatures")
        farr.createFromNumpy(features, offsets)
        del features  

        recHitSimClusIdx = np.expand_dims(self.splitJaggedArray(recHitSimClusIdx).content.astype(np.int32), axis=1)
        
        print('noise',(100*np.count_nonzero(recHitSimClusIdx<0))//recHitSimClusIdx.shape[0],'% of hits')
        print('truth eta min max',np.min(np.abs(recHitTruthEta[recHitSimClusIdx>=0])),np.max(np.abs(recHitTruthEta[recHitSimClusIdx>=0])))
        print('non-boundary truth positions', 
              np.count_nonzero(np.abs(np.abs(recHitTruthZ[recHitSimClusIdx>=0])-320)>5)/recHitTruthZ[recHitSimClusIdx>=0].shape[0])
        
        
        #now all numpy
        recHitTruthX[recHitSimClusIdx<0] = recHitX[recHitSimClusIdx<0]
        recHitTruthY[recHitSimClusIdx<0] = recHitY[recHitSimClusIdx<0]
        recHitTruthZ[recHitSimClusIdx<0] = recHitZ[recHitSimClusIdx<0]
        recHitTruthEnergyCorrMu[recHitSimClusIdx<0] = recHitEnergy[recHitSimClusIdx<0]
        recHitTruthTime[recHitSimClusIdx<0] = recHitTime[recHitSimClusIdx<0]
        
        
        #import matplotlib.pyplot as plt
        #plt.hist(np.abs(recHitTruthEnergyCorrMu[recHitSimClusIdx>=0]/recHitTruthDepEnergy[recHitSimClusIdx>=0])) 
        #plt.yscale('log')
        #plt.savefig("scat.pdf")
        
        truth = np.concatenate([
            np.array(recHitSimClusIdx,dtype='float32'), # 0
            recHitTruthEnergyCorrMu,
            recHitTruthX,
            recHitTruthY,
            recHitTruthZ,  #4
            zeroFeature, #truthHitAssignedDirX,
            zeroFeature, #6
            zeroFeature,
            recHitTruthEta     ,
            recHitTruthPhi,
            recHitTruthTime,  #10
            zeroFeature,
            zeroFeature,
            recHitTruthDepEnergy, #13
            
            zeroFeature, #14
            zeroFeature, #15
            recHitTruthPID #16 - 16+n_classes #won't be used anymore
            
            ], axis=1)
        
        
        
        t_idxarr = SimpleArray(recHitSimClusIdx, offsets, name="recHitTruthClusterIdx")
        
        t_energyarr = SimpleArray(name="recHitTruthEnergy")
        t_energyarr.createFromNumpy(recHitTruthEnergy, offsets)
        
        t_posarr = SimpleArray(name="recHitTruthPosition")
        t_posarr.createFromNumpy(np.concatenate([recHitTruthX, recHitTruthY],axis=-1), offsets)
        
        t_time = SimpleArray(name="recHitTruthTime")
        t_time.createFromNumpy(recHitTruthTime, offsets)
        
        t_pid = SimpleArray(name="recHitTruthID")
        t_pid.createFromNumpy(recHitTruthPID, offsets)
        
        #remaining truth is mostly for consistency in the plotting tools
        t_rest = SimpleArray(name="recHitTruth")
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
    
    def createTruthDict(self, truth, truthidx=None):
        out = {}
        keys = ['truthHitAssignementIdx',             #np.array(recHitSimClusIdx,dtype='float32'), # 0
                'truthHitAssignedEnergies',           # recHitTruthEnergyCorrMu,
                'truthHitAssignedX',                  # recHitTruthX,
                'truthHitAssignedY',                  # recHitTruthY,
                'truthHitAssignedZ',                  # recHitTruthZ,  #4
                'truthHitAssignedDirX',               # zeroFeature, #truthHitAssignedDirX,
                'truthHitAssignedDirY',               # zeroFeature, #6
                'truthHitAssignedDirZ',               # zeroFeature,
                'truthHitAssignedEta',                # recHitTruthEta     ,
                'truthHitAssignedPhi',                # recHitTruthPhi,
                'truthHitAssignedT',                  # recHitTruthTime,  #10
                'truthHitAssignedDirEta',             # zeroFeature,
                'truthHitAssignedDirR',               # zeroFeature,
                'truthHitAssignedDepEnergies',        # recHitTruthDepEnergy, #13
                'ticlHitAssignementIdx'  , #17        # zeroFeature, #14  
                'ticlHitAssignedEnergies', #18        # zeroFeature, #15  
                'truthHitAssignedPIDs',               # recHitTruthPID #16
                'truthHitAssignedEnergiesUncorr']     #  
                                                                  
        for key, i in zip(keys, range(len(keys))):
            out[key] = truth[:,i:i+1]
        
        if truthidx is not None:
            out['truthHitAssignementIdx']=truthidx
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
        import os
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
    info = data.convertFromSourceFile("/eos/cms/store/user/kelong/ML4Reco/Gun10Part_CHEPDef/Gun10Part_CHEPDef_fineCalo_nano.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
