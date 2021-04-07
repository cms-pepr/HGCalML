from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
from uproot3_methods import TLorentzVectorArray
import awkward0 as ak
import pickle
import gzip
import numpy as np
from numba import jit
from IPython import embed
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
    
    def isValid(self):
        return True #needs to be filled

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
        # Kinda hacky...
        func = self.buildObs if label != "SimCluster" else self.bestMatch

        obs = map(lambda x: func(tree, x, label, ext), hitTypes)
        # For awkward1
        # jagged = np.concatenate([x for x in obs], axis=1)
        # off = np.cumsum(ak.to_numpy(ak.num(jagged)))
        # off = np.insert(off, 0, 0)
        jagged = ak.JaggedArray.concatenate([x for x in obs], axis=1)
        if splitIdx is not None:
            jagged = self.splitJaggedArray(jagged, splitIdx)

        return np.expand_dims(jagged.content, axis=1) if flatten else jagged

    def bestMatch(self, tree, base, match, ext=None):
        matches = tree[f"{base}_{match}_MatchIdx"].array()
        nmatches = tree[f"{base}_{match}NumMatch"].array()

        bestmatch = []
        for nmatch, match in zip(nmatches, matches):
            # First index is zero, not nmatches, don't need count of last entry
            offsets = np.zeros(len(nmatch), dtype='int32')
            offsets[1:] = np.cumsum(nmatch)[:-1]
            bestmatch.append(np.where(nmatch > 0, match[offsets], -1))
        obs = ak.JaggedArray.fromiter(bestmatch)
        if ext:
            obs = tree[ext].array()[obs]
        return obs

    def truthObjects(self, sc, indices, null, splitIdx=None):
        vals = sc[indices]
        offsets = vals.offsets
        vals[indices < 0] = null
        if splitIdx is not None:
            vals = self.splitJaggedArray(vals, splitIdx)
        return np.expand_dims(vals.content.astype(np.float32), axis=1)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        return self.base_convertFromSourceFile(filename, weighterobjects, istraining, treename=treename)

    def matchMergedUnmerged(self, merged, unmerged):
        mm = merged.cross(unmerged, nested=True)
        mm = mm[mm.i1.mergedIdx == mm.localindex]
        return mm

    def replaceMuonEnergy(self, matched):
        muons = matched.i1[abs(matched.i1.id) == 13]
        musum = muons.sum()
        muDepE = muons.depE.sum()
        # The sum basically just serves to collapse the inner array (should always have size 1)
        return (matched.i0 - musum).sum().energy + muDepE
      
    def mergeDepositedEnergy(self, matched):
        return matched.i1.sum().energy 
      
    def base_convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events",
                                   removeTracks=True):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]

        hits = ["RecHitHGC"+x for x in ["EE", "HEF", "HEB"]]
        recHitZUnsplit = self.hitObservable(tree, hits, "z", flatten=False)
        splitBy = recHitZUnsplit < 0
        recHitZ = self.splitJaggedArray(recHitZUnsplit, splitBy)
        offsets = recHitZ.offsets
        recHitZ = np.expand_dims(recHitZ.content, axis=1)

        recHitX = self.hitObservable(tree, hits, "x", splitIdx=splitBy)
        recHitY = self.hitObservable(tree, hits, "y", splitIdx=splitBy)
        recHitEnergy = self.hitObservable(tree, hits, "energy", splitIdx=splitBy)
        recHitDetaId = self.hitObservable(tree, hits, "detId", splitIdx=splitBy)
        recHitTime = self.hitObservable(tree, hits, "time", splitIdx=splitBy)
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))

        recHitSimClusIdx = self.hitObservable(tree, hits, "SimCluster", ext="SimCluster_MergedSimClusterIdx", flatten=False)

        mergedSC = TLorentzVectorArray.from_ptetaphim(tree["MergedSimCluster_pt"].array(),
                            tree["MergedSimCluster_eta"].array(),
                            tree["MergedSimCluster_phi"].array(),
                            tree["MergedSimCluster_mass"].array(),
        )
        unmergedSC = TLorentzVectorArray.from_ptetaphim(tree["SimCluster_pt"].array(),
                            tree["SimCluster_eta"].array(),
                            tree["SimCluster_phi"].array(),
                            tree["SimCluster_mass"].array(),
        )

        unmergedSC["mergedIdx"] = tree["SimCluster_MergedSimClusterIdx"].array()
        unmergedSC["depE"] = tree["SimCluster_recEnergy"].array()
        unmergedSC["id"] = tree["SimCluster_pdgId"].array()

        matched = self.matchMergedUnmerged(mergedSC, unmergedSC)
        simClusterDepEnergy = self.mergeDepositedEnergy(matched)
        simClusterEnergyMuCorr = self.replaceMuonEnergy(matched)

        simClusterEnergy = tree["MergedSimCluster_boundaryEnergy"].array()
        simClusterX = tree["MergedSimCluster_impactPoint_x"].array()
        simClusterY = tree["MergedSimCluster_impactPoint_y"].array()
        simClusterZ = tree["MergedSimCluster_impactPoint_z"].array()
        simClusterTime = tree["MergedSimCluster_impactPoint_t"].array()
        simClusterPdgId = tree["MergedSimCluster_pdgId"].array()

        # Mark simclusters outside of volume or with very few hits as noise
        # Maybe not a good idea if the merged SC pdgId is screwed up
        #noNeutrons = simClusterPdgId[recHitSimClusIdx] == 2112
        outside = (np.abs(simClusterX[recHitSimClusIdx]) > 300) | (np.abs(simClusterY[recHitSimClusIdx]) > 300) 
        fewHits = tree["MergedSimCluster_nHits"].array()[recHitSimClusIdx] < 10
        recHitSimClusIdx[outside | fewHits] = -1

        recHitTruthPID = self.truthObjects(simClusterPdgId, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthEnergy = self.truthObjects(simClusterEnergy, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthDepEnergy = self.truthObjects(simClusterDepEnergy, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthEnergyCorrMu = self.truthObjects(simClusterEnergyMuCorr, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthX = self.truthObjects(simClusterX, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthY = self.truthObjects(simClusterY, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthZ = self.truthObjects(simClusterZ, recHitSimClusIdx, 0., splitIdx=splitBy)
        recHitTruthTime = self.truthObjects(simClusterZ, recHitSimClusIdx, -1, splitIdx=splitBy)
        recHitTruthR = np.sqrt(recHitTruthX*recHitTruthX+recHitTruthY*recHitTruthY+recHitTruthZ*recHitTruthZ)
        recHitTruthTheta = np.arccos(np.divide(recHitTruthZ, recHitTruthR, out=np.zeros_like(recHitTruthZ), where=recHitTruthR!=0))
        recHitTruthPhi = np.arctan(np.divide(recHitTruthY, recHitTruthX, out=np.zeros_like(recHitTruthY), where=recHitTruthX!=0))
        recHitTruthEta = -np.log(np.tan(recHitTruthTheta/2))

        # Placeholder 
        zeroFeature = np.zeros(shape=(len(recHitEnergy), 1), dtype='float32')

        features = np.stack([
            recHitEnergy,
            recHitEta,
            zeroFeature, #indicator if it is track or not
            recHitTheta,
            recHitR,
            recHitX,
            recHitY,
            recHitZ,
            recHitTime,
            ], axis=-1)

        farr = SimpleArray(name="recHitFeatures")
        farr.createFromNumpy(features, offsets)
        del features  

        recHitSimClusIdx = np.expand_dims(self.splitJaggedArray(recHitSimClusIdx, splitIdx=splitBy).content.astype(np.float32), axis=1)
        truth = np.stack([
            recHitSimClusIdx, # 0
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
            
            ], axis=-1)
        
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
                'truthHitAssignedPIDs',
                'truthHitAssignedEnergiesUncorr'] 

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
    info = data.convertFromSourceFile("/eos/cms/store/user/kelong/ML4Reco/Gun50Part_CHEPDef/0_nanoML.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
