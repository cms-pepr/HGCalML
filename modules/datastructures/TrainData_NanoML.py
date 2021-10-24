from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
from uproot3_methods import TLorentzVectorArray
import awkward0 as ak
import awkward as ak1
import pickle
import gzip
import numpy as np
from numba import jit
import mgzip
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import time
#from IPython import embed
import os

def find_pcas(x_to_fit,PCA_n=2,min_hits=10):
    if x_to_fit.shape[0] < min_hits : #minimal number of hits , with less PCA does not make sense
        return None
    x_to_fit = StandardScaler().fit_transform(x_to_fit) # normalizing the features
    pca = PCA(n_components=PCA_n)
    pca.fit(x_to_fit)
    x_transformed = pca.fit_transform(x_to_fit)
    
    means=[x_transformed[:,i].mean() for i in range(0,PCA_n)]
    covs = np.cov(x_transformed.T)
    metric = 'mahalanobis'    
    mdist = cdist(x_transformed,[means] , metric=metric, V=covs)[:,0]
    return np.round(mdist,1) # return rounded distance 


def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.+1e-3)
    
def calc_phi(x, y, z):
    return np.arctan2(x, y)
    

######## helper classes ###########

class CollectionBase(object):
    def __init__(self, tree):
        '''
        Always use _readArray, not direct uproot to avoid compatibility issues
        
        define the following in a derived class:
        - _readTree(self,tree)
          - needs to include call to _readSplits(self,tree,splitlabel)
        - _assignTruth(self,tree)
        
        '''

        self.splitIdx=None
        self.features=None
        self.truth={}
        self.featurenames=[]
        
        self._readTree(tree)
        self._assignTruth(tree)

    def _readTree(self, tree):
        pass
    
    def _assignTruth(self, tree):
        pass
    
    def _readSplits(self, tree, splitlabel):
        split = ak1.from_awkward0(tree[splitlabel].array())
        self.splitIdx= split < 0
    
    def _splitJaggedArray(self, jagged):
        if self.splitIdx is None:
            raise ValueError("First determine split indices by running _readSplits")
        split1 = jagged[self.splitIdx]
        split2 = jagged[~self.splitIdx]
        arr = ak1.concatenate([split1,split2],axis=0)
        return arr
        
    def _assignTruthByIndexAndSplit(self, tree, label, indices, null=0):
        sc = label
        if type(label) is str:
            sc = ak1.from_awkward0(tree[label].array())
        vals = sc[indices]
        vals = ak1.where(indices<0, ak1.zeros_like(vals)+null, vals)
        ja = self._splitJaggedArray(vals)
        return self._expand(ja)
    
    def _expand(self, ja):
        arr = ja[...,np.newaxis]
        return arr
    
        #old: slow but working
        starttime = time.time()
        nplist=[]
        for a in ja:
            npexp = np.expand_dims(a.to_numpy(),axis=1)
            nplist.append(npexp)
        arr = ak1.from_iter(nplist)
        print('expand took',time.time()-starttime,'s')
        return arr
    
    def _readSplitAndExpand(self, tree, label):
        obs = self._readArray(tree, label)
        ja = self._splitJaggedArray(obs)
        return self._expand(ja)
    
    def _readAndSplit(self, tree, label):
        obs = self._readArray(tree, label)
        return self._splitJaggedArray(obs)

    def _readArray(self, tree, label):#for uproot3/4 ak0 to ak1 transition period
        arr = ak1.from_awkward0(tree[label].array())
        return arr
    
    def _checkshapes(self, a, b):
        assert len(a) == len(b)
        for c,d in zip(a,b):
            ok = c.to_numpy().shape[-1] == d.to_numpy().shape[-1] 
            ok = ok or c.to_numpy().shape[-1]==0 # one of the collections
            ok = ok or d.to_numpy().shape[-1]==0 # can be empty. awkward seems to be ok with that
            if not ok:
                print(c.to_numpy().shape[-1], d.to_numpy().shape[-1])
                raise RuntimeError("shape mismatch")
    
    def _checkConsistency(self):
        
        fhitspevent = [a.to_numpy().shape[0] for a in self.features]
        
        #now check if truth checks out
        for k in self.truth.keys():
            t=self.truth[k]
            if len(t) != len(fhitspevent):
                raise RuntimeError("Truth array ",k, "does not match feature length (",len(t),'vs',len(fhitspevent))
            for fah,ta in zip(fhitspevent,t):
                tah = ta.to_numpy().shape[0]
                if fah != tah:
                    raise RuntimeError("Truth subarray for",k,"has",tah,"hits, but expected",fah)
    
    
    def append(self, rhs):
        '''
        like concatenate, axis=1
        so that the track collection can be appended to the rechit collection
        '''
        self.splitIdx= ak1.concatenate([self.splitIdx,rhs.splitIdx],axis=1) 
        self._checkshapes(self.features,rhs.features)
        self.features = ak1.concatenate([self.features, rhs.features],axis=1)
        newtruth={}
        for k in self.truth.keys():
            self._checkshapes(self.truth[k],rhs.truth[k])
            newtruth[k] = ak1.concatenate([self.truth[k], rhs.truth[k]],axis=1)
        self.truth = newtruth
        
    def akToNumpyAndRs(self,awkarr):
        rs = np.array([0]+[len(a) for a in awkarr],dtype='int64')
        rs = np.cumsum(rs,axis=0)
        a = np.concatenate([a.to_numpy() for a in awkarr], axis=0)
        if 'float' in str(a.dtype):
            a = np.array(a, dtype='float32')
        elif 'int' in str(a.dtype):
            a = np.array(a, dtype='int32')
        else:
            raise ValueError(a.dtype, "is an unrecognised array format")
        return a, rs
    
    def getFinalFeaturesNumpy(self):
        '''
        returns features and row splits
        '''
        self._checkConsistency()
        return self.akToNumpyAndRs(self.features)#self.features.offsets()
    
    def getFinalFeaturesSA(self):
        a,rs = self.getFinalFeaturesNumpy()
        sa = SimpleArray(a,rs,name="recHitFeatures")
        #sa.setFeatureNames(self.featurenames) #not yet
        return sa
    
    def getFinalTruthDictNumpy(self):
        '''
        returns truth and row splits
        '''
        self._checkConsistency()
        out={}
        for k in self.truth.keys():
            out[k] = self.akToNumpyAndRs(self.truth[k])
        return out
    
    def getFinalTruthDictSA(self):
        truthdict = self.getFinalTruthDictNumpy()
        out={}
        for k in truthdict.keys():
            a,rs = truthdict[k]
            out[k] = SimpleArray(a,rs,name=k)
        return out
    
    def filter(self,mask):
        assert len(self.truth) and len(self.features)
        self.features = self.features[mask]
        for k in self.truth.keys():
            self.truth[k] = self.truth[k][mask]
    
    
class RecHitCollection(CollectionBase):
    def __init__(self, use_true_muon_momentum=False, **kwargs):
        '''
        Guideline: this is more about clarity than performance. 
        If it improves clarity read stuff twice or more!
        '''
        self.use_true_muon_momentum = use_true_muon_momentum
        
        #call this last!
        super(RecHitCollection, self).__init__(**kwargs)
                                       
    def _readTree(self, tree):
        
        # no truth here! Only features
        
        self._readSplits(tree, splitlabel='RecHitHGC_z')
        
        
        recHitEnergy = self._readSplitAndExpand(tree,"RecHitHGC_energy")
        recHitTime   = self._readSplitAndExpand(tree,"RecHitHGC_time")
        recHitX = self._readSplitAndExpand(tree,"RecHitHGC_x")
        recHitY = self._readSplitAndExpand(tree,"RecHitHGC_y")
        recHitZ = self._readSplitAndExpand(tree,"RecHitHGC_z")
        recHitHitR = self._readSplitAndExpand(tree,"RecHitHGC_hitr")
        
        recHitR = np.sqrt(recHitX*recHitX+recHitY*recHitY+recHitZ*recHitZ)
        recHitTheta = np.arccos(recHitZ/recHitR)
        recHitEta = -np.log(np.tan(recHitTheta/2))
        
        zeros = ak1.from_iter([np.zeros_like(a.to_numpy()) for a in recHitEta])
        
        self.features = ak1.concatenate([
            recHitEnergy,
            recHitEta,
            zeros, #indicator if it is track or not
            recHitTheta,
            recHitR,
            recHitX,
            recHitY,
            recHitZ,
            recHitTime,
            recHitHitR
            ], axis=-1)
        
        #this is just for bookkeeping
        self.featurenames = [
            'recHitEnergy',
            'recHitEta',
            'isTrack',
            'recHitTheta',
            'recHitR',
            'recHitX',
            'recHitY',
            'recHitZ',
            'recHitTime',
            'recHitHitR'
            ]
        #done
    
    def _createSpectators(self, tree):
        starttime = time.time()
        recHitX = self._readAndSplit(tree,"RecHitHGC_x")
        recHitY = self._readAndSplit(tree,"RecHitHGC_y")
        recHitZ = self._readAndSplit(tree,"RecHitHGC_z")
        recHitSimClusIdx = self._readAndSplit(tree,"RecHitHGC_BestMergedSimClusterIdx")
        
        #Define spectators 
        recHit_df_events = [pd.DataFrame({"recHitX":recHitX[i],
                                  "recHitY":recHitY[i],
                                  "recHitZ":recHitZ[i],
                                  "recHitSimClusIdx":recHitSimClusIdx[i]
                                  }) for i in range(len(recHitX))] 
         
        for ievent in range(len(recHit_df_events)):
            df_event = recHit_df_events[ievent]
            unique_shower_idx = np.unique(df_event['recHitSimClusIdx'])
            df_event['spectator_distance'] = 0. #
            df_event['recHitSimClus_nHits'] =  df_event.groupby('recHitSimClusIdx').recHitX.transform(len) #adding number of rec hits that are associated to this truth cluster
            for idx in unique_shower_idx:
                df_shower = df_event[df_event['recHitSimClusIdx']==idx]
                x_to_fit = df_shower[['recHitX','recHitY','recHitZ']].to_numpy()
                spectators_shower_dist = None
                try:
                    spectators_shower_dist = find_pcas(x_to_fit,PCA_n=2,min_hits=10)
                except:
                    pass
                if (spectators_shower_dist is not None) : 
                    spectators_idx = (df_shower.index.tolist())
                    df_event.loc[spectators_idx,'spectator_distance'] = spectators_shower_dist
                del df_shower
            del df_event
            
        recHitSpectatorFlag = ak1.from_iter([np.expand_dims(recHit_df_events[i]['spectator_distance'].to_numpy(),axis=1)
                                                       for i in range(len(recHit_df_events))])
        
        print('ended spectators after', time.time()-starttime,'s')
        return recHitSpectatorFlag   
    
    def _maskNoiseSC(self, tree,noSplitRecHitSimClusIdx):
        goodSimClus = self._readArray(tree, "MergedSimCluster_isTrainable")
        goodSimClus = goodSimClus[noSplitRecHitSimClusIdx]
        return ak1.where(goodSimClus, noSplitRecHitSimClusIdx, -1)
    
    def _createTruthAssociation(self, tree):
        noSplitRecHitSimClusIdx = self._readArray(tree,"RecHitHGC_BestMergedSimClusterIdx")
        return self._maskNoiseSC(tree,noSplitRecHitSimClusIdx)
    
    
    def _assignTruth(self, tree):
        
        assert self.splitIdx is not None
        
        nonSplitRecHitSimClusIdx = self._createTruthAssociation(tree)
        
        recHitTruthPID    = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_pdgId",nonSplitRecHitSimClusIdx)
        recHitTruthEnergy = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_boundaryEnergy",nonSplitRecHitSimClusIdx)
        
        if not self.use_true_muon_momentum:
            recHitDepEnergy = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_recEnergy",nonSplitRecHitSimClusIdx)
            recHitTruthEnergy = ak1.where(np.abs(recHitTruthPID[:,:,0])==13, recHitDepEnergy, recHitTruthEnergy)
            
        recHitTruthX      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_x",nonSplitRecHitSimClusIdx)
        recHitTruthY      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_y",nonSplitRecHitSimClusIdx)
        recHitTruthZ      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_z",nonSplitRecHitSimClusIdx)
        recHitTruthTime   = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_t",nonSplitRecHitSimClusIdx)
        
        fullyContained = ak1.where(np.abs(recHitTruthZ)[:,:,0]<323.,#somehow that seems necessary
                                   ak1.ones_like(recHitTruthZ),
                                   ak1.zeros_like(recHitTruthZ))
        
        recHitEnergy = self._readSplitAndExpand(tree,"RecHitHGC_energy")
        recHitTime   = self._readSplitAndExpand(tree,"RecHitHGC_time")
        recHitX = self._readSplitAndExpand(tree,"RecHitHGC_x")
        recHitY = self._readSplitAndExpand(tree,"RecHitHGC_y")
        recHitZ = self._readSplitAndExpand(tree,"RecHitHGC_z")
        
        # should not expand here to allow indexing as done below
        recHitSimClusIdx = self._splitJaggedArray(nonSplitRecHitSimClusIdx)
        
        # set noise to rec features
        recHitTruthEnergy = ak1.where(recHitSimClusIdx<0, recHitEnergy, recHitTruthEnergy)
        recHitTruthX = ak1.where(recHitSimClusIdx<0, recHitX, recHitTruthX)
        recHitTruthY = ak1.where(recHitSimClusIdx<0, recHitY, recHitTruthY)
        recHitTruthZ = ak1.where(recHitSimClusIdx<0, recHitZ, recHitTruthZ)
        recHitTruthTime = ak1.where(recHitSimClusIdx<0, recHitTime, recHitTruthTime)

        recHitSpectatorFlag = self._createSpectators(tree)
        #remove spectator flag for noise
        recHitSpectatorFlag = ak1.where(recHitSimClusIdx<0 , ak1.zeros_like(recHitSpectatorFlag), recHitSpectatorFlag)#this doesn't work for some reason!
        
        self.truth={}
        self.truth['t_idx'] = self._expand(recHitSimClusIdx)# now expand to a trailing dimension
        self.truth['t_energy'] = recHitTruthEnergy
        self.truth['t_pos'] = ak1.concatenate([recHitTruthX, recHitTruthY,recHitTruthZ],axis=-1)
        self.truth['t_time'] = recHitTruthTime
        self.truth['t_pid'] = recHitTruthPID
        self.truth['t_spectator'] = recHitSpectatorFlag
        self.truth['t_fully_contained'] = fullyContained
        
        

############

 
class TrackCollection(CollectionBase):
    def __init__(self, **kwargs):
        '''
        Guideline: this is more about clarity than performance. 
        If it improves clarity read stuff twice or more!
        '''
        super(TrackCollection, self).__init__(**kwargs)
                                       
    def _readTree(self, tree):
        
        self._readSplits(tree, splitlabel='Track_HGCFront_z')
        
        trackPt = self._readSplitAndExpand(tree,"Track_pt")
        trackEta = self._readSplitAndExpand(tree,"Track_HGCFront_eta")
        trackVertEta = self._readSplitAndExpand(tree,"Track_eta")
        trackMom = trackPt * np.cosh(trackVertEta)
        impactX = self._readSplitAndExpand(tree,"Track_HGCFront_x")
        impactY = self._readSplitAndExpand(tree,"Track_HGCFront_y")
        impactZ = self._readSplitAndExpand(tree,"Track_HGCFront_z")
        chi2 = self._readSplitAndExpand(tree,"Track_normChiSq")
        
        impactR = np.sqrt(impactX**2+impactY**2+impactZ**2)
        impactTheta = np.arccos(impactZ/impactR)
        
        self.features = ak1.concatenate([
            trackMom,
            trackEta,
            ak1.ones_like(trackMom), #indicator if it is track or not
            impactTheta,
            impactR,
            impactX,
            impactY,
            impactZ,
            ak1.zeros_like(trackMom),#no time info (yet,could be from MTD here)
            chi2 #this is radius for hits, here chi2 for tracks, since it's kinda realted to the impact points resolution...
            ], axis=-1)
        
        #this is just for bookkeeping, keep them the same as for hits
        self.featurenames = [
            'recHitEnergy',
            'recHitEta',
            'isTrack',
            'recHitTheta',
            'recHitR',
            'recHitX',
            'recHitY',
            'recHitZ',
            'recHitTime',
            'recHitHitR'
            ]
    
    
    def _getMatchIdxs(self, tree):
        
        #match by eta phi
        def deltaPhi(a,b):
            d = np.abs(a-b)
            return np.where(d>np.pi,d-np.pi,d)
        
        #no split here
        truthMom    = self._readArray(tree,"MergedSimCluster_boundaryEnergy")
        truthEta      = self._readArray(tree,"MergedSimCluster_impactPoint_eta")
        truthPhi      = self._readArray(tree,"MergedSimCluster_impactPoint_phi")
        truthpos = ak1.concatenate([self._expand(truthEta),self._expand(truthPhi)],axis=-1)
        
        impactEta = self._readArray(tree,"Track_HGCFront_eta")
        impactPhi = self._readArray(tree,"Track_HGCFront_phi")
        impactpos = ak1.concatenate([self._expand(impactEta),self._expand(impactPhi)],axis=-1)
        
        trackPt = self._readArray(tree,"Track_pt")
        trackVertEta = self._readArray(tree,"Track_eta")
        trackMom = trackPt * np.cosh(trackVertEta)
        
        #match by x,y, and momentum
        finalidxs = []
        for tpos, ipos, tmom, imom, ipt in zip(truthpos, impactpos, truthMom, trackMom, trackPt):
            # create default
            tpos, ipos, tmom, imom,ipt = tpos.to_numpy(), ipos.to_numpy(), tmom.to_numpy(), imom.to_numpy(), ipt.to_numpy()
            
            tpos = np.expand_dims(tpos, axis=0) #one is truth
            tmom = np.expand_dims(tmom, axis=0) #one is truth
            ipos = np.expand_dims(ipos, axis=1)
            imom = np.expand_dims(imom, axis=1)
            
            ipt = np.expand_dims(ipt,axis=1)
            #this is in cm. 
            posdiffsq = np.sum( (tpos[:,:,0:1]-ipos[:,:,0:1])**2 +deltaPhi(tpos[:,:,1:2],ipos[:,:,1:2])**2, axis=-1) # Trk x K
            #this is in %
            momdiff = 100.*np.abs(tmom - imom)/(imom+1e-3) #rel diff
            #scale position by 100 (DeltaR)
            totaldiff = np.sqrt(100.**2*posdiffsq + (momdiff*np.exp(-0.05*ipt))**2)#weight momentum difference less with higher momenta
            
            closestSC = np.argmin(totaldiff, axis=1) # Trk
            
            #more than 5 percent/1cm total difference
            closestSC[totaldiff[np.arange(len(closestSC)),closestSC] > 5] = -1
            
            finalidxs.append(closestSC)
            
        return ak1.from_iter(finalidxs)
        
    
    def _assignTruth(self, tree):
        
        nonSplitTrackSimClusIdx = self._getMatchIdxs(tree)
        
        truthEnergy = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_boundaryEnergy",nonSplitTrackSimClusIdx)
        
        
        truthPID    = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_pdgId",nonSplitTrackSimClusIdx)
        truthX      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_x",nonSplitTrackSimClusIdx)
        truthY      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_y",nonSplitTrackSimClusIdx)
        truthZ      = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_z",nonSplitTrackSimClusIdx)
        truthTime   = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_impactPoint_t",nonSplitTrackSimClusIdx)
        
        
        #some manual sets
        
        zeros = ak1.zeros_like(truthEnergy)
        splittruthidx = self._splitJaggedArray(nonSplitTrackSimClusIdx)
        
        spectator = ak1.where(splittruthidx<0, zeros+10., zeros)
        
        trackPt = self._readSplitAndExpand(tree,"Track_pt")
        trackVertEta = self._readSplitAndExpand(tree,"Track_eta")
        trackMom = trackPt * np.cosh(trackVertEta)
        
        impactX = self._readSplitAndExpand(tree,"Track_HGCFront_x")
        impactY = self._readSplitAndExpand(tree,"Track_HGCFront_y")
        impactZ = self._readSplitAndExpand(tree,"Track_HGCFront_z")
        
        truthX = ak1.where(splittruthidx<0, impactX, truthX)
        truthY = ak1.where(splittruthidx<0, impactY, truthY)
        truthZ = ak1.where(splittruthidx<0, impactZ, truthZ)
        
        truthEnergy = ak1.where(splittruthidx<0, trackMom, truthEnergy)
        
        truthidx = self._expand(splittruthidx)
        
        self.truth={}
        self.truth['t_idx'] = truthidx# for now
        self.truth['t_energy'] = truthEnergy
        self.truth['t_pos'] = ak1.concatenate([truthX,truthY,truthZ],axis=-1)
        self.truth['t_time'] = truthTime
        self.truth['t_pid'] = truthPID
        self.truth['t_spectator'] = spectator
        self.truth['t_fully_contained'] = zeros+1
        
    
####################### end helpers        
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.include_tracks = False

    
    def fileIsValid(self, filename):
        #uproot does not raise exceptions early enough for testing
        import ROOT
        try:
            fileTimeOut(filename, 2)
            tree = uproot.open(filename)["Events"]
            f=ROOT.TFile.Open(filename)
            t=f.Get("Events")
            if t.GetEntries() < 1:
                raise ValueError("")
        except Exception as e:
            print('problem with file',filename)
            print(e)
            return False
        return True
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]
        
        
        rechitcoll = RecHitCollection(use_true_muon_momentum=self.include_tracks,tree=tree)
        
        #in a similar manner, we can also add tracks from conversions etc here
        if self.include_tracks:
            trackcoll = TrackCollection(tree=tree)
            rechitcoll.append(trackcoll)
            
        farr = rechitcoll.getFinalFeaturesSA()
        t = rechitcoll.getFinalTruthDictSA()
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'] ],[], []
    

    def interpretAllModelInputs(self, ilist):
        '''
        input: the full list of keras inputs
        returns: td
         - rechit feature array
         - t_idxarr
         - t_energyarr
         - t_posarr
         - t_time
         - t_pid
         - t_spectator
         - t_fully_contained
         - row_splits
         
        (for copy-paste: feat,  t_idx, t_energy, t_pos, t_time, t_pid, t_spectator ,t_fully_contained, row_splits)
        '''
        return ilist[0], ilist[2], ilist[4], ilist[6], ilist[8], ilist[10], ilist[12], ilist[14], ilist[1] 
     
    def createFeatureDict(self,infeat,addxycomb=True):
        '''
        infeat is the full list of features, including truth
        '''
        
        #small compatibility layer with old usage.
        feat = infeat
        if type(infeat) == list:
            feat=infeat[0]
        
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
        'recHitHitR'  : feat[:,9:10] ,            #recHitTime  
        }
        if addxycomb:
            d['recHitXY']  = feat[:,5:7]    
            
        return d
    
    def createTruthDict(self, allfeat, truthidx=None):
        _, _, t_idx, _, t_energy, _, t_pos, _, t_time, _, t_pid, _,\
        t_spectator, _, t_fully_contained,_ = allfeat
        
        
        out={
            'truthHitAssignementIdx': t_idx,
            'truthHitAssignedEnergies': t_energy,
            'truthHitAssignedX': t_pos[:,0:1],
            'truthHitAssignedY': t_pos[:,1:2],
            'truthHitAssignedZ': t_pos[:,2:3],
            'truthHitAssignedEta': calc_eta(t_pos[:,0:1], t_pos[:,1:2], t_pos[:,2:3]),
            'truthHitAssignedPhi': calc_phi(t_pos[:,0:1], t_pos[:,1:2], t_pos[:,2:3]),
            'truthHitAssignedT': t_time,
            'truthHitAssignedPIDs': t_pid,
            'truthHitSpectatorFlag': t_spectator,
            'truthHitFullyContainedFlag': t_fully_contained,
            }
        return out
    
    def createPandasDataFrame(self, eventno=-1):
        #since this is only needed occationally
        
        if self.nElements() <= eventno:
            raise IndexError("Event wrongly selected")
        
        tdc = self.copy()
        if eventno>=0:
            tdc.skim(eventno)
        
        f = tdc.transferFeatureListToNumpy(False)
        featd = self.createFeatureDict(f[0])
        rs = f[1]
        truthd = self.createTruthDict(f)
        
        featd.update(truthd)
        
        del featd['recHitXY'] #so that it's flat
        
        featd['recHitLogEnergy'] = np.log(featd['recHitEnergy']+1.+1e-8)
        
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

    def writeOutPredictionDict(self, dumping_data, outfilename):
        if not str(outfilename).endswith('.bin.gz'):
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'

        with mgzip.open(outfilename, 'wb', thread=8, blocksize=2*10**7) as f2:
            pickle.dump(dumping_data, f2)

    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)


class TrainData_NanoMLTracks(TrainData_NanoML):
    def __init__(self):
        TrainData_NanoML.__init__(self)
        self.include_tracks = True


def main():
    data = TrainData_NanoML()
    info = data.convertFromSourceFile("/eos/cms/store/user/kelong/ML4Reco/Gun10Part_CHEPDef/Gun10Part_CHEPDef_fineCalo_nano.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
