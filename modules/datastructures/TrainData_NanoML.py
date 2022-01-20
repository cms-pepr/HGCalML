from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
import awkward as ak1
import pickle
import gzip
import numpy as np
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
    return np.arctan2(y,x)#cms like
    

def deltaPhi(a,b):
    d = a-b
    d = np.where(d>2.*np.pi, d-2.*np.pi, d)
    return np.where(d<-2.*np.pi,d+2.*np.pi, d)
    
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
        
    def addUniqueIndices(self):
        '''
        Adds a '1' to exactly one hit representing
        a truth object, per truth object. such that
        number of truth objects = sum(unique)
        and 
        object properties = truth_per_hit_property[unique]
        This can be very helpful in loss functions etc.
        '''
        t_idx = self.truth['t_idx']
        nplist=[]
        for a in t_idx: #these are numpy arrays
            a = a.to_numpy()
            _,idx = np.unique(a,return_index=True)
            un = np.zeros_like(a)
            un[idx]=1
            nplist.append(un)
        akarr = ak1.from_iter(nplist)
        self.truth['t_is_unique']=akarr
        
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
    def __init__(self, use_true_muon_momentum=False, 
                 cp_plus_pu_mode=False,
                 cp_plus_pu_mode_reduce=False,
                 **kwargs):
        '''
        Guideline: this is more about clarity than performance. 
        If it improves clarity read stuff twice or more!
        '''
        self.use_true_muon_momentum = use_true_muon_momentum
        self.cp_plus_pu_mode = cp_plus_pu_mode
        self.cp_plus_pu_mode_reduce = cp_plus_pu_mode_reduce
        
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
        
        zeros = ak1.zeros_like(recHitEta)
        
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
            
        print('spectators calculated after',time.time()-starttime,'s')
        recHitSpectatorFlag = ak1.Array([np.expand_dims(recHit_df_events[i]['spectator_distance'].to_numpy(),axis=1)
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
    
    def _assignTruth(self,tree):
        if self.cp_plus_pu_mode:
            self._assignTruthCPPU(tree)
        else:
            self._assignTruthDef(tree)
            
    def _assignTruthCPPU(self, tree):
        '''
        

        '''
        print('\n\n>>>>>>WARNING: using calo particle plus PU mode: unless this is a special testing sample, you should not be using this function!<<<<\n\n')
        print('\n\n>>>>>> The configuration here also assumes exactly 1 non PU particle per endcap!<<<<<<<\n\n')
        
        
        assert self.splitIdx is not None
        
        scidx = self._readArray(tree,"RecHitHGC_BestSimClusterIdx")
        cpidx = self._readArray(tree,"SimCluster_CaloPartIdx")
        nonSplitRecHitSimClusIdx = ak1.where(scidx >= 0, cpidx[scidx], -1)#mark noise
        
        non_pu_cp = (self._readArray(tree,"CaloPart_eventId")+self._readArray(tree,"CaloPart_bunchCrossing"))<1
        
        recHitNotPU = self._assignTruthByIndexAndSplit(tree,non_pu_cp,nonSplitRecHitSimClusIdx)
        
        recHitTruthPID    = self._assignTruthByIndexAndSplit(tree,"CaloPart_pdgId",nonSplitRecHitSimClusIdx)
        recHitTruthEnergy = self._assignTruthByIndexAndSplit(tree,"CaloPart_energy",nonSplitRecHitSimClusIdx)
        
        fzeros      = self._assignTruthByIndexAndSplit(tree,"CaloPart_pt",nonSplitRecHitSimClusIdx)*0.
        recHitTruthX      = fzeros
        recHitTruthY      = fzeros 
        recHitTruthZ      = fzeros 
        recHitTruthTime   = fzeros 
        
        recHitDepEnergy = fzeros
        
        fullyContained = ak1.ones_like(fzeros)
        
        
        from globals import pu
        recHitSimClusIdx = self._splitJaggedArray(nonSplitRecHitSimClusIdx)
        recHitSimClusIdx = self._expand(recHitSimClusIdx)
        recHitSimClusIdx = recHitSimClusIdx + pu.t_idx_offset*(1-recHitNotPU)*(recHitSimClusIdx>=0)#only for not noise
        #pu_idx_offset[recHitNotPU[...,0]>0]=0
        #recHitSimClusIdx += pu_idx_offset
        #recHitSimClusIdx = recHitSimClusIdx + pu.t_idx_offset * (1-recHitNotPU[...,0])
        #testing
        #recHitSimClusIdx = recHitNotPU #(1+recHitSimClusIdx)*recHitNotPU[...,0]#[...,0]
        
        recHitSpectatorFlag = fzeros
        
        self.truth={}
        self.truth['t_idx'] = recHitSimClusIdx
        self.truth['t_energy'] = recHitTruthEnergy
        self.truth['t_pos'] = ak1.concatenate([recHitTruthX, recHitTruthY,recHitTruthZ],axis=-1)
        self.truth['t_time'] = recHitTruthTime
        self.truth['t_pid'] = recHitTruthPID
        self.truth['t_spectator'] = recHitSpectatorFlag
        self.truth['t_fully_contained'] = fullyContained
        self.truth['t_rec_energy'] = recHitDepEnergy
        
        if not self.cp_plus_pu_mode_reduce:
            return
        print('\n>>>reducing set by ∆R: handle with care<<<\n')
        #return
        #now remove access particles around initial one
        #this might not work because of close by gun
        #.... it does not work....
        #npu_cpeta = self._readArray(tree,"CaloPart_eta")
        #npu_cpphi = self._readArray(tree,"CaloPart_phi")
        
        
        recHitX = self._readSplitAndExpand(tree,"RecHitHGC_x")# EC x V x 1
        recHitY = self._readSplitAndExpand(tree,"RecHitHGC_y")
        recHitZ = self._readSplitAndExpand(tree,"RecHitHGC_z")
        
        cp_x = ak1.mean(recHitX[recHitNotPU[...,0]>0],axis=1)
        cp_y = ak1.mean(recHitY[recHitNotPU[...,0]>0],axis=1)
        cp_z = ak1.mean(recHitZ[recHitNotPU[...,0]>0],axis=1)
        
        recHitEta = calc_eta(recHitX, recHitY, recHitZ)
        recHitPhi = calc_phi(recHitX, recHitY, recHitZ)
        
        #return
        # ...
        #just loop over EC, there are only very few
        hitselector=[]
        for i_endcap in range(len(recHitEta)):
            
            ecps_eta=calc_eta(cp_x[i_endcap], cp_y[i_endcap], cp_z[i_endcap])
            ecps_phi=calc_phi(cp_x[i_endcap], cp_y[i_endcap], cp_z[i_endcap])
            
            ec_heta = recHitEta[i_endcap].to_numpy() # V x 1
            ec_hphi = recHitPhi[i_endcap].to_numpy() 
            
            deta = ec_heta-ecps_eta
            dphi = deltaPhi(ec_hphi,ecps_phi) # V x 1
            
            drsq = deta**2 + dphi**2  # V x 1
            
            close = np.array(drsq < 0.5**2,dtype='int')[...,0]
            #either_close = np.sum(close, axis=1)#does sum work on bool?
            #print(close.shape)
            hitselector.append(close)
            
            #last
        
        hitselector = ak1.from_iter(hitselector) # EC x V'
        #hitselector = recHitNotPU[...,0]
        #print('hitselector',ak1.sum(hitselector))
        
        for k in self.truth.keys():
            self.truth[k] = self.truth[k][hitselector>0]
        self.features = self.features[hitselector>0]
         
            
    def _assignTruthDef(self, tree):
        
        assert self.splitIdx is not None
        
        nonSplitRecHitSimClusIdx = self._createTruthAssociation(tree)
        
        recHitTruthPID    = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_pdgId",nonSplitRecHitSimClusIdx)
        recHitTruthEnergy = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_boundaryEnergy",nonSplitRecHitSimClusIdx)
        
        recHitDepEnergy = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_recEnergy",nonSplitRecHitSimClusIdx)
        if not self.use_true_muon_momentum:
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
        recHitDepEnergy = ak1.where(recHitSimClusIdx<0, recHitEnergy, recHitDepEnergy)

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
        self.truth['t_rec_energy'] = recHitDepEnergy
        
        

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
        
        impactR = np.sqrt(impactX**2+impactY**2+impactZ**2)+1e-3
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
        self.truth['t_rec_energy'] = trackMom
        
    
####################### end helpers        
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.include_tracks = False
        self.cp_plus_pu_mode = False

    
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
        
        
        rechitcoll = RecHitCollection(use_true_muon_momentum=self.include_tracks,
                                      cp_plus_pu_mode=self.cp_plus_pu_mode,
                                      tree=tree)
        
        #in a similar manner, we can also add tracks from conversions etc here
        if self.include_tracks:
            trackcoll = TrackCollection(tree=tree)
            rechitcoll.append(trackcoll)
        
        # adds t_is_unique
        rechitcoll.addUniqueIndices()
        
        # converts to DeepJetCore.SimpleArray
        farr = rechitcoll.getFinalFeaturesSA()
        t = rechitcoll.getFinalTruthDictSA()
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ],[], []
    

    def interpretAllModelInputs(self, ilist, returndict=True):
        if not returndict:
            raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE') 
        '''
        input: the full list of keras inputs
        returns: td
         - rechit feature array
         - t_idx
         - t_energy
         - t_pos
         - t_time
         - t_pid :             non hot-encoded pid
         - t_spectator :       spectator score, higher: further from shower core
         - t_fully_contained : fully contained in calorimeter, no 'scraping'
         - t_rec_energy :      the truth-associated deposited 
                               (and rechit calibrated) energy, including fractional assignments)
         - t_is_unique :       an index that is 1 for exactly one hit per truth shower
         - row_splits
         
        '''
        out = {
            'features':ilist[0],
            'rechit_energy': ilist[0][:,0:1], #this is hacky. FIXME
            't_idx':ilist[2],
            't_energy':ilist[4],
            't_pos':ilist[6],
            't_time':ilist[8],
            't_pid':ilist[10],
            't_spectator':ilist[12],
            't_fully_contained':ilist[14],
            'row_splits':ilist[1]
            }
        #keep length check for compatibility
        if len(ilist)>16:
            out['t_rec_energy'] = ilist[16]
        if len(ilist)>18:
            out['t_is_unique'] = ilist[18]
        return out
         
     
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
        '''
        This is deprecated and should be replaced by a more transparent way.
        '''
        print(__name__,'createTruthDict: should be deprecated soon and replaced by a more uniform interface')
        data = self.interpretAllModelInputs(allfeat,returndict=True)
        
        out={
            'truthHitAssignementIdx': data['t_idx'],
            'truthHitAssignedEnergies': data['t_energy'],
            'truthHitAssignedX': data['t_pos'][:,0:1],
            'truthHitAssignedY': data['t_pos'][:,1:2],
            'truthHitAssignedZ': data['t_pos'][:,2:3],
            'truthHitAssignedEta': calc_eta(data['t_pos'][:,0:1], data['t_pos'][:,1:2], data['t_pos'][:,2:3]),
            'truthHitAssignedPhi': calc_phi(data['t_pos'][:,0:1], data['t_pos'][:,1:2], data['t_pos'][:,2:3]),
            'truthHitAssignedT': data['t_time'],
            'truthHitAssignedPIDs': data['t_pid'],
            'truthHitSpectatorFlag': data['t_spectator'],
            'truthHitFullyContainedFlag': data['t_fully_contained'],
            }
        if 't_rec_energy' in data.keys():
            out['t_rec_energy']=data['t_rec_energy']
        if 't_hit_unique' in data.keys():
            out['t_is_unique']=data['t_hit_unique']
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
        
        #allarr = []
        #for k in featd:
        #    allarr.append(featd[k])
        #allarr = np.concatenate(allarr,axis=1)
        #
        #frame = pd.DataFrame (allarr, columns = [k for k in featd])
        #for k in featd.keys():
        #    featd[k] = [featd[k]]
        #frame = pd.DataFrame()
        for k in featd.keys():
            #frame.insert(0,k,featd[k])
            featd[k] = np.squeeze(featd[k],axis=1)
        
        frame = pd.DataFrame.from_records(featd)
        
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
        
        

class TrainData_NanoMLCPPU(TrainData_NanoML):
    def __init__(self):
        TrainData_NanoML.__init__(self)
        self.cp_plus_pu_mode=True


def main():
    data = TrainData_NanoML()
    info = data.convertFromSourceFile("/eos/cms/store/user/kelong/ML4Reco/Gun10Part_CHEPDef/Gun10Part_CHEPDef_fineCalo_nano.root",
                    [], False)
    print(info)    

if __name__ == "__main__":
    main()
