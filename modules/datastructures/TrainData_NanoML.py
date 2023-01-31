from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import uproot3 as uproot
import awkward as ak1
import pickle
import gzip
import numpy as np
import gzip
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

'''

    valued_pids = tf.zeros_like(t_pid)+4 #defaults to 4 as unkown
    valued_pids = tf.where(tf.math.logical_or(t_pid==22, tf.abs(t_pid) == 11), 0, valued_pids) #isEM
    
    valued_pids = tf.where(tf.abs(t_pid)==211, 1, valued_pids) #isHad
    valued_pids = tf.where(tf.abs(t_pid)==2212, 1, valued_pids) #proton isChHad
    valued_pids = tf.where(tf.abs(t_pid)==321, 1, valued_pids) #K+
    
    valued_pids = tf.where(tf.abs(t_pid)==13, 2, valued_pids) #isMIP
    
    valued_pids = tf.where(tf.abs(t_pid)==111, 3, valued_pids) #pi0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==2112, 3, valued_pids) #neutron isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==130, 3, valued_pids) #K0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==310, 3, valued_pids) #K0 short
    valued_pids = tf.where(tf.abs(t_pid)==3122, 3, valued_pids) #lambda isNeutrHadOrOther
    
    valued_pids = tf.cast(valued_pids, tf.int32)[:,0]
    
    known = tf.where(valued_pids==4,tf.zeros_like(valued_pids),1)
    valued_pids = tf.where(known<1,3,valued_pids)#set to 3
    known = tf.expand_dims(known,axis=1) # V x 1 style
'''

#for import
n_id_classes = 6 
id_idx_to_str = {
    0: 'muon',
    1: 'elec',
    2: 'gamma',
    3: 'chad',
    4: 'nhad',
    5: 'amb'}


id_str_to_idx = {
    'muon':0,
    'elec':1,
    'gamma':2,
    'chad':3,
    'nhad':4,
    'amb': 5}

def pdgIDToOneHot(pdgid, charge=None):
    #PF: [muon, Electron, gamma, charged had, neutral hadr, amb] -> 6
    # 
    
    #everything is a netrual hadron to begin with
    #create start array, use float multi to cast to float
    templ = ak1.zeros_like(pdgid,dtype=np.float64) 
    
    #default is nhad
    onehot = ak1.concatenate(4*[templ] + [templ+1.] + [templ],axis=-1)
    
    muon  = ak1.concatenate([templ+1.] + 5*[templ],axis=-1)
    elec  = ak1.concatenate(1*[templ] + [templ+1.] + 4*[templ],axis=-1)
    gamma = ak1.concatenate(2*[templ] + [templ+1.] + 3*[templ],axis=-1)
    chad  = ak1.concatenate(3*[templ] + [templ+1.] + 2*[templ],axis=-1)
    amb   = ak1.concatenate(5*[templ] + [templ+1.],axis=-1)
    #nhad  =  ak1.concatenate(4*[templ] + [templ+1.],axis=-1)
    
    #from IPython import embed
    #embed()
    
    if charge is not None:
        onehot = ak1.where(charge[...,0] != 0., chad, onehot)
        
    onehot = ak1.where(np.abs(pdgid[...,0])==11, elec, onehot)
    onehot = ak1.where(np.abs(pdgid[...,0])==13, muon, onehot)
    onehot = ak1.where(np.abs(pdgid[...,0])==22, gamma, onehot)
    onehot = ak1.where(np.abs(pdgid[...,0])==0, amb, onehot)
    
    return onehot


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
        
    
    def shuffle(self, seed=42):
        
        def _shuffle_array(awkarr, sorting):
            newa = awkarr[sorting]
            assert len(newa) == len(awkarr)
            return newa
        
        sorting = None
        for k in self.truth.keys():
            if sorting is None:
                sorting = np.arange(len(self.truth[k]))
                np.random.seed(seed)
                np.random.shuffle(sorting)
                
            self.truth[k] = _shuffle_array(self.truth[k],sorting)
            
        self.features = _shuffle_array(self.features, sorting)
        
        #this is quick, no need for printouts
        
    def checkConsistency(self):
        t_idx = self.truth['t_idx']
        for i,a in enumerate(t_idx):
            ua = np.unique(a.to_numpy())
            for ut in ua:
                pass #TBI
    
    def cleanUp(self):
        
        return #doesn't do anything right now, logic below but does not work with awkward
        '''
        removes showers that might harm the training 
        (scraping on either side or otherwise showers without tracks that have E_true >> E_deposited)
        '''
        for i_e in range(len(self.features)): #events
            t_idx = self.truth['t_idx'][i_e]
            feat = self.features[i_e]
            u_t_idx = np.unique(t_idx)
            for uut in u_t_idx:
                sel = t_idx[:,0] == uut
                u_t_energy = self.truth['t_energy'][i_e][sel][0] #at least one hit so this is safe
                #print(feat.to_numpy().shape)
                s_feat = feat[sel]
                #print(s_feat.to_numpy().shape)
                s_e = s_feat[:,0]
                dep_e = np.sum(s_e)
                
                if dep_e / u_t_energy < 0.05: #automatically not fulfilled by PF with tracks
                    print('found one', dep_e, ' ', u_t_energy)
                    print('replacing true energy')
                    self.truth['t_energy'][i_e][sel] = dep_e
            
                
    
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
                 usematchidx = "RecHitHGC_MergedSimClusterBestMatchIdx",
                 **kwargs):
        '''
        Guideline: this is more about clarity than performance. 
        If it improves clarity read stuff twice or more!
        '''
        self.use_true_muon_momentum = use_true_muon_momentum
        self.cp_plus_pu_mode = cp_plus_pu_mode
        self.cp_plus_pu_mode_reduce = cp_plus_pu_mode_reduce
        self.usematchidx = usematchidx
        
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
        recHitSimClusIdx = self._readAndSplit(tree, self.usematchidx)
        
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
        noSplitRecHitSimClusIdx = self._readArray(tree, self.usematchidx)
        return self._maskNoiseSC(tree,noSplitRecHitSimClusIdx)
    
    def _assignTruth(self,tree):
        if self.cp_plus_pu_mode:
            self._assignTruthCPPU(tree)
        else:
            self._assignTruthDef(tree)
            
    def _assignTruthCPPU(self, tree):
        print('\n\n>>>>>>WARNING: using calo particle plus PU mode: unless this is a special testing sample, you should not be using this function!<<<<\n\n')
        print('\n\n>>>>>> The configuration here also assumes exactly 1 non PU particle per endcap!<<<<<<<\n\n')
        raise ValueError("not to be used anymore")
        
        assert self.splitIdx is not None
        
         
    def _assignTruthDef(self, tree):
        
        assert self.splitIdx is not None
        
        nonSplitRecHitSimClusIdx = self._createTruthAssociation(tree)
        
        recHitTruthPID    = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_pdgId",nonSplitRecHitSimClusIdx)
        recHitTruthcharge    = self._assignTruthByIndexAndSplit(tree,"MergedSimCluster_charge",nonSplitRecHitSimClusIdx)
        
        recHitTruthOneHotID = pdgIDToOneHot(recHitTruthPID, recHitTruthcharge)
        
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
        
        #mask stray hits from the other side
        recHitSimClusIdx = ak1.where( recHitTruthZ[...,0] * recHitZ[...,0] < 0, -1, recHitSimClusIdx)
        
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
        
        #DEBUG!!!
        #ticlidx = self._readSplitAndExpand(tree,'RecHitHGC_TICLCandIdx')
        
        self.truth={}     #DEBUG!!!
        self.truth['t_idx'] = self._expand(recHitSimClusIdx)# now expand to one trailing dimension
        self.truth['t_energy'] = recHitTruthEnergy
        self.truth['t_pos'] = ak1.concatenate([recHitTruthX, recHitTruthY,recHitTruthZ],axis=-1)
        self.truth['t_time'] = recHitTruthTime
        self.truth['t_pid'] = recHitTruthOneHotID
        self.truth['t_spectator'] = recHitSpectatorFlag
        self.truth['t_fully_contained'] = fullyContained
        self.truth['t_rec_energy'] = recHitDepEnergy
        
        

############
class RecHitPFCollection(RecHitCollection):
    def __init__(self, **kwargs):
        
        super(RecHitPFCollection, self).__init__(
                 use_true_muon_momentum=False, 
                 cp_plus_pu_mode=False,
                 cp_plus_pu_mode_reduce=False,
                 usematchidx = "RecHitHGC_BestPFTruthPartIdx",
                 **kwargs)
        
    
    def _maskNoiseSC(self, tree,noSplitRecHitSimClusIdx):
        return noSplitRecHitSimClusIdx
        
        #this needs some work, does this exist for PF?
        goodSimClus = self._readArray(tree, "MergedSimCluster_isTrainable")
        goodSimClus = goodSimClus[noSplitRecHitSimClusIdx]
        return ak1.where(goodSimClus, noSplitRecHitSimClusIdx, -1)
    
    
    def _assignTruthDef(self, tree):
        '''
        This one is (obviously) different for PF
        '''
        assert self.splitIdx is not None
        
        nonSplitRecHitSimClusIdx = self._createTruthAssociation(tree)
        
        recHitTruthPID    = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_pdgId",nonSplitRecHitSimClusIdx)
        recHitTruthCharge = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_charge",nonSplitRecHitSimClusIdx)
        recHitTruthOneHotID = pdgIDToOneHot(recHitTruthPID, recHitTruthCharge)
        
        recHitTruthMomentum = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_p",nonSplitRecHitSimClusIdx)
        
        #this is tricky, would need to be boundary info
        recHitTruthX      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_x",nonSplitRecHitSimClusIdx)
        recHitTruthY      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_y",nonSplitRecHitSimClusIdx)
        recHitTruthZ      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_z",nonSplitRecHitSimClusIdx)
        recHitTruthTime   = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_t",nonSplitRecHitSimClusIdx)
        
        recHitDepEnergy = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_rec_energy",nonSplitRecHitSimClusIdx)
        
        
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
        recHitTruthMomentum = ak1.where(recHitSimClusIdx<0, recHitEnergy, recHitTruthMomentum)
        recHitTruthX = ak1.where(recHitSimClusIdx<0, recHitX, recHitTruthX)
        recHitTruthY = ak1.where(recHitSimClusIdx<0, recHitY, recHitTruthY)
        recHitTruthZ = ak1.where(recHitSimClusIdx<0, recHitZ, recHitTruthZ)
        recHitTruthTime = ak1.where(recHitSimClusIdx<0, recHitTime, recHitTruthTime)
        
        recHitSpectatorFlag = self._createSpectators(tree)
        #remove spectator flag for noise
        recHitSpectatorFlag = ak1.where(recHitSimClusIdx<0 , ak1.zeros_like(recHitSpectatorFlag), recHitSpectatorFlag)#this doesn't work for some reason!
        
        self.truth={}    
        self.truth['t_idx'] = self._expand(recHitSimClusIdx)# now expand to a trailing dimension
        self.truth['t_energy'] = recHitTruthMomentum
        self.truth['t_pos'] = ak1.concatenate([recHitTruthX, recHitTruthY,recHitTruthZ],axis=-1)
        self.truth['t_time'] = recHitTruthTime
        self.truth['t_pid'] = recHitTruthOneHotID
        self.truth['t_spectator'] = recHitSpectatorFlag
        self.truth['t_fully_contained'] = fullyContained
        self.truth['t_rec_energy'] = recHitDepEnergy
        
####        
 
class TrackCollection(RecHitPFCollection):
    def __init__(self, **kwargs):
        '''
        Guideline: this is more about clarity than performance. 
        If it improves clarity read stuff twice or more!
        
        For tracks, only PF makes sense in the first place, so no non-PF implementation here
        
        '''
        super(TrackCollection, self).__init__(**kwargs)
                                       
    def _readTree(self, tree):
        
        self._readSplits(tree, splitlabel='Track_HGCFront_z')
        
        trackEta = self._readSplitAndExpand(tree,"Track_HGCFront_eta")
        trackMom = self._readSplitAndExpand(tree,"Track_p")
        trackCharge = self._readSplitAndExpand(tree,"Track_charge")
        impactX = self._readSplitAndExpand(tree,"Track_HGCFront_x")
        impactY = self._readSplitAndExpand(tree,"Track_HGCFront_y")
        impactZ = self._readSplitAndExpand(tree,"Track_HGCFront_z")
        dec_z = self._readSplitAndExpand(tree,"Track_DecayVtx_z") / 100. #norm only this one
        
        impactR = np.sqrt(impactX**2+impactY**2+impactZ**2)+1e-3
        impactTheta = np.arccos(impactZ/impactR)
        
        self.features = ak1.concatenate([
            trackMom,
            trackEta,
            trackCharge, #also indicator if it is track or not
            impactTheta,
            impactR,
            impactX,
            impactY,
            impactZ,
            ak1.zeros_like(trackMom),#no time info (yet,could be from MTD here)
            dec_z # decay z, basically distinguished good from bad tracks given PF truth
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
    
    
    
    
    def _assignTruth(self, tree):
        
        assert self.splitIdx is not None
        
        nonSplitTrackSimClusIdx = self._readArray(tree, "Track_PFTruthPartIdx")#no 'best' this is 1:1
        
        trackTruthPID    = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_pdgId",nonSplitTrackSimClusIdx)
        trackTruthCharge = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_charge",nonSplitTrackSimClusIdx)
        trackTruthOneHotID = pdgIDToOneHot(trackTruthPID, trackTruthCharge)
        
        recHitTruthMomentum = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_p",nonSplitTrackSimClusIdx)
        
        #this is tricky, would need to be boundary info
        trackTruthX      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_x",nonSplitTrackSimClusIdx)
        trackTruthY      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_y",nonSplitTrackSimClusIdx)
        trackTruthZ      = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_z",nonSplitTrackSimClusIdx)
        trackTruthTime   = self._assignTruthByIndexAndSplit(tree,"PFTruthPart_calo_t",nonSplitTrackSimClusIdx)
        
        trackMom = self._readSplitAndExpand(tree,"Track_p")
        
        trackDepEnergy = trackMom
        
        zeros = ak1.zeros_like(trackDepEnergy)
        
        trackSimClusIdx = self._splitJaggedArray(nonSplitTrackSimClusIdx)
        
        self.truth={}     #DEBUG!!!
        self.truth['t_idx'] = self._expand(trackSimClusIdx)# now expand to a trailing dimension
        self.truth['t_energy'] = recHitTruthMomentum
        self.truth['t_pos'] = ak1.concatenate([trackTruthX, trackTruthY,trackTruthZ],axis=-1)
        self.truth['t_time'] = trackTruthTime
        self.truth['t_pid'] = trackTruthOneHotID
        self.truth['t_spectator'] = zeros #never spect
        self.truth['t_fully_contained'] = zeros+1.
        self.truth['t_rec_energy'] = trackDepEnergy
        
        
####################### end helpers        
    
class TrainData_NanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
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
        
        
        rechitcoll = RecHitCollection(tree=tree)
        
        if istraining:
            rechitcoll.cleanUp()
        
        #in a similar manner, we can also add tracks from conversions etc here
        rechitcoll.shuffle()
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
        #print(__name__,'createTruthDict: should be deprecated soon and replaced by a more uniform interface')
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
            if featd[k].shape[1] == 1:
                featd[k] = np.squeeze(featd[k],axis=1)
            elif k=='truthHitAssignedPIDs' or k== 't_pid':
                featd[k] =  np.argmax(featd[k], axis=-1)
            else:
                raise ValueError("only pid one-hot allowed to have more than one additional dimension, tried to squeeze "+ k)
        
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
        '''
        this function should not be necessary... why break with DJC standards?
        '''
        if not str(outfilename).endswith('.bin.gz'):
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'

        with gzip.open(outfilename, 'wb') as f2:
            pickle.dump(dumping_data, f2)

    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)


class TrainData_NanoMLPF(TrainData_NanoML):
    def __init__(self):
        TrainData_NanoML.__init__(self)
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        tree = uproot.open(filename)[treename]
        
        
        rechitcoll = RecHitPFCollection(tree=tree)
        trackcoll = TrackCollection(tree=tree)
        
        rechitcoll.append(trackcoll)
        
        rechitcoll.shuffle()
        # adds t_is_unique
        rechitcoll.addUniqueIndices()
        
        # converts to DeepJetCore.SimpleArray
        farr = rechitcoll.getFinalFeaturesSA()
        t = rechitcoll.getFinalTruthDictSA()
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ],[], []
        
        

class TrainData_NanoMLCPPU(TrainData_NanoML):
    def __init__(self):
        TrainData_NanoML.__init__(self)
        self.cp_plus_pu_mode=True
        
    def writeOutPredictionDict(self, dumping_data, outfilename):
        print('TrainData_NanoMLCPPU: special prediction write out - removing PU with DR>0.5 to all non-pu particles')
        from globals import pu
        '''
        dumping_data is a list of lists, each item corresponding to one event.
        within each item there is a 
          - feature dict with items of dimension V x X
          - truth dict with items of dimension V x X
          - predicted dict with items of dimension V x X
          - all np arrays
        '''
        
        #overwrite predicted, features, truth
        new_dumping_data=[]
        for e_dict in dumping_data:#loop over events - given event size should no add overhead
            f,t,p = e_dict
            
            
            Energy = f['recHitEnergy']
            X = f['recHitX']
            Y = f['recHitY']
            Z = f['recHitZ']
            coords = np.concatenate([X,Y,Z],axis=-1)#V x3
            t_idx = t['truthHitAssignementIdx']
            no_pu = t_idx < pu.t_idx_offset
            no_pu = np.logical_and(no_pu, t_idx>=0)[:,0] #V
            

            no_pu_uidx = np.unique(t_idx[no_pu])
            
            print('no_pu_uidx',no_pu_uidx.shape) # Ncp
            print('no_pu',no_pu.shape) # V x 1
            print('coords',coords.shape) # V x 1
            
            cp_coords = coords[no_pu]#V' x 3
            cp_coords = np.expand_dims(cp_coords,axis=0)#1 xV' x 3
            cp_tidxs = np.expand_dims(t_idx[no_pu],axis=0)#1 x V x 1
            
            print('cp_tidxs',cp_tidxs.shape)
            
            no_pu_uidx = np.expand_dims( np.expand_dims(no_pu_uidx,1),1)# Ncp x 1 x 1
            
            cp_coords = np.where(no_pu_uidx==cp_tidxs, cp_coords, 0 )
            print('cp_coords',cp_coords.shape)
            cp_mask = np.where(no_pu_uidx==cp_tidxs, np.ones_like(cp_coords), 0.)
            
            cp_coords = np.sum(cp_coords,axis=1)/(np.sum(cp_mask,axis=1)+1e-6)# Ncp x 3
            cp_eta = calc_eta(cp_coords[:,0:1],cp_coords[:,1:2],cp_coords[:,2:3])#Ncp x 1
            cp_phi = calc_phi(cp_coords[:,0:1],cp_coords[:,1:2],cp_coords[:,2:3])
            
            cp_eta = np.transpose(cp_eta,[1,0]) # 1 x Ncp
            cp_phi = np.transpose(cp_phi,[1,0])
            
            print('cp_eta',cp_eta)
            print('cp_phi',cp_phi)
            
            hit_eta = f['recHitEta']
            hit_phi = calc_phi(X,Y,Z) # V x 1
            
            d_eta = hit_eta-cp_eta # V x Ncp
            d_phi = deltaPhi(hit_phi,cp_phi) # V x Ncp
            
            DR = d_eta**2 + d_phi**2
            sel = DR < 0.5**2
            sel = np.sum(sel,axis=1)>0
            Nsel = np.sum(sel)
            print('Nsel',Nsel)
            
            #sanity check
            
            o_dicts = []
            for dd in e_dict:
                for k in dd.keys():
                    if k=='row_splits':
                        dd[k]=np.array([0,Nsel],dtype='int32')
                    else:
                        dd[k]=dd[k][sel]

                o_dicts.append(dd)
            
            new_dumping_data.append(o_dicts)
        
        super(TrainData_NanoMLCPPU, self).writeOutPredictionDict(new_dumping_data, outfilename)
        
        
        
        
        
        
        
        


