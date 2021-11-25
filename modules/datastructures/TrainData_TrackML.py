from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray 
import awkward0 as ak
import pickle
import gzip
import numpy as np
from numba import jit
import mgzip
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
#from IPython import embed
import os
import numpy as np

class TrainData_TrackML(TrainData):
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

    def feat2dict(self, feat):
        d = {
            'x': feat[:, 0:1],
            'y': feat[:, 1:2],
            'z': feat[:, 2:3],
            'volume_id': feat[:, 3:4],
            'layer_id': feat[:, 4:5],
            'module_id': feat[:, 5:6]
        }
        return d


    def truth2dict(self, truth):

        raise ValueError("needs update after format change")
        d = {
            'particle_id': truth[:, 0:1],
            'hit_id': truth[:, 1:2],
            'tx': truth[:, 2:3],
            'ty': truth[:, 3:4],
            'tz': truth[:, 4:5],
            'weight': truth[:, 5:6]
        }
        return d

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
            'truthHitAssignedT': t_time,
            'truthHitAssignedPIDs': t_pid,
            'truthHitSpectatorFlag': t_spectator,
            'truthHitFullyContainedFlag': t_fully_contained,
            }
        return out

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

    def createFromCsvsIntoStandard(self, filename_truth,filename_hits,filename_cells,filename_particles, outfilename):
        df_hits = pd.read_csv(filename_hits, sep=',')
        df_truth = pd.read_csv(filename_truth, sep=',')
        df_particles = pd.read_csv(filename_particles, sep=',')
        df_cells = pd.read_csv(filename_cells, sep=',')

        cells_hit_ids = df_cells['hit_id'].to_numpy(copy=True)
        cells_hit_weights = df_cells['value'].to_numpy(copy=True)

        hit_ids = df_hits['hit_id']

        last_hit_id = -1
        last_sum = -1
        hit_ids_c = []
        rechitEnergy = []
        #    recHitEnergy,recHitEta,zeroFeature,recHitTheta,recHitR,recHitX,recHitY,recHitZ,recHitTime,

        for id, weight in zip(cells_hit_ids, cells_hit_weights):
            if id == last_hit_id:
                rechitEnergy[-1] += weight
            else:
                hit_ids_c.append(id)
                rechitEnergy.append(weight)
                last_hit_id = id

        assert np.sum(hit_ids-hit_ids_c) == 0

        df_hits['energy'] = rechitEnergy

        ptx = np.sqrt(df_particles['px'] ** 2 + df_particles['py'] ** 2)

        p = np.sqrt(df_particles['px'] ** 2 + df_particles['py'] ** 2 + df_particles['pz'] ** 2)

        # print("Particles", df_particles.shape)

        # print("Truths before", df_truth['particle_id'].shape)

        # print(len(np.unique(df_particles['particle_id'])), len(np.unique(df_truth['particle_id'])))
        df_particles = df_particles[(ptx > 1.5)]
        df_truth = df_truth[np.isin(df_truth['particle_id'], df_particles['particle_id'])]
        # print(len(np.unique(df_particles['particle_id'])), len(np.unique(df_truth['particle_id'])))
        # 0/0
        # df_truth['pt'] = pt
        # print("Truth after",df_truth.shape)

        # print("Hits before", df_hits.shape)
        df_hits = df_hits[np.isin(df_hits['hit_id'], df_truth['hit_id'])]
        df_cells = df_cells[np.isin(df_cells['hit_id'], df_truth['hit_id'])]

        rechHitEnergy = df_hits['energy'].to_numpy(dtype=np.float32)
        recHitX = df_hits['x'].to_numpy(dtype=np.float32)
        recHitY = df_hits['y'].to_numpy(dtype=np.float32)
        recHitZ = df_hits['z'].to_numpy(dtype=np.float32)
        recHitR = np.sqrt(recHitX ** 2. + recHitY ** 2. + recHitZ ** 2.)
        recHitTheta = np.arccos(recHitZ / recHitR)
        recHitEta = -np.log(np.tan(recHitTheta / 2))
        zeroFeature = recHitEta * 0

        # print("Hits after", df_hits.shape)


        particle_id = df_truth['particle_id'].to_numpy()

        particle_id2 = particle_id.copy()

        # unique_pids = np.unique(particle_id)

        # x = df_particles['particle_id']
        # print(unique_pids.shape, x.shape)
        # print(np.sum(x-unique_pids))
        # 0/0

        recHitTruthEnergy = zeroFeature.copy()
        recHitTruthDepEnergy = zeroFeature.copy()
        unique_pids = df_particles['particle_id']
        px = df_particles['px'].to_numpy()
        py = df_particles['py'].to_numpy()
        pz = df_particles['pz'].to_numpy()
        for i, u in enumerate(unique_pids):
            particle_id2[particle_id == u] = i
            recHitTruthEnergy[particle_id==u] = np.sqrt(px[i]**2 + py[i]**2 + 0*pz[i]**2)
            recHitTruthDepEnergy[particle_id==u] = np.sum(rechHitEnergy[particle_id==u])
        df_truth['particle_id'] = particle_id2
        df_truth['p'] = recHitTruthEnergy

        recHitSimClusIdx = df_truth['particle_id'].to_numpy().astype(np.int32)
        # recHitTruthEnergy = df_truth['pt'].to_numpy(dtype=np.float32)

        recHitTruthX = df_truth['tx'].to_numpy().astype(np.float32)
        recHitTruthY = df_truth['ty'].to_numpy().astype(np.float32)
        recHitTruthZ = df_truth['tz'].to_numpy().astype(np.float32)
        recHitTruthR = np.sqrt(recHitTruthX ** 2. + recHitTruthY ** 2. + recHitTruthZ ** 2.).astype(np.float32)
        recHitTruthTheta = np.arccos(recHitTruthZ / recHitTruthR).astype(np.float32)
        recHitTruthEta = -np.log(np.tan(recHitTruthTheta / 2)).astype(np.float32)
        recHitTruthPhi = np.arctan(recHitTruthY / recHitTruthX).astype(np.float32)
        recHitTruthTime = zeroFeature
        recHitTruthDepEnergy = recHitTruthDepEnergy.astype(np.float32)
        recHitTruthPID = zeroFeature

        truth = np.stack([
            np.array(recHitSimClusIdx, dtype='float32'),  # 0
            recHitTruthEnergy,
            recHitTruthX,
            recHitTruthY,
            recHitTruthZ,  # 4
            zeroFeature,  # truthHitAssignedDirX,
            zeroFeature,  # 6
            zeroFeature,
            recHitTruthEta,
            recHitTruthPhi,
            recHitTruthTime,  # 10
            zeroFeature,
            zeroFeature,
            recHitTruthDepEnergy,  # 13
            zeroFeature,  # 14
            zeroFeature,  # 15
            recHitTruthPID,  # 16 - 16+n_classes #won't be used anymore
            zeroFeature,
            zeroFeature], axis=1)
        truth = truth.astype(np.float32)

        features = np.stack([
            rechHitEnergy,
            recHitEta,
            zeroFeature,  # indicator if it is track or not
            recHitTheta,
            recHitR,
            recHitX,
            recHitY,
            recHitZ,
            zeroFeature,
            zeroFeature,
        ], axis=1)
        features = features.astype(np.float32)

        rs = np.array([0,len(features)], np.int64)

        farr = SimpleArray(name="recHitFeatures")
        farr.createFromNumpy(features, rs)

        t_rest = SimpleArray(name="recHitTruth")
        t_rest.createFromNumpy(truth, rs)

        # rs[1] = 100

        # print(rs, rs.dtype)
        #
        # 0/0

        t_idxarr = SimpleArray(name="recHitTruthClusterIdx")
        t_idxarr.createFromNumpy(recHitSimClusIdx[..., np.newaxis], rs)


        t_energyarr = SimpleArray(name="recHitTruthEnergy")
        t_energyarr.createFromNumpy(recHitTruthEnergy[..., np.newaxis], rs)

        t_posarr = SimpleArray(name="recHitTruthPosition")
        t_posarr.createFromNumpy(np.concatenate([recHitTruthX[..., np.newaxis], recHitTruthY[..., np.newaxis]], axis=-1), rs)

        # print(np.concatenate([recHitTruthX[..., np.newaxis], recHitTruthY[..., np.newaxis]], axis=-1).shape)
        # 0/0

        t_time = SimpleArray(name="recHitTruthTime")
        t_time.createFromNumpy(recHitTruthTime[..., np.newaxis], rs)

        t_pid = SimpleArray(name="recHitTruthID")
        t_pid.createFromNumpy(recHitTruthPID[..., np.newaxis], rs)

        t_spectator = SimpleArray(
            name="recHitSpectatorFlag")  # why do we have inconsistent namings, where is it needed? wrt. to truth array
        t_spectator.createFromNumpy(zeroFeature[..., np.newaxis], rs)

        t_fully_contained = SimpleArray(name="recHitFullyContainedFlag")
        t_fully_contained.createFromNumpy((zeroFeature[..., np.newaxis]+1).astype(np.int32), rs)

        # remaining truth is mostly for consistency in the plotting tools
        t_rest = SimpleArray(name="recHitTruth")
        t_rest.createFromNumpy(truth, rs)

        x,y,z = [farr, t_idxarr, t_energyarr, t_posarr, t_time, t_pid, t_spectator, t_fully_contained], [], []
        self._store(x,y,z)
        self.writeToFile(outfilename)
        print("Storing in new format")

    def createFromCsvs(self, filename_truth,filename_hits,filename_cells,filename_particles, outfilename):
        df_hits = pd.read_csv(os.path.join('/Users/shahrukhqasim/Downloads/kaggle_trackml/train_100_events/', filename_hits), sep=',')
        df_truth = pd.read_csv(os.path.join('/Users/shahrukhqasim/Downloads/kaggle_trackml/train_100_events/', filename_truth), sep=',')
        df_particles = pd.read_csv(os.path.join('/Users/shahrukhqasim/Downloads/kaggle_trackml/train_100_events/', filename_particles), sep=',')
        df_cells = pd.read_csv(os.path.join('/Users/shahrukhqasim/Downloads/kaggle_trackml/train_100_events/', filename_cells), sep=',')

        pt = np.sqrt(df_particles['px'] ** 2 + df_particles['py'] ** 2)
        df_particles = df_particles[(pt>1.5)]

        # print("Particles", df_particles.shape)

        # print("Truths before", df_truth['particle_id'].shape)
        df_truth = df_truth[np.isin(df_truth['particle_id'], df_particles['particle_id'])]
        # print("Truth after",df_truth.shape)

        # print("Hits before", df_hits.shape)
        df_hits = df_hits[np.isin(df_hits['hit_id'], df_truth['hit_id'])]
        # print("Hits after", df_hits.shape)

        particle_id = df_truth['particle_id']

        particle_id2 = particle_id.copy()

        unique_pids = np.unique(particle_id)
        for i, u in enumerate(unique_pids):
            particle_id2[particle_id==u] = i

        # print(np.unique(particle_id2))

        particle_id2 = particle_id2.astype(np.int32)


        farr = np.stack((df_hits['x'],
                         df_hits['y'],
                         df_hits['z'],
                         df_hits['volume_id'],
                         df_hits['layer_id'],
                         df_hits['module_id']), axis=-1)

        farr = farr.astype(np.float32)

        tarr = np.stack((particle_id2,
                         df_truth['hit_id'],
                         df_truth['tx'],
                         df_truth['ty'],
                         df_truth['tz'],
                         df_truth['weight']), axis=-1)

        tarr = tarr.astype(np.float32)
        #
        rs = np.array([0,len(farr)], np.int64)

        print(rs, len(farr), len(tarr))
        # rs[1] = 100

        # print(rs, rs.dtype)
        #
        # 0/0

        farr_ = SimpleArray(name="feats")
        farr_.createFromNumpy(farr, rs)

        tarr_ = SimpleArray(name="truths")
        tarr_.createFromNumpy(tarr, rs)


        tarr_2 = SimpleArray(name="truths")
        tarr_2.createFromNumpy(tarr, rs)

        # print(farr.shape, tarr.shape)

        self._store([farr_, tarr_], [tarr_2], [])

        # print(outfilename)
        self.writeToFile(outfilename)


    def writeOutPredictionDict(self, dumping_data, outfilename):
        if not str(outfilename).endswith('.bin.gz'):
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'

        with mgzip.open(outfilename, 'wb', thread=8, blocksize=2*10**7) as f2:
            pickle.dump(dumping_data, f2)

    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)

