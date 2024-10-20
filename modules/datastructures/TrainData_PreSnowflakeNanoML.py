from DeepJetCore import TrainData
from DeepJetCore import SimpleArray
import pickle
import numpy as np
import tensorflow as tf
#from IPython import embed
import os
import gzip
import pdb

from datastructures.TrainData_NanoML import TrainData_NanoML
from djcdata.dataPipeline import TrainDataGenerator
from DebugLayers import switch_off_debug_plots
# from Layers import NormaliseTruthIdxs -> This caused circular imports
import time


from ragged_tools import normalise_index
class NormaliseTruthIdxs(tf.keras.layers.Layer):

    def __init__(self, active=True, add_rs_offset=True, **kwargs):
        '''
        changes arbitrary truth indices to well defined indices such that
        sort(unique(t_idx)) = -1, 0, 1, 2, 3, 4, 5, ... for each row split

        This should be called after every layer that could have modified
        the truth indices or removed hits, if the output needs to be regular.

        This Layer takes < 10ms usually so can be used generously.

        :param active: determines if it should be active.
                       In pure inference mode that might not be needed

        Inputs: truth indices, row splits
        Output: new truth indices

        '''
        if 'dynamic' in kwargs:
            super(NormaliseTruthIdxs, self).__init__(**kwargs)
        else:
            super(NormaliseTruthIdxs, self).__init__(dynamic=True,**kwargs)

        self.active = active
        self.add_rs_offset = add_rs_offset

    def get_config(self):
        config = {'active': self.active,
                  'add_rs_offset': self.add_rs_offset}
        base_config = super(NormaliseTruthIdxs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        assert len(inputs) == 2
        t_idx, rs = inputs

        #double unique
        if not self.active or rs.shape[0] == None:
            return t_idx

        return normalise_index(t_idx, rs, self.add_rs_offset)


def _getkeys(file):

    from DeepJetCore.wandb_interface import wandb_wrapper
    tmp = wandb_wrapper.active
    wandb_wrapper.active = False #prevent accidental logging when loading the model

    from DeepJetCore.modeltools import load_model
    tmp_model = load_model(file)
    print(tmp_model.output_shape.keys())
    output_keys = list(tmp_model.output_shape.keys())
    
    output_keys.remove('rs_down')
    output_keys.remove('sel_idx_up')
    output_keys.remove('rs_up')
    output_keys.remove('nidx_down')
    output_keys.remove('distsq_down')
    output_keys.remove('weights_down')
    output_keys.remove('row_splits') #will be recalculated
    output_keys.remove('no_noise_idx_stage_0') 
    output_keys.remove('no_noise_idx_stage_1') 
    output_keys.remove('orig_features') 
    output_keys.remove('orig_row_splits') 

    # output_keys.remove('orig_row_splits') # PZ: didn't exist
    if 'no_noise_row_splits' in output_keys:
        output_keys.remove('no_noise_row_splits')
    del tmp_model #close tf

    wandb_wrapper.active = tmp
    return output_keys

def calc_theta(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return np.arctan(r/z)


def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.+1e-3)


def calc_phi(x, y, z):
    return np.arctan2(y,x)#cms like

#just load once at import

class TrainData_PreSnowflakeNanoML(TrainData):

    model_keys = None

    def set_model(self):
        # self.path_to_pretrained = os.getenv("HGCALML")+'/models/double_tree_condensation_block/KERAS_check_model_last.h5'
        # self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre-cluster_tmp/KERAS_check_model_last.h5'
        self.path_to_pretrained = "/mnt/home/jkieseler/trainout/May24/dps_Aug24/KERAS_check_model_last.h5"
        self.path_to_pretrained = "/mnt/home/pzehetner/Paper/Models/precluster_new2/KERAS_check_model_block_0_epoch_01.h5"

    def __init__(self):
        TrainData.__init__(self)

        self.set_model()

        if TrainData_PreSnowflakeNanoML.model_keys is None:
            TrainData_PreSnowflakeNanoML.model_keys=_getkeys(self.path_to_pretrained)#load only once

        self.no_fork=True #make sure conversion can use gpu

        self.include_tracks = False
        self.cp_plus_pu_mode = False
        #preselection model used

        self.output_keys = TrainData_PreSnowflakeNanoML.model_keys

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename=""):
         
        #prevent accidental logging when loading the model
        from DeepJetCore.wandb_interface import wandb_wrapper
        tmp = wandb_wrapper.active
        wandb_wrapper.active = False 

        #this needs GPU
        from DeepJetCore.modeltools import load_model
        from LossLayers import LossLayerBase
        from MetricsLayers import MLBase
        model = load_model(self.path_to_pretrained)
        model = switch_off_debug_plots(model)
        #turn off all losses
        for l in model.layers:
            if isinstance(l, LossLayerBase):
                l.active = False
            l.trainable = False
            if hasattr(l, 'record_metrics'):
                l.record_metrics = False
            if isinstance(l, MLBase):
                print('deactivating metrics layer',l.name)
                l.active=False
            

        #also turn off metrics layers


        print("Loaded preselection model : ", self.path_to_pretrained)


        #outdict = model.output_shape
        td = TrainData_NanoML()
        td.readFromFile(filename)
        print("Reading from file : ", filename)

        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setSquaredElementsLimit(False)
        gen.setSkipTooLargeBatches(False)
        gen.setBuffer(td)

        nevents = gen.getNBatches()
        #print("Nevents : ", nevents)
        tot_in = 0
        rs = [[0]]#row splits need one extra dimension
        newout = {}
        feeder = gen.feedNumpyData()
        for i in range(nevents):
            start = time.time()
            feat,_ = next(feeder)
            tot_in += len(feat[0])
            out = model(feat)
            # print(out.keys())
            # Available keys: 
            # 'rs_down', 'rs_up', 'nidx_down', 'distsq_down', 'sel_idx_up', 
            # 'weights_down', 'prime_coords', 'coords', 'rechit_energy', 'features',
            # 'is_track', 'row_splits', 'orig_features', 'orig_row_splits', 't_idx',
            # 't_energy', 't_pos', 't_time', 't_pid', 't_spectator', 't_fully_contained',
            # 't_rec_energy', 't_is_unique', 't_only_minbias', 't_shower_class',
            # 't_spectator_weight', 'no_noise_idx_stage_0', 'no_noise_idx_stage_1', 'survived_both_stages'
            out['t_rec_energy_old']  = out['t_rec_energy'] # This has to be recalculated
            energies = out['rechit_energy']                                                         # (N,1)
            energy_hit = tf.where(out['is_track'] == 0, energies, tf.zeros_like(energies))[:,0]     # (N,)
            rs_tmp = out['row_splits'].numpy()
            id_flat = tf.squeeze(NormaliseTruthIdxs()([out['t_idx'], rs_tmp]))                      # (N,)
            num_classes = tf.unique(id_flat)[0].shape[0]
            energy_sum_per_class = tf.math.unsorted_segment_sum(energy_hit, id_flat, num_classes)   # (K,)
            energy_sum = tf.gather(energy_sum_per_class, id_flat)[:,tf.newaxis]                     # (N,1)
            track_or_noise = tf.logical_or(out['is_track'] == 1, out['t_idx'] == -1)
            out['t_rec_energy'] = tf.where(track_or_noise, energies, energy_sum)
            
            assert tf.math.reduce_min(id_flat) == -1
            assert tf.math.reduce_max(id_flat) + 2 == num_classes
            """
            energy_flat = tf.where(recHitID==0, rechit_energy, tf.zeros_like(rechit_energy))
            energy_flat = tf.squeeze(energy_flat)
            num_classes = tf.shape(unique_ids)[0]
            energy_sum_per_class = tf.math.unsorted_segment_sum(energy_flat, id_flat, num_classes) #TODO REMOVE!
            energy_sum = tf.gather(energy_sum_per_class, id_flat)
            energy_sum = tf.reshape(energy_sum, tf.shape(t_rec_energy))
            """
            # BLOCK to calculate new deposited energy
            # For all hits: -> New sum of all hits
            # For all noise: -> Noise energy is shower energy
            # For all tracks: -> Do nothing
            if not (rs_tmp[-1] == len(out[self.output_keys[0]])):
                print( self.output_keys[0], rs_tmp[-1],  len(out[self.output_keys[0]]) )
                raise ValueError('row splits do not match input size')

            #format time to ms and round to int
            print('reduction', len(feat[0]), '->', rs_tmp[-1], '(' ,round(rs_tmp[-1]/len(feat[0]) * 100.) ,'%)', 'time', round((time.time() - start) * 1000.), 'ms' )
            rs.append([rs_tmp[1]])


            if i == 0:
                for k in self.output_keys:
                    newout[k] = [out[k].numpy()]
            else:
                for k in self.output_keys:
                    newout[k].append(out[k].numpy())


        for k in self.output_keys:
            newout[k] = np.concatenate(newout[k],axis=0)
        #td.clear()
        #gen.clear()

        #####
        # converting to DeepJetCore.SimpleArray
        rs = np.array(rs, dtype='int64')
        rs = np.cumsum(rs,axis=0)
        rs = rs[:,0]
        print(rs)
        print([(k,newout[k].shape) for k in newout.keys()])
        print('reduction',tot_in, '->', rs[-1])

        outSA = []
        for k2 in self.output_keys:
            print(k2, ' ', newout[k2].shape)
            outSA.append(SimpleArray(newout[k2],rs,name=k2))

        del model
        wandb_wrapper.active = tmp

        return outSA,[], []


    def interpretAllModelInputs(self, ilist, returndict=True):
        #taken from TrainData_NanoML since it is similar, check for changes there
        if not returndict:
            raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE')

        out={}
        #data, rs, data, rs
        i_k=0
        out['row_splits'] = ilist[1]
        for i_k in range(len(self.output_keys)):
            out[self.output_keys[i_k]] = ilist[2*i_k]
        return out



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


    def createFeatureDict(self, infeat, addxycomb=True):
        '''
        infeat is the full list of features, including truth
        '''

        #small compatibility layer with old usage.
        feat = self.interpretAllModelInputs(infeat)
        # keys: ['row_splits', 't_idx', 't_energy',
        #        't_pos', 't_time', 't_pid', 't_spectator',
        #        't_fully_contained', 't_rec_energy', 't_is_unique',
        #        't_spectator_weight', 'prime_coords', 'rechit_energy',
        #        'is_track', 'coords', 'cond_coords', 'up_features',
        #        'select_prime_coords', 'features', 'sel_idx_up']
        coords = feat['coords']
        x = coords[:,0:1]
        y = coords[:,1:2]
        z = coords[:,2:3]

        # create pseudorapidity from x,y,z
        r = np.sqrt(x**2 + y**2)
        eta = calc_eta(x, y, z)
        theta = calc_theta(x, y, z)
        phi = calc_phi(x, y, z)

        d = {
        'recHitEnergy': feat['rechit_energy'],
        'recHitEta'   : eta,                        #recHitEta
        'recHitID'    : feat['is_track'],           #recHitID #indicator if it is track or not
        'recHitTheta' : theta,                      #recHitTheta
        'recHitR'     : r,                          #recHitR
        'recHitX'     : x,                          #recHitX
        'recHitY'     : y,                          #recHitY
        'recHitZ'     : z,                          #recHitZ
        'recHitTime'  : np.zeros_like(x),           #recHitTime
        'recHitHitR'  : np.zeros_like(x),           #What is this?
        }
        if addxycomb:
            d['recHitXY']  = np.concatenate([x,y], axis=1)

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
        if 't_only_minbias' in data.keys():
            out['t_only_minbias'] = data['t_only_minbias']
        if 't_minbias_weighted' in data.keys():
            out['t_minbias_weighted'] = data['t_minbias_weighted']
        return out


class TrainData_PreselectionNanoMLPF2(TrainData_PreSnowflakeNanoML):

    def set_model(self):
        self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_pf2/KERAS_model.h5'





