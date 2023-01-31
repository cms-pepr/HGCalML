from DeepJetCore.TrainData import TrainData
from DeepJetCore import SimpleArray
import pickle
import numpy as np
#from IPython import embed
import os
import gzip

from datastructures.TrainData_NanoML import TrainData_NanoML
from DeepJetCore.dataPipeline import TrainDataGenerator
from DebugLayers import switch_off_debug_plots


def _getkeys(file):
    from DeepJetCore.modeltools import load_model
    tmp_model = load_model(file)
    output_keys = list(tmp_model.output_shape.keys())
    output_keys.remove('row_splits')
    output_keys.remove('orig_row_splits')
    if 'no_noise_row_splits' in output_keys:
        output_keys.remove('no_noise_row_splits')
    del tmp_model #close tf
    return output_keys


def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.+1e-3)
    
   
def calc_phi(x, y, z):
    return np.arctan2(y,x)#cms like
    
#just load once at import

class TrainData_PreselectionNanoML(TrainData):
    
    model_keys = None
    
    def set_model(self):
        self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_june22/KERAS_model.h5'
    
    def __init__(self):
        TrainData.__init__(self)
        
        self.set_model()
        
        if TrainData_PreselectionNanoML.model_keys is None:
            TrainData_PreselectionNanoML.model_keys=_getkeys(self.path_to_pretrained)#load only once
        
        self.no_fork=True #make sure conversion can use gpu
        
        self.include_tracks = False
        self.cp_plus_pu_mode = False
        #preselection model used
        
        self.output_keys = TrainData_PreselectionNanoML.model_keys

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename=""):

        #this needs GPU
        from DeepJetCore.modeltools import load_model
        model = load_model(self.path_to_pretrained)
        model = switch_off_debug_plots(model)
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
            feat,_ = next(feeder)
            tot_in += len(feat[0])
            out = model(feat)
            rs_tmp = out['row_splits'].numpy()
            
            print('reduction', len(feat[0]), '->', rs_tmp[-1])
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
        print(rs)
        print([(k,newout[k].shape) for k in newout.keys()])
        print('reduction',tot_in, '->', rs[-1])
        
        outSA = []
        for k2 in self.output_keys:
            outSA.append(SimpleArray(newout[k2],rs,name=k2))

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


    def createFeatureDict(self,infeat,addxycomb=True):
        '''
        infeat is the full list of features, including truth
        '''
        
        #small compatibility layer with old usage.
        feat = infeat
        if (isinstance(feat, tuple)) and (len(feat) == 2):
            feat = feat[0]
        if not isinstance(feat, list) or not len(feat) == 30:
            raise ValueError('Expected first entry of features to be a list of length 30')
        if not isinstance(feat[26], np.ndarray) or not feat[26].shape[1] == 10:
            raise ValueError('Expected to find Nx10 array in features[0][26]')
        feat = feat[26] # This is not very elegant, but that's where the features are stored.
        
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


class TrainData_PreselectionNanoMLPF2(TrainData_PreselectionNanoML):
    
    def set_model(self):
        self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_pf2/KERAS_model.h5'
    




