from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
from DeepJetCore.modeltools import load_model
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

from datastructures.TrainData_NanoML import TrainData_NanoML
from DeepJetCore.dataPipeline import TrainDataGenerator
from model_blocks import pre_selection_model_full, pre_selection_staged

class TrainData_ReductionNanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.include_tracks = False
        self.cp_plus_pu_mode = False


    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename=""):

        #reduction model used
        path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
        model = load_model(path_to_pretrained)

        td = TrainData_NanoML()
        tdclass = TrainData_NanoML
        td.readFromFile(filename)

        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setSquaredElementsLimit(False)
        gen.setSkipTooLargeBatches(False)
        gen.setBuffer(td)

        nevents = gen.getNBatches()
        #generator = gen.feedNumpyData()

        rs = []
        rs.append(0)
        feat_arr = []
        t_idx_arr = []
        t_energy_arr = []
        t_pos_arr = []
        t_time_arr = []
        t_pid_arr = []
        t_spectator_arr = []
        t_fully_contained_arr = []
        t_rec_energy_arr = []
        t_is_unique_arr = []

        for i in range(nevents):
            #data_in = next(generator)
            out = model(next(gen.feedNumpyData()))
            #print("Check keys : ", out.keys())
            rs_tmp = out['rs'].numpy()
            rs.append(rs_tmp[1])
            if i == 0:
                feat_arr = out['features'].numpy()
                t_idx_arr = out['t_idx'].numpy()
                t_energy_arr = out['t_energy'].numpy()
                t_pos_arr = out['t_pos'].numpy()
                t_time_arr = out['t_time'].numpy()
                t_pid_arr = out['t_pid'].numpy()
                t_spectator_arr = out['t_spectator'].numpy()
                t_fully_contained_arr = out['t_fully_contained'].numpy()
                t_rec_energy_arr = out['t_rec_energy'].numpy()
                t_is_unique_arr = out['t_is_unique'].numpy()
            else:
                 feat_arr = np.concatenate((feat_arr, out['features'].numpy()), axis=0)
                 t_idx_arr = np.concatenate((t_idx_arr, out['t_idx'].numpy()), axis=0)
                 t_energy_arr = np.concatenate((t_energy_arr,out['t_energy'].numpy()), axis=0)
                 t_pos_arr = np.concatenate((t_pos_arr,out['t_pos'].numpy()), axis=0)
                 t_time_arr = np.concatenate((t_time_arr,out['t_time'].numpy()), axis=0)
                 t_pid_arr = np.concatenate((t_pid_arr,out['t_pid'].numpy()), axis=0)
                 t_spectator_arr = np.concatenate((t_spectator_arr,out['t_spectator'].numpy()), axis=0)
                 t_fully_contained_arr = np.concatenate((t_fully_contained_arr,out['t_fully_contained'].numpy()), axis=0)
                 t_rec_energy_arr = np.concatenate((t_rec_energy_arr,out['t_rec_energy'].numpy()), axis=0)
                 t_is_unique_arr = np.concatenate((t_is_unique_arr,out['t_is_unique'].numpy()), axis=0)


        #td.clear()
        #gen.clear()

        #####
        # converting to DeepJetCore.SimpleArray
        rs = np.array(rs, dtype='int64')
        rs = np.cumsum(rs,axis=0)

        feat_arr = np.array(feat_arr, dtype='float32')
        farr = SimpleArray(feat_arr,rs,name="recHitFeatures")

        t_idx_arr = np.array(t_idx_arr, dtype='int32')
        t_idx = SimpleArray(t_idx_arr,rs,name="t_idx")
        t_energy_arr = np.array(t_energy_arr, dtype='float32')
        t_energy = SimpleArray(t_energy_arr,rs,name="t_energy")
        t_pos_arr = np.array(t_pos_arr, dtype='float32')
        t_pos = SimpleArray(t_pos_arr,rs,name="t_pos")
        t_time_arr = np.array(t_time_arr, dtype='float32')
        t_time = SimpleArray(t_time_arr,rs,name="t_time")
        t_pid_arr = np.array(t_pid_arr, dtype='int32')
        t_pid = SimpleArray(t_pid_arr,rs,name="t_pid")
        t_spectator_arr = np.array(t_spectator_arr, dtype='float32')
        t_spectator = SimpleArray(t_spectator_arr,rs,name="t_spectator")
        t_fully_contained_arr = np.array(t_fully_contained_arr, dtype='float32')
        t_fully_contained = SimpleArray(t_fully_contained_arr,rs,name="t_fully_contained")
        t_rec_energy_arr = np.array(t_rec_energy_arr, dtype='float32')
        t_rec_energy = SimpleArray(t_rec_energy_arr,rs,name="t_rec_energy")
        t_is_unique_arr = np.array(t_is_unique_arr, dtype='int32')
        t_is_unique = SimpleArray(t_is_unique_arr,rs,name="t_is_unique")

        return [farr,
                t_idx, t_energy, t_pos, t_time,
                t_pid, t_spectator, t_fully_contained,
                t_rec_energy, t_is_unique],[], []


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
