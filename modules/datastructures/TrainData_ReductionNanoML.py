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
        #reduction model used
        self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
        print("Preselection model used : ", self.path_to_pretrained)
        #mo = load_model(self.path_to_pretrained)
        #del mo

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename=""):

        #reduction model used
        #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
        model = load_model(self.path_to_pretrained)
        #model = self.model
        #print("Check Loaded preselection model : ", self.path_to_pretrained)


        #outdict = model.output_shape
        list_outkeys = list(model.output_shape.keys())
        for l in model.output_shape.keys():
            if 'orig_' in l:
                list_outkeys.remove(l)
            elif l == "rs":
                list_outkeys.remove(l)

        #print("Otput keys considered : ", list_outkeys)

        td = TrainData_NanoML()
        tdclass = TrainData_NanoML
        td.readFromFile(filename)
        print("Reading from file : ", filename)

        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setSquaredElementsLimit(False)
        gen.setSkipTooLargeBatches(False)
        gen.setBuffer(td)

        nevents = gen.getNBatches()
        print("Nevents : ", nevents)

        rs = []
        newout = {}
        rs.append(0)

        #for i in range(5):
        for i in range(nevents):
            out = model(next(gen.feedNumpyData()))
            #print("Check keys : ", out.keys())
            rs_tmp = out['rs'].numpy()
            rs.append(rs_tmp[1])
            if i == 0:
                for k in list_outkeys:
                    newout[k] = out[k].numpy()
            else:
                for k in list_outkeys:
                    newout[k] = np.concatenate((newout[k], out[k].numpy()), axis=0)


        #td.clear()
        #gen.clear()

        #####
        # converting to DeepJetCore.SimpleArray
        rs = np.array(rs, dtype='int64')
        rs = np.cumsum(rs,axis=0)

        outSA = {}
        for k2 in list_outkeys:
            nameSA = k2
            if nameSA == "features":
                nameSA = "recHitFeatures"
            outSA[k2] = SimpleArray(newout[k2],rs,name=nameSA)

        return [outSA["features"],
                outSA["t_idx"], outSA["t_energy"], outSA["t_pos"], outSA["t_time"],
                outSA["t_pid"], outSA["t_spectator"], outSA["t_fully_contained"],
                outSA["t_rec_energy"], outSA["t_is_unique"]],[], []


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
