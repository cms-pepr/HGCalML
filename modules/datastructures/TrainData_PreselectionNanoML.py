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

class TrainData_PreselectionNanoML(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.include_tracks = False
        self.cp_plus_pu_mode = False
        #preselection model used
        self.path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_may22/KERAS_model.h5'

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename=""):

        model = load_model(self.path_to_pretrained)
        print("Loaded preselection model : ", self.path_to_pretrained)


        #outdict = model.output_shape
        list_outkeys = list(model.output_shape.keys())
        for l in model.output_shape.keys():
            if 'orig_' in l:
                list_outkeys.remove(l)
            elif l == "row_splits":
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
        #print("Nevents : ", nevents)

        rs = []
        newout = {}
        rs.append(0)

        for i in range(nevents):
            out = model(next(gen.feedNumpyData()))
            rs_tmp = out['row_splits'].numpy()
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
                outSA["rechit_energy"],
                outSA["t_idx"], outSA["t_energy"], outSA["t_pos"], outSA["t_time"],
                outSA["t_pid"], outSA["t_spectator"], outSA["t_fully_contained"],
                outSA["t_rec_energy"], outSA["t_is_unique"]],[], []


    def interpretAllModelInputs(self, ilist, returndict=True):
        #taken from TrainData_NanoML since it is similar, check for changes there
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
            'rechit_energy':[2],
            't_idx':ilist[4],
            't_energy':ilist[6],
            't_pos':ilist[8],
            't_time':ilist[10],
            't_pid':ilist[12],
            't_spectator':ilist[14],
            't_fully_contained':ilist[16],
            't_rec_energy':ilist[18],
            't_is_unique':ilist[20],
            'row_splits':ilist[1]
            }
        #keep length check for compatibility
        if len(ilist)>16:
            out['t_rec_energy'] = ilist[16]
        if len(ilist)>18:
            out['t_is_unique'] = ilist[18]
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
