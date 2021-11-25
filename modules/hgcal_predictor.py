
from argparse import ArgumentParser
from Layers import RobustModel
import tensorflow.python.keras.saving.saved_model.load

import tensorflow.python.keras.saving.save
import tensorflow as tf
import imp
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
from datastructures.TrainData_NanoML import TrainData_NanoML

import os
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.training.gpuTools import DJCSetGPUs
from DeepJetCore.training.training_base import custom_objects_list
from datastructures.TrainData_TrackML import TrainData_TrackML


class HGCalPredictor():
    def __init__(self, input_source_files_list, training_data_collection, predict_dir, batch_size=1, unbuffered=False, model_path=None, max_files=4, inputdir=None):
        self.input_data_files = []
        self.inputdir = None
        self.predict_dir = predict_dir
        self.batch_size = batch_size
        self.unbuffered=unbuffered
        self.max_files = max_files
        print("Using HGCal predictor class")

        ## prepare input lists for different file formats

        if input_source_files_list[-6:] == ".djcdc":
            print('reading from data collection', input_source_files_list)
            predsamples = DataCollection(input_source_files_list)
            self.inputdir = predsamples.dataDir
            for s in predsamples.samples:
                self.input_data_files.append(s)

        elif input_source_files_list[-6:] == ".djctd":
            self.inputdir = os.path.abspath(os.path.dirname(input_source_files_list))
            infile = os.path.basename(input_source_files_list)
            self.input_data_files.append(infile)
        else:
            print('reading from text file', input_source_files_list)
            self.inputdir = os.path.abspath(os.path.dirname(input_source_files_list))
            with open(input_source_files_list, "r") as f:
                for s in f:
                    self.input_data_files.append(s.replace('\n', '').replace(" ", ""))

        self.dc = None
        if input_source_files_list[-6:] == ".djcdc" and not training_data_collection[-6:] == ".djcdc":
            self.dc = DataCollection(input_source_files_list)
            if self.batch_size < 1:
                self.batch_size = 1
            print('No training data collection given. Using batch size of', self.batch_size)
        else:
            self.dc = DataCollection(training_data_collection)

        if inputdir is not None:
            self.inputdir = inputdir

        self.model_path = model_path
        self.input_data_files = self.input_data_files[0:min(max_files, len(self.input_data_files))]

    def predict(self, model=None, model_path=None, output_to_file=True):
        if model_path==None:
            model_path = self.model_path

        if model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError('Model file not found')

        assert model_path is not None or model is not None

        outputs = []
        if output_to_file:
            os.system('mkdir -p ' + self.predict_dir)

        if model is None:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects_list)

        all_data = []
        for inputfile in self.input_data_files:

            print('predicting ', self.inputdir + "/" + inputfile)

            use_inputdir = self.inputdir
            if inputfile[0] == "/":
                use_inputdir = ""
            outfilename = "pred_" + os.path.basename(inputfile)

            td = self.dc.dataclass()

            if type(td) is not TrainData_NanoML  and type(td) is not TrainData_TrackML:
                raise RuntimeError("TODO: make sure this works for other traindata formats")

            if inputfile[-5:] == 'djctd':
                if self.unbuffered:
                    td.readFromFile(use_inputdir + "/" + inputfile)
                else:
                    td.readFromFileBuffered(use_inputdir + "/" + inputfile)
            else:
                print('converting ' + inputfile)
                td.readFromSourceFile(use_inputdir + "/" + inputfile, self.dc.weighterobjects, istraining=False)

            gen = TrainDataGenerator()
            print(self.batch_size)
            self.dc.setBatchSize(1)
            self.batch_size = 1
            if self.batch_size < 1:
                self.batch_size = self.dc.getBatchSize()
            gen.setBatchSize(self.batch_size)
            gen.setSquaredElementsLimit(self.dc.batch_uses_sum_of_squares)
            gen.setSkipTooLargeBatches(False)
            gen.setBuffer(td)

            num_steps = gen.getNBatches()
            generator = gen.feedNumpyData()

            dumping_data = []

            for i in range(num_steps):
                data_in = next(generator)
                predictions_dict = model.call_with_dict_as_output(data_in[0], numpy=True)
                features_dict = td.createFeatureDict(data_in[0])
                truth_dict = td.createTruthDict(data_in[0])

                print("Num rechits", len(features_dict['recHitX']), data_in[0][1])

                dumping_data.append([features_dict, truth_dict, predictions_dict])

            td.clear()
            gen.clear()
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
            if output_to_file:
                td.writeOutPredictionDict(dumping_data, self.predict_dir + "/" + outfilename)
            outputs.append(outfilename)
            if not output_to_file:
                all_data.append(dumping_data)
        if output_to_file:
            with open(self.predict_dir + "/outfiles.txt", "w") as f:
                for l in outputs:
                    f.write(l + '\n')

        if not output_to_file:
            return all_data
