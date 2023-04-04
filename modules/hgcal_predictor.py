import gzip
import pickle
import numpy as np

from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
from datastructures.TrainData_NanoML import TrainData_NanoML
from datastructures.TrainData_PreselectionNanoML import TrainData_PreselectionNanoML

import os
from DeepJetCore.modeltools import load_model
from datastructures import TrainData_TrackML
import time

class HGCalPredictor():
    def __init__(self, 
            input_source_files_list, 
            training_data_collection, 
            predict_dir, 
            unbuffered=False, 
            model_path=None, 
            max_files=4, 
            inputdir=None,
            toydata=False
            ):
        self.input_data_files = []
        self.inputdir = None
        self.predict_dir = predict_dir
        self.unbuffered=unbuffered
        self.max_files = max_files
        self.toydata = toydata
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
        else:
            self.dc = DataCollection(training_data_collection)

        if inputdir is not None:
            self.inputdir = inputdir

        self.model_path = model_path
        if max_files > 0:
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
            model = load_model(model_path)

        all_data = []
        for inputfile in self.input_data_files:

            use_inputdir = self.inputdir
            if inputfile[0] == "/":
                use_inputdir = ""
            outfilename = "pred_" + os.path.basename(inputfile)
            
            print('predicting ', use_inputdir +'/' + inputfile)

            td = self.dc.dataclass()

            #also allows for inheriting classes now, like with tracks or special PU
            if not isinstance(td, TrainData_NanoML)  and type(td) is not TrainData_TrackML:
                print(td.__class__.__name__, "not yet fully supported")
            elif not isinstance(td, TrainData_PreselectionNanoML):
                print(td.__class__.__name__, "support still experimental")
            else:
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
            # the batch size must be one otherwise we need to play tricks with the row splits later on
            gen.setBatchSize(1)
            gen.setSquaredElementsLimit(False)
            gen.setSkipTooLargeBatches(False)
            gen.setBuffer(td)

            num_steps = gen.getNBatches()
            generator = gen.feedNumpyData()

            dumping_data = []
            extra_data = [] # Only used for toy data test sets

            thistime = time.time()
            for _ in range(num_steps):
                data_in = next(generator)
                if self.toydata:
                    # The toy data set has a different input shape
                    # this is only true for the testing part of the 
                    # toy data set. If predicting the training set
                    # initialize the hgcal_predictor toydata set to False
                    # The last four entries contain PU and PID
                    # we store them separately
                    predictions_dict = model(data_in[0][:-4])
                    truth_info = data_in[0][-4:]
                    extra_data.append([truth_info])
                else:
                    predictions_dict = model(data_in[0])
                for k in predictions_dict.keys():
                    # predictions_dict[k] = predictions_dict[k].numpy()
                    predictions_dict[k] = np.array(predictions_dict[k])
                features_dict = td.createFeatureDict(data_in[0])
                truth_dict = td.createTruthDict(data_in[0])
                
                dumping_data.append([features_dict, truth_dict, predictions_dict])
                
            totaltime = time.time() - thistime
            print('took approx',totaltime/num_steps,'s per endcap (also includes dict building)')

            td.clear()
            gen.clear()
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
            extrafile = os.path.splitext(outfilename)[0] + '_extra_' + '.pkl'
            if output_to_file:
                td.writeOutPredictionDict(dumping_data, self.predict_dir + "/" + outfilename)
                if self.toydata:
                    with open(os.path.join(self.predict_dir, extrafile), 'wb') as f:
                        pickle.dump(extra_data, f)
            outputs.append(outfilename)
            if not output_to_file:
                all_data.append(dumping_data)

        if output_to_file:
            with open(self.predict_dir + "/outfiles.txt", "w") as f:
                for l in outputs:
                    f.write(l + '\n')

        if not output_to_file:
            return all_data
