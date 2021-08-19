#!/usr/bin/env python3

from argparse import ArgumentParser
from Layers import RobustModel
import tensorflow.python.keras.saving.saved_model.load

import tensorflow.python.keras.saving.save
import tensorflow as tf

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection',
                    help="the training data collection. Used to infer data format and batch size.")
parser.add_argument('inputSourceFileList',
                    help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.", default="-1")
parser.add_argument("--gpu", help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered",
                    help="do not read input in memory buffered mode (for lower memory consumption on fast disks)",
                    default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False,
                    action="store_true")

args = parser.parse_args()
batchsize = int(args.b)

import imp
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
from datastructures.TrainData_NanoML import TrainData_NanoML

import os
from keras.models import load_model
from keras import backend as K
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.training.gpuTools import DJCSetGPUs
from DeepJetCore.training.training_base import custom_objects_list

inputdatafiles = []
inputdir = None

## prepare input lists for different file formats

if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection', args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)

elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file', args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ", ""))

DJCSetGPUs(args.gpu)

custom_objs = get_custom_objects()


model = tf.keras.models.load_model(args.inputModel, custom_objects=custom_objects_list)

if type(model) is not RobustModel:
    raise RuntimeError("Script only valid for RobustModel")


# model = tf.saved_model.load(args.inputModel)
# print(type(model))


dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of', batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p ' + args.outputDir)


for inputfile in inputdatafiles:

    print('predicting ', inputdir + "/" + inputfile)


    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir = ""
    outfilename = "pred_" + os.path.basename(inputfile)

    td = dc.dataclass()

    if type(td) is not TrainData_NanoML:
        raise RuntimeError("TODO: make sure this works for other traindata formats")


    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir + "/" + inputfile)
        else:
            td.readFromFileBuffered(use_inputdir + "/" + inputfile)
    else:
        print('converting ' + inputfile)
        td.readFromSourceFile(use_inputdir + "/" + inputfile, dc.weighterobjects, istraining=False)

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    print('batch size', batchsize)
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    num_steps = gen.getNBatches()
    generator = gen.feedNumpyData()

    dumping_data = []

    for i in range(num_steps):
        data_in = next(generator)
        predictions_dict = model.call_with_dict_as_output(data_in[0])
        features_dict = td.createFeatureDict(data_in[0][0])
        truth_dict = td.createTruthDict(data_in[1][0])

        dumping_data.append([features_dict, truth_dict, predictions_dict])


    td.clear()
    gen.clear()

    td.writeOutPredictionDict(dumping_data, args.outputDir + "/" + outfilename)
    outputs.append(outfilename)

with open(args.outputDir + "/outfiles.txt", "w") as f:
    for l in outputs:
        f.write(l + '\n')





