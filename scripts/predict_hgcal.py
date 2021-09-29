#!/usr/bin/env python3

from argparse import ArgumentParser
from hgcal_predictor import HGCalPredictor

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



HGCalPredictor(args.inputSourceFileList, args.trainingDataCollection, args.outputDir, batchsize, unbuffered=False).predict(model_path=args.inputModel)





