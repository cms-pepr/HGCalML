#!/usr/bin/env python3

from argparse import ArgumentParser
from hgcal_predictor import HGCalPredictor
import setGPU

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('inputModel')
parser.add_argument('data_collection',
                    help="Data collection file in djcdc format from which to pick files to run inference on. "
                         "You can use valsamples.djcdc in training folder as a starter.")
parser.add_argument('--data_dir',
                    help="Directory in which data (in form of djctd format files) is located. (default will be read from data collection)",
                    default=None)
parser.add_argument('output_dir', help="will be created if it doesn't exist.")
parser.add_argument("--unbuffered",
                    help="do not read input in memory buffered mode (for lower memory consumption on fast disks)",
                    default=False, action="store_true")

parser.add_argument("--max_files", help="Limit number of files", default=-1)

args = parser.parse_args()


HGCalPredictor(args.data_collection, args.data_collection, args.output_dir, inputdir=args.data_dir, unbuffered=False, max_files=int(args.max_files)).predict(model_path=args.inputModel)





