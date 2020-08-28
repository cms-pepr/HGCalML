#!/usr/bin/env python3

from argparse import ArgumentParser
import os

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
args = parser.parse_args()

infile = args.inputFile
outdir=args.outputDir+"/"
os.system('mkdir -p '+outdir)


from datastructures import TrainData_window
from DeepJetCore.compiled.c_simpleArray import simpleArray

td = TrainData_window()
td.readFromFile(infile)


feat,rs = td.transferFeatureListToNumpy()
nrs = int(rs[-1])
print(nrs)
rs=rs[:nrs]
truth,_ = td.transferTruthListToNumpy()

print(feat.shape)
print(truth.shape)

farr = simpleArray()
farr.createFromNumpy(feat, rs)

truth[:,1] = truth[:,16]

print(truth.shape)
tarr = simpleArray()
tarr.createFromNumpy(truth, rs)

#tarr.cout()

td_out = TrainData_window()
td_out._store([farr],[tarr],[])

td_out.writeToFile(outdir+infile)