#!/usr/bin/env python3

from argparse import ArgumentParser
parser = ArgumentParser('Inspect all batchnorm layers of a model')
parser.add_argument('inputModel')
args = parser.parse_args()

from DeepJetCore.modeltools import load_model
from Layers import ScaledGooeyBatchNorm2

m = load_model(args.inputModel)

for l in m.layers:
    if isinstance(l, ScaledGooeyBatchNorm2):
        ws = l.weights
        print('\n\nLayer',l.name,':\n')
        for w in ws:
            print(w.name,w.shape,'\n',w.numpy())

