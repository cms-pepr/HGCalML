#!/usr/bin/env python3



from argparse import ArgumentParser
from LossLayers import LossLayerBase

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputFile')
args = parser.parse_args()


from DeepJetCore.modeltools import load_model

m=load_model(args.inputFile)

for l in m.layers:
    if isinstance(l, LossLayerBase):
        print('deactivating layer',l)
        l.active=False
        
m.save(args.outputFile)