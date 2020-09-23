#!/usr/bin/env python3

from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
args = parser.parse_args()

from DeepJetCore.modeltools import loadModelAndFixLayers
import tensorflow as tf
model = loadModelAndFixLayers(args.inputModel,"")
model.save(
    args.outputDir, overwrite=True, include_optimizer=False, save_format="tf",
    signatures=None, options=None
)


print('inputs:')
for l in model.layers:
    if isinstance(l, tf.keras.layers.InputLayer):
        print('name:',l.name, 'dtype:',l.dtype,'shape',l.output_shape)
