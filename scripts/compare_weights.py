#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
from DeepJetCore.modeltools import load_model
from Layers import ScaledGooeyBatchNorm2

parser = ArgumentParser('Compare weights of two models')
parser.add_argument('model1')
parser.add_argument('model2')
parser.add_argument('--output',default='compare_weights.txt')
parser.add_argument('--verbose',action='store_true')
args = parser.parse_args()

outputstring = ''
outputstring += 'Comparing models:\n'
outputstring += '\t' + args.model1 + '\n'
outputstring += '\t' + args.model2 + '\n'

model1 = load_model(args.model1)
model2 = load_model(args.model2)

layers1 = model1.layers
layers2 = model2.layers

common_layers1 = []
common_layers2 = []
for l1 in layers1:
    for l2 in layers2:
        if l1.name == l2.name:
            common_layers1.append(l1)
            common_layers2.append(l2)

layer1_exclusive = []
for l1 in layers1:
    if l1 not in common_layers1:
        layer1_exclusive.append(l1)

layer2_exclusive = []
for l2 in layers2:
    if l2 not in common_layers2:
        layer2_exclusive.append(l2)

outputstring += f"Common layers: {len(common_layers1)}\n"
outputstring += f"Common layers: {len(common_layers2)}\n"
outputstring += f"Exclusive layers in model 1: {len(layer1_exclusive)}\n"
outputstring += f"Exclusive layers in model 2: {len(layer2_exclusive)}\n"

# Check that common layers are the same
for l1,l2 in zip(common_layers1,common_layers2):
    # check that names match
    if l1.name != l2.name:
        raise ValueError(f"Layer names do not match: {l1.name} != {l2.name}")
    if len(l1.weights) != len(l2.weights):
        outputstring += f"ERROR: Layer {l1.name} has different number of weights: {len(l1.weights)} != {len(l2.weights)}\n"
    n_weights = min(len(l1.weights),len(l2.weights))
    for i in range(n_weights):
        if not args.verbose:
            break
        if np.mean(l1.weights[i].numpy()) != np.mean(l2.weights[i].numpy()):
            outputstring += f"WARNING: Layer {l1.name} has different mean values for weight {i}: {np.mean(l1.weights[i].numpy())} != {np.mean(l2.weights[i].numpy())}\n"


# write output string
with open(args.output,'w') as f:
    f.write(outputstring)
