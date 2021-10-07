#!/usr/bin/env python3

import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser = ArgumentParser('plot the viscosity of a GooeyBatchNorm layer as a function of the number of batches')
parser.add_argument('start_viscosity',type=float)
parser.add_argument('max_viscosity',type=float)
parser.add_argument('fluidity_decay',type=float)
args = parser.parse_args()

print('start at ', args.start_viscosity, 'end at', args.max_viscosity, 'decay',args.fluidity_decay) 

assert args.max_viscosity<= 1 and args.start_viscosity < 1 and args.fluidity_decay < 1 and args.fluidity_decay> 0

epsilonvisc = args.max_viscosity-(args.max_viscosity - args.start_viscosity)/1000.

def visc(viscosity, max_viscosity, fluidity_decay):
    return viscosity + (max_viscosity - viscosity)*fluidity_decay
    
counts = []
ovisc=[]
count = 0
tvisc = args.start_viscosity
#quick check, get range
while( tvisc < epsilonvisc):
    counts.append(count)
    tvisc = visc(tvisc, args.max_viscosity, args.fluidity_decay)
    ovisc.append(tvisc)
    count+=1
    
plt.plot(counts,ovisc)
plt.xlabel('# of batches')
plt.ylabel('viscosity')
plt.show()