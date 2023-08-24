"""
Script to check deposited vs true energy in events
"""
import pdb
import gzip
import os
from argparse import ArgumentParser
import pickle
import numpy as np
import matplotlib.pyplot as plt

from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
from datastructures.TrainData_NanoML import TrainData_NanoML
from datastructures.TrainData_PreselectionNanoML import TrainData_PreselectionNanoML

parser = ArgumentParser('DataValidation')
parser.add_argument('data_collection',
        help="Data collection file in djcdc format from which to pick files to run inference on.\
                You can use valsamples.djcdc in training folder as a starter.")
parser.add_argument('output_dir', help="will be created if it doesn't exist.")
parser.add_argument('--event', help="if given, writes last event")
args = parser.parse_args()

outputdir = args.output_dir
if not os.path.isdir(outputdir):
    print(f"Output directory '{outputdir}' not found, creating it")
    os.mkdir(outputdir)
dc = DataCollection(args.data_collection)

inputfiles = [dc.dataDir + sample for sample in dc.samples]

deposited = []
truth = []
# File loop
for inputfile in inputfiles:
    td = dc.dataclass()
    td.readFromFileBuffered(inputfile)

    gen = TrainDataGenerator()
    gen.setBatchSize(1)
    gen.setSquaredElementsLimit(False)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    num_steps = gen.getNBatches()
    generator = gen.feedNumpyData()

    for i in range(10):
        data = next(generator)
        pdb.set_trace()
        features_dict = td.createFeatureDict(data[0])
        truth_dict = td.createTruthDict(data[0])
        deposited.append(truth_dict['t_rec_energy'])
        truth.append(truth_dict['truthHitAssignedEnergies'])

print(features_dict.keys())
print(truth_dict.keys())

if args.featureDict:
    with gzip.open(os.path.join(args.output_dir, args.event), 'wb') as f:
        event = {
            'features': features_dict,
            'truth': truth_dict,
            }
        pickle.dump(event, f)

deposited_energies = np.concatenate(deposited)
truth_energies = np.concatenate(truth)
ratios = deposited_energies / truth_energies

print("Zero truth: ", np.sum(truth_energies == 0), "out of", len(truth_energies))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
ax.hist(ratios, bins=30)
ax.set_title("t_rec_energy / truthHitAssignedEnergies", fontsize=30)
ax.set_xlabel("Ratio - i.e. perfect correction factor", fontsize=20)
# ax.set_yscale('log')
ax.set_xlim((-0.5, 3))
fig.savefig(os.path.join(outputdir, 'correction.jpg'))

