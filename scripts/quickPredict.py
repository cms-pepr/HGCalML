#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser('Apply a model to a (test) source sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
parser.add_argument('-i', help="input traindata file", default="/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/ml.TestDataSet/Xmas19/windowntup_99.djctd")
parser.add_argument("-e", help="event number ", default="0")

args = parser.parse_args()

import DeepJetCore
from keras.models import load_model
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
from DeepJetCore.evaluation import predict_from_TrainData
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.TrainData import TrainData
import matplotlib.pyplot as plt
from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot
from index_dicts import create_index_dict, create_feature_dict

td=TrainData()
td.readFromFile(args.i)
td.skim(int(args.e))
#td=td.split(int(args.e)+1)#get the first e+1 elements
#if int(args.e)>0:
#    td.split(1) #reduce to the last element (the e'th one)
    

model=load_model(args.inputModel, custom_objects=get_custom_objects())

predicted = predict_from_TrainData(model,td,batchsize=100000)


pred = predicted[0]
feat = td.transferFeatureListToNumpy()
rs = feat[1]
feat = feat[0]
#weights = td.transferWeightListToNumpy()
truth = td.transferTruthListToNumpy()[0]
td.clear()


print(feat.shape)
print(truth.shape)


fig = plt.figure(figsize=(10,4))
ax = [fig.add_subplot(1,2,1, projection='3d'), fig.add_subplot(1,2,2)]
    
data = create_index_dict(truth, pred, usetf=False)
feats = create_feature_dict(feat)

make_cluster_coordinates_plot(plt, ax[1], 
                              data['truthHitAssignementIdx'], #[ V ]
                              data['predBeta'],               #[ V ]
                              data['predCCoords'])



make_original_truth_shower_plot(plt, ax[0], 
                                data['truthHitAssignementIdx'],                      
                                 feats['recHitEnergy'], 
                                 feats['recHitX'],
                                 feats['recHitY'],
                                 feats['recHitZ'])


plt.tight_layout()
fig.savefig("event_"+args.e+".pdf")
plt.close()







