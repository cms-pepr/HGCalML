'''

Compatible with the dataset here:
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoML
and (soon)
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoMLTracks

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/OctProd/NanoML

not compatible with datasets before end of October 2021

'''
from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import Reshape,BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import ElementScaling,MultiBackScatterOrGather,NeighbourGroups, AccumulateNeighbours,RecalcDistances,WeightFeatures,WeightedNeighbourMeans,DownSample, CreateIndexFromMajority, ProcessFeatures, SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from initializers import EyeInitializer
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from datastructures import TrainData_NanoML

from plotting_callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining, plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR
from lossLayers import LLNoiseClassifier,LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs
from GravNetLayersRagged import MultiBackScatter,LNC2,EdgeCreator, EdgeSelector, GroupScoreFromEdgeScores,NoiseFilter,LNC,ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackScatter, KNN, MessagePassing, RobustModel
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter, first_coordinate_adjustment
from Regularizers import AverageDistanceRegularizer



from debugLayers import PlotCoordinates

'''

make this about coordinate shifts


rewrite LNC2 layer to just return the selection indices, backscatter/gather info and directed neighbour indices
then write new selection layers with sum/flat option (and just use the accumulate one)


'''


td = TrainData_NanoML()


def pretrain_model(Inputs,
                  debug_outdir=None
                  ):
    # Input preprocessing below. Not much to change here

    feat, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(
        Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)

    t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([t_spectator, t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0, 1, name="energy_selector")(feat)
    time = SelectFeatures(8, 9)(feat_norm)
    orig_coords = SelectFeatures(5, 8)(feat_norm)

    x = feat_norm
    sel_gidx = gidx_orig

    allfeat = [x]
    backgatheredids = []
    scatterids = []
    backgathered = []
    backgathered_coords = []
    energysums = []

    # here the actual network starts

    ############## Keep this part to reload the noise filter with pre-trained weights for other trainings
    
    #this takes O(200ms) for 100k hits
    coords,nidx,dist, x = first_coordinate_adjustment(
        orig_coords, x, energy, rs, t_idx, 
        debug_outdir,
        trainable=True,
        name='first_coords',
        debugplots_after=240
        )
    
    dist = LLLocalClusterCoordinates(
            print_loss=True,
            scale=1.
            )([dist, nidx, t_idx, t_spectator_weight])
    
    
    
    model_outputs = [('dist', dist),
                     ('row_splits', row_splits)]

    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(pretrain_model,debug_outdir=train.outputDir+'/intplots')
    train.setCustomOptimizer(tf.keras.optimizers.Adam(
        #larger->slower forgetting
        #beta_1: linear
        #beta_2: sq
        #make it slower for our weird fluctuating batches
        #beta_1=0.99, #0.9
        #beta_2=0.99999 #0.999
        #clipnorm=0.001
        #amsgrad=True,
        #epsilon=1e-2
        ))

    train.compileModel(learningrate=1e-4,
                       loss=None)


verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


cb = []



#cb += build_callbacks(train)

#cb=[]
learningrate = 1e-2
nbatch = 300000 #why not


for l in train.keras_model.layers:
    if "knn"  in l.name:
        print(l.name)
        l.radius = 1
    if hasattr(l, 'layers'):
        for ll in l.layers:
            if "knn"  in ll.name:
                print(ll.name)
                ll.radius = 1

train.compileModel(learningrate=learningrate, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)

print('reducing learning rate to 1e-3')
learningrate = 1e-3
nbatch = 300000 #why not

train.compileModel(learningrate=learningrate, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)

print('reducing learning rate to 1e-4')
learningrate = 1e-4
nbatch = 300000 #why not

train.compileModel(learningrate=learningrate, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=10,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)