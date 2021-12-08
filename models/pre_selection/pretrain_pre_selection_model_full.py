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

from datastructures import TrainData_NanoML


#from tensorflow.keras import Model
from Layers import RobustModel

from model_blocks import create_outputs, noise_pre_filter, first_coordinate_adjustment, reduce_indices
from tensorflow.keras.layers import Dense, Concatenate
from Layers import ProcessFeatures, CastRowSplits
from DeepJetCore.DJCLayers import SelectFeatures

from lossLayers import LLLocalClusterCoordinates,LLNotNoiseClassifier, LLClusterCoordinates
from lossLayers import LLNeighbourhoodClassifier, CreateTruthSpectatorWeights

from Regularizers import AverageDistanceRegularizer
from initializers import EyeInitializer

from debugLayers import PlotCoordinates

from clr_callback import CyclicLR


from model_blocks import pre_selection_model_full

def pretrain_model(Inputs,
                   td,
                   debug_outdir=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)

    out = pre_selection_model_full(orig_inputs,
                             debug_outdir,
                             reduction_threshold=0.5,
                             use_edges=True,
                             trainable=True,
                             debugplots_after=1500,
                             omit_reduction=False
                             )

    #same outputs as the model block
    model_outputs = [(k,out[k]) for k in out.keys()]

    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    
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
                                  backup_after_batches=2500,
                                  additional_callbacks=cb)

print('reducing learning rate to 1e-3')
learningrate = 1e-4
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
                                  backup_after_batches=2500,
                                  additional_callbacks=cb)

print('reducing learning rate to 1e-4')
learningrate = 5e-5
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