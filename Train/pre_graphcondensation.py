'''


Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np

from datastructures import TrainData_NanoML

#from tensorflow.keras import Model
from Layers import DictModel
from GraphCondensationLayers import CreateMultiAttentionGraphTransitions, PushUp, PullDown, SelectUp, Mix, UpDenseDown, DenseOnUp

from GravNetLayersRagged import RaggedGravNet, DistanceWeightedMessagePassing, ScaledGooeyBatchNorm, SortAndSelectNeighbours, EdgeConvStatic
from LayersRagged import RaggedGlobalExchange

from LossLayers import LLFullObjectCondensation, LLValuePenalty

from DebugLayers import PlotCoordinates

from model_blocks import  pre_graph_condensation, condition_input, create_outputs, intermediate_graph_condensation
from tensorflow.keras.layers import Concatenate, Dense, Add


plot_frequency= 3*12 #every 3 minutes #360 #200 #600#every 15 minutes approx
record_frequency = 5

batchnorm_options={
    'viscosity': .1,
    'fluidity_decay': 1e-4,
    'max_viscosity': .99,
    'learn': True
    }

#loss options:
loss_options={
    'energy_loss_weight': .25,
    'q_min': 1.5,
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0.0,
    'too_much_beta_scale': 1e-5 ,
    'position_loss_weight':1e-5,
    'timing_loss_weight':0.1,
    'beta_loss_scale':2.,
    'beta_push': 0#0.01 #push betas gently up at low values to not lose the gradients
    }

def pretrain_model(Inputs,
                   td, 
                   publish,
                   debugplots_after=record_frequency*plot_frequency, 
                   debug_outdir=None):

    
    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    orig_inputs = condition_input(orig_inputs, no_scaling = True)
        
    print('orig_inputs',orig_inputs.keys())
    
    precond,trans = pre_graph_condensation(orig_inputs,
                             debug_outdir,
                             trainable=True,
                             debugplots_after=debugplots_after,
                             record_metrics=True,
                             K_loss = 48,
                             publish = publish,
                             dynamic_spectators = True,
                             flatten = True
                             )
    
    #that's it for pre-training
    precond.update(trans)
    
    return DictModel(inputs=Inputs, outputs=precond)
    

import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Jan2023/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   publish = publishpath,
                   debug_outdir=train.outputDir+'/intplots')
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam(clipnorm=10.))
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
    #start somewhere
    #from model_tools import apply_weights_from_path
    #import os
    #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    #train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    



verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback



cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/hit_reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*hits*','*lost*'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/batch_norm.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*_mean_correction','*_var_correction','*_mean', '*_var'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics='*_loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/oc_losses.html',
    #    record_frequency = record_frequency ,
    #    plot_frequency = plot_frequency,
    #    select_metrics='FullOCLoss*',
    #    publish=publishpath #no additional directory here (scp cannot create one)
    #    ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]


#cb += build_callbacks(train)

#cb=[]

#train.change_learning_rate(1e-3)

#train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

nbatch = 80000 #//2
nbatch2 = nbatch #120000

#train.change_learning_rate(1e-3)
#train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(1e-3)

train.trainModel(nepochs=3,batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(5e-5)

train.trainModel(nepochs=3+4,batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(1e-5)


train.trainModel(nepochs=100,batchsize=nbatch2,additional_callbacks=cb)
