'''
Noise filter
Replaces preselection model, but is much simpler.

'''
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
from datastructures import TrainData_NanoML
from Layers import DictModel
import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

from model_blocks import  pre_selection_model
from noise_filter import noise_filter

K=12 #12

def noise_model(Inputs, td, debugplots_after=600, debug_outdir=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
 
    print('orig_inputs',orig_inputs.keys())
    
    filtered = noise_filter(
            orig_inputs,
            debug_outdir,
            trainable=True,
            debugplots_after=5*debugplots_after,
            record_metrics=True,
            K=K)
    
    print('noise_filtered', filtered.keys())
    # this will create issues with the output and 
    # is only needed if used in a full dim model.
    # so it's ok to pop it here for training
    filtered.pop('scatterids')
    
    return DictModel(inputs=Inputs, outputs=filtered)

train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(noise_model, 
            td = train.train_data.dataclass(), 
            debug_outdir=train.outputDir+'/intplots')
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()
    

# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))
publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/June2022/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 
publishpath = None
plot_frequency=600
cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction','*amb_truth_fraction'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/hit_reduction_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics=['*reduction*hits*','*_reduction*lost*'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/noise_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics=['*noise*accuracy','*noise*loss','*noise*reduction'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/time.html',
        record_frequency= 10,#doesn't change anyway
        plot_frequency = plot_frequency,
        select_metrics='*time*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*_loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    ]

#cb=[]
nbatch = 200000 
train.change_learning_rate(1e-4)
train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 5e-5')
train.change_learning_rate(5e-5)
nbatch = 200000 

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
