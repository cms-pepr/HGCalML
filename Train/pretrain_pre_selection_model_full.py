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
from Layers import DictModel, PrintMeanAndStd

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

from MetricsLayers import MLReductionMetrics
from model_blocks import pre_selection_model_full, pre_selection_staged

def pretrain_model(Inputs,
                   td,
                   debug_outdir=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)

    presel = pre_selection_model_full(orig_inputs,
                             debug_outdir,
                             reduction_threshold=0.75,#high threshold
                             use_edges=True,
                             trainable=True,
                             debugplots_after=1500,
                             omit_reduction=False,
                             record_metrics=True,
                             use_multigrav=True,
                             )
    
    
    # this will create issues with the output and is only needed if used in a full dim model.
    # so it's ok to pop it here for training
    presel.pop('scatterids')
    
    return DictModel(inputs=Inputs, outputs=presel)

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



from configSaver import copyModules
copyModules(train.outputDir)#save the modules

verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Jan2022/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 


plot_frequency=100
cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*reduction*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/knn_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*slicing*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/noise_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*noise*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/space_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*cluster*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/regularizers.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*regularizer*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/multiattention.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*coord_add*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]


#cb += build_callbacks(train)

#cb=[]
nbatch = 400000 
train.change_learning_rate(2e-4)

train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-5')
train.change_learning_rate(5e-5)

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
