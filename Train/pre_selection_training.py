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

from LossLayers import LLLocalClusterCoordinates,LLNotNoiseClassifier, LLClusterCoordinates
from LossLayers import LLNeighbourhoodClassifier, CreateTruthSpectatorWeights

from Regularizers import AverageDistanceRegularizer

from DebugLayers import PlotCoordinates

from clr_callback import CyclicLR

from MetricsLayers import MLReductionMetrics
from model_blocks import pre_selection_model_full, pre_selection_staged

def pretrain_model(Inputs,
                   td,
                   debug_outdir=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)

    presel = pre_selection_model_full(orig_inputs,
                             debug_outdir,
                             trainable=True,
                             debugplots_after=1500,
                             record_metrics=True,
                             use_multigrav=True,
                             )
    
    
    # this will create issues with the output and is only needed if used in a full dim model.
    # so it's ok to pop it here for training
    presel.pop('scatterids')
    
    return DictModel(inputs=Inputs, outputs=presel)

import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    
    
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    



from config_saver import copyModules
copyModules(train.outputDir)#save the modules

verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Jan2022/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 


plot_frequency=200
cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction','*_reduction*lost*','*cluster*time'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/hit_reduction_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*reduction*hits*',#includes time
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
        output_file=train.outputDir+'/space_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='*cluster*loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/fillspace_metrics.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='ll_fill_space_loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics=['ll_edge_classifier_loss','ll_cluster_coordinates_loss','ll_not_noise_classifier_loss'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/coord_losses.html',
        record_frequency= 2,
        plot_frequency = plot_frequency,
        select_metrics='ll_cluster_coordinates_*loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]


#cb += build_callbacks(train)

#cb=[]
nbatch = 400000 
train.change_learning_rate(8e-4)

train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-5')
train.change_learning_rate(5e-5)

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
