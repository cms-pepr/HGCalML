"""
Training script
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

# import wandb very early
from DeepJetCore.wandb_interface import wandb_wrapper as wandb
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras import Model

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient, ScalarMultiply

import training_base_hgcal
from Layers import LLRegulariseGravNetSpace, DummyLayer
from Layers import ScaledGooeyBatchNorm2
from Layers import MixWhere
from Layers import ProcessFeatures
from Layers import RaggedGravNet
from Layers import PlotCoordinates
from Layers import DistanceWeightedMessagePassing, TranslationInvariantMP
from Layers import LLFillSpace
from Layers import LLExtendedObjectCondensation
from Layers import LLExtendedObjectCondensation2
from Layers import LLExtendedObjectCondensation3
from Layers import LLExtendedObjectCondensation4
from Layers import DictModel
from Layers import RaggedGlobalExchange
from Layers import SphereActivation
from Layers import Multi
from Layers import ShiftDistance
# from Layers import LLRegulariseGravNetSpace
from Layers import SplitOffTracks, ConcatRaggedTensors
from Regularizers import AverageDistanceRegularizer
from Initializers import EyeInitializer
from model_blocks import tiny_pc_pool, condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from model_tools import apply_weights_from_path
from model_blocks import random_sampling_unit, random_sampling_block2
from noise_filter import noise_filter
from callbacks import plotClusteringDuringTraining
from callbacks import plotClusterSummary
from callbacks import NanSweeper, DebugPlotRunner
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

from model_blocks import tree_condensation_block

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")

if not train.args.no_wandb:
    wandb.init(
        project=train.args.wandb_project,
        config={},
    )
    wandb.save(sys.argv[0]) # Save python file
else:
    wandb.active=False
    
train = training_base_hgcal.HGCalTraining(parser=parser)


PLOT_FREQUENCY = 200

###############################################################################
### Define Model ##############################################################
###############################################################################


def config_model(Inputs, td, debug_outdir=None, plot_debug_every=PLOT_FREQUENCY, 
                 #check_keys is just for debugging
                 check_keys = False):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False, new_prime=True)
    
    pre_processed['features'] = ProcessFeatures()(pre_processed['features'])    

    out, graph = tree_condensation_block(pre_processed, 
                                  
                            #the latter overwrites the default arguments such that it is in training mode
                            debug_outdir=debug_outdir, plot_debug_every=plot_debug_every,
                            trainable = True,
                            record_metrics = True,
                            produce_output = check_keys)
    
    # check if there are keys in out missing that are in pre_processed, just for debugging
    if check_keys:
        
        not_there = []
        for key in pre_processed.keys():
            if key not in out.keys():
                print(f"Key {key} missing in out")
                not_there.append(key)
    
        if len(not_there):
            print("Keys in pre_processed:")
            print(pre_processed.keys())
            print("Keys in out:")
            print(out.keys())
            print('Keys missing in out:')
            print(not_there)
            raise ValueError("Keys missing in out")
    
    return Model(inputs=Inputs, outputs=graph)


###############################################################################
### Set up training ###########################################################
###############################################################################



if not train.modelSet():
    train.setModel(
        config_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        plot_debug_every=PLOT_FREQUENCY,
        )
    train.setCustomOptimizer(tf.keras.optimizers.Adam())#clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################

# None

###############################################################################
### Actual Training ###########################################################
###############################################################################

#set the learning rate to 1e-2

train.change_learning_rate(1e-2)
train.trainModel(
        nepochs=2,
        batchsize=90000,
        add_progbar=True,
        additional_callbacks=[]
        )

train.change_learning_rate(1e-3)
train.trainModel(
        nepochs=20,
        batchsize=90000,
        add_progbar=True,
        additional_callbacks=[]
        )
