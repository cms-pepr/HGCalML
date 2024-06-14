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

from model_blocks import tree_condensation_block, tree_condensation_block2, post_tree_condensation_push, double_tree_condensation_block
from Layers import PlotGraphCondensationEfficiency

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")

# get the args so far but ignore the other args to pass them on
# this prevents errors with wandb in layers

pre_args,_ = parser.parse_known_args() 

if not pre_args.no_wandb:
    wandb.init(
        project=pre_args.wandb_project,
        config={},
    )
    wandb.wandb().save(sys.argv[0]) # Save python file
else:
    wandb.active=False
#parses the rest of the arguments
train = training_base_hgcal.HGCalTraining(parser=parser)

PLOT_FREQUENCY = 600

###############################################################################
### Define Model ##############################################################
###############################################################################

'''
Total params: 8,652
Trainable params: 8,646
Non-trainable params: 6

in presel model

>> GravNet_tree_condensation_block_net
Total params: 351,026
Trainable params: 342,361
Non-trainable params: 8,665
'''

def config_model(Inputs, td, debug_outdir=None, plot_debug_every=PLOT_FREQUENCY, 
                 #check_keys is just for debugging
                 check_keys = False):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step, common to all models #############################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False, new_prime=True)
    
    pre_processed['features'] = ProcessFeatures()(pre_processed['features'])    

    ###########################################################################
    ### Model definition ######################################################
    ###########################################################################

    out, graph, sels = double_tree_condensation_block(pre_processed,
                             debug_outdir=debug_outdir, plot_debug_every=plot_debug_every,
                             trainable = True,
                             record_metrics = True, push_heads = 8)
    
    graph.update(out) #just so everything is connected
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

train.change_learning_rate(2e-4)
train.trainModel(
        nepochs=2,
        batchsize=120000,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=[],
        collect_gradients = 4 #average out more gradients
        )

# loop through model layers and turn  batch normalisation to fixed
def fix_batchnorm(m):
    for layer in m.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

#apply to all models
train.applyFunctionToAllModels(fix_batchnorm)
#recompile
train.compileModel(learningrate=1e-3)
print('entering second training phase')

train.change_learning_rate(1e-4)
train.trainModel(
        nepochs=20,
        batchsize=120000,
        add_progbar=False,
        additional_callbacks=[],
        collect_gradients = 4
        )