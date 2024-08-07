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
from Layers import PlotCoordinates, ZerosLike
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
from Layers import ScaledGooeyBatchNorm2, CastRowSplits
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

from model_blocks import GravNet_plus_TEQMP
from Layers import PlotGraphCondensationEfficiency
from datastructures import TrainData_PreSnowflakeNanoML

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

PLOT_FREQUENCY = 2 # a bit more than one 2 hours

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


NEIGHBOURS = [128,128,128]
DSHAPE = 64
DENSE_ACTIVATION = 'elu'
LOSS_IMPLEMENTATION = "hinge"
N_CLUSTER_SPACE_COORDINATES = 4
DISTANCE_SCALE = True
LOSS_OPTIONS = {
    "beta_loss_scale": 1.0,
    "too_much_beta_scale": 0.0,
    "energy_loss_weight": 0.01,
    "classification_loss_weight": 0.01,
    "position_loss_weight": 0.0,
    "timing_loss_weight": 0.0,
    "q_min": 1.0,
    "use_average_cc_pos": 0.9999,
    "use_energy_weights": False,
}
loss_layer = LLExtendedObjectCondensation3



def config_model(Inputs, td, debug_outdir=None, plot_debug_every=PLOT_FREQUENCY, 
                 #check_keys is just for debugging
                 check_keys = False):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step, common to all models #############################
    ###########################################################################

    pre_processed = td.interpretAllModelInputs(Inputs)

    if not isinstance(td, TrainData_PreSnowflakeNanoML): #not preselected
        pre_processed = condition_input(pre_processed, no_scaling=True, no_prime=False, new_prime=True)
        pre_processed['features'] = ProcessFeatures()(pre_processed['features'])  
        
    else:
        pre_processed['row_splits'] = CastRowSplits()(pre_processed['row_splits']) #cast to int32, stored as int64

    #plot the prime coordinates for debugging
    pre_processed['prime_coords'] = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        publish = 'wandb',
        name='input_coords',
        )([pre_processed['prime_coords'], 
           pre_processed['rechit_energy'], 
           pre_processed['t_idx'], 
           pre_processed['row_splits']])  

    x = pre_processed['features']
    rs = pre_processed['row_splits']
    prime_coords = pre_processed['prime_coords']
    c_coords = pre_processed['coords']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    is_track = pre_processed['is_track']

    ###########################################################################
    ### Model definition ######################################################
    ###########################################################################


    all_features = [c_coords, x]

    for i in range(len(NEIGHBOURS)):

        x = Dense(DSHAPE, activation=DENSE_ACTIVATION, name = f'dense_{i}_a')(x)
        x = Dense(DSHAPE, activation=DENSE_ACTIVATION, name = f'dense_{i}_b')(x)
        x = Dense(DSHAPE, activation=DENSE_ACTIVATION, name = f'dense_{i}_c')(x)

        x = Concatenate()([prime_coords, c_coords, x])

        x = BatchNormalization(name = f'bn_{i}_a')(x)

        # run GravNet_plus_TEQMP
        x = GravNet_plus_TEQMP(f'gn_{i}',
                               x, prime_coords, energy, t_idx, rs,
                               DSHAPE,
                               NEIGHBOURS[i], 
                               n_gn_coords = 4,
                               teq_nodes = [64,64,64],

                               debug_outdir=debug_outdir,
                               plot_debug_every=plot_debug_every,
                               debug_publish = 'wandb',
                               trainable = True)
        
        x = BatchNormalization(name = f'bn_{i}_b')(x)
        all_features.append(x)

    x = Concatenate()(all_features)
    x = Dense(128, activation=DENSE_ACTIVATION, name = f'dense_final_a')(x)
    x = Dense(128, activation=DENSE_ACTIVATION, name = f'dense_final_b')(x)
    x = Dense(64,  activation=DENSE_ACTIVATION, name = f'dense_final_c')(x)
    x = BatchNormalization(name = f'bn_final')(x)

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x,
                n_ccoords=N_CLUSTER_SPACE_COORDINATES,
                fix_distance_scale=not DISTANCE_SCALE,
                is_track=is_track,
                set_track_betas_to_one=True)
    
    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation',
        publish = 'wandb',
        )([pred_ccoords, pred_beta, pre_processed['t_idx'], rs])

    pre_processed['t_spectator_weight'] = ZerosLike()(pred_beta) #this should not be necessary
    
    pred_beta = loss_layer(
            scale=1.,
            
            record_metrics=True,
            print_loss=False,
            name="ExtendedOCLoss",
            implementation = LOSS_IMPLEMENTATION,
            **LOSS_OPTIONS)(
                    [pred_beta, pred_ccoords, pred_dist, pred_energy_corr, pred_energy_low_quantile,
                        pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, energy,
                        pre_processed['t_idx'] , pre_processed['t_energy'] , pre_processed['t_pos'] ,
                        pre_processed['t_time'] , pre_processed['t_pid'] , pre_processed['t_spectator_weight'],
                        pre_processed['t_fully_contained'], pre_processed['t_rec_energy'],
                        pre_processed['t_is_unique'], pre_processed['row_splits']])

    model_outputs = {
        'pred_beta': pred_beta,
        'pred_ccoords': pred_ccoords,
        'pred_energy_corr_factor': pred_energy_corr,
        'pred_energy_low_quantile': pred_energy_low_quantile,
        'pred_energy_high_quantile': pred_energy_high_quantile,
        'pred_pos': pred_pos,
        'pred_time': pred_time,
        'pred_id': pred_id,
        'pred_dist': pred_dist,
        'rechit_energy': energy,
        'row_splits': pre_processed['row_splits'],
        'no_noise_sel': pre_processed['survived_both_stages']
        }

    return Model(inputs=Inputs, outputs=model_outputs)


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


train.change_learning_rate(1e-3)
train.trainModel(
        nepochs=10,
        batchsize=10000,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=[],
        collect_gradients = 4 #average out more gradients
        )

train.change_learning_rate(2e-4)
train.trainModel(
        nepochs=1+2+10,
        batchsize=10000,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=[],
        collect_gradients = 4 #average out more gradients
        )

# loop through model layers and turn  batch normalisation to fixed
def fix_batchnorm(m):
    for layer in m.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        if isinstance(layer, ScaledGooeyBatchNorm2):
            layer.trainable = False

#apply to all models
train.applyFunctionToAllModels(fix_batchnorm)
#recompile
train.compileModel(learningrate=1e-3)
print('entering second training phase')

train.change_learning_rate(3e-5)
train.trainModel(
        nepochs=1+2+20+10,
        batchsize=60000,
        add_progbar=False,
        additional_callbacks=[],
        collect_gradients = 4
        )
