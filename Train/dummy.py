"""
python3 dummy.py <path to dataCollection.djcdc>  <output dir> --no_wandb
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

# import wandb
from DeepJetCore.wandb_interface import wandb_wrapper as wandb
from wandb_callback import wandbCallback
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras import Model

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient, ScalarMultiply

import training_base_hgcal
from Layers import ScaledGooeyBatchNorm2
from Layers import MixWhere
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


####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")

train = training_base_hgcal.HGCalTraining(parser=parser)

loss_layer = LLExtendedObjectCondensation2

if not train.args.no_wandb:
    wandb.init(
        project=train.args.wandb_project,
        config=wandb_config,
    )
    wandb.save(sys.argv[0]) # Save python file
    wandb.save(train.args.configFile) # Save config file
else:
    wandb.active=False


###############################################################################
### Define Model ##############################################################
###############################################################################

N_CLUSTER_SPACE_COORDINATES  = 3
DENSE_ACTIVATION = 'elu'
DISTANCE_SCALE = False

LOSS_OPTIONS = {
    'energy_loss_weight': 0.,
    'q_min': 5.,
    'use_average_cc_pos': 0.99,
    'classification_loss_weight':0.0, # to make it work0.5,
    'too_much_beta_scale': 0.0,
    'position_loss_weight':0.,
    'timing_loss_weight':0.0,
    'beta_loss_scale':1., #2.0
    # 'implementation': 'hinge_full_grad' #'hinge_manhatten'#'hinge'#old school
    }

def dummy_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False)

    prime_coords = pre_processed['prime_coords']
    c_coords = prime_coords
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    x = pre_processed['features']
    x = ScaledGooeyBatchNorm2(
            learn=True,
            no_bias_gamma=True,
            no_gaus=False)([x, is_track])

    x = ScaledGooeyBatchNorm2(
            invert_condition=True,
            learn=True,
            no_bias_gamma=True,
            no_gaus=False)([x, is_track])

    c_coords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name='input_c_coords',
        # publish = publishpath
        )([c_coords, energy, t_idx, rs])
    c_coords = extent_coords_if_needed(prime_coords, x, N_CLUSTER_SPACE_COORDINATES)

    x = Concatenate()([x, c_coords, is_track])
    x = Dense(64, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x)
    x = Dense(64,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION)(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION)(x)
    x = ScaledGooeyBatchNorm2(
        name=f"batchnorm_final")(x)

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x,
                n_ccoords=N_CLUSTER_SPACE_COORDINATES,
                fix_distance_scale=not DISTANCE_SCALE,
                is_track=is_track,
                set_track_betas_to_one=True)

    # pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)([pred_ccoords, rs, t_idx])


    pred_beta = loss_layer(
            scale=1.,
            use_energy_weights=True,
            record_metrics=True,
            print_loss=False,
            name="ExtendedOCLoss",
            implementation = 'hinge',
            **LOSS_OPTIONS)(
                    [pred_beta, pred_ccoords, pred_dist, pred_energy_corr, pred_energy_low_quantile,
                        pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, energy,
                        pre_processed['t_idx'] , pre_processed['t_energy'] , pre_processed['t_pos'] ,
                        pre_processed['t_time'] , pre_processed['t_pid'] , pre_processed['t_spectator_weight'],
                        pre_processed['t_fully_contained'], pre_processed['t_rec_energy'],
                        pre_processed['t_is_unique'], pre_processed['row_splits']])

    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation'
        )([pred_ccoords, pred_beta, pre_processed['t_idx'], rs])

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
        # 'no_noise_sel': pre_processed['no_noise_sel'],
        # 'no_noise_rs': pre_processed['no_noise_rs'],
        }

    # return DictModel(inputs=Inputs, outputs=model_outputs)
    return Model(inputs=Inputs, outputs=model_outputs)


###############################################################################
### Set up training ###########################################################
###############################################################################



if not train.modelSet():
    train.setModel(
        dummy_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        )
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]
RECORD_FREQUENCY = 10
PLOT_FREQUENCY = 40

cb = [NanSweeper()] #this takes a bit of time checking each batch but could be worth it

###############################################################################
### Actual Training ###########################################################
###############################################################################

train.trainModel(
        nepochs=3,
        batchsize=120000,
        add_progbar=True,
        additional_callbacks=cb
        )

