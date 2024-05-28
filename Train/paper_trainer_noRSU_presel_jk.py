"""
Training script
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

# import wandb
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

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")



N_CLUSTER_SPACE_COORDINATES = 6
N_GRAVNET_SPACE_COORDINATES = 4
d_shape = 64
NEIGHBOURS = [32, 64]
LOSS_IMPLEMENTATION = "hinge"
GRAVNET_ITERATIONS = len(NEIGHBOURS)
LOSS_OPTIONS = {
    "beta_loss_scale": 1.0,
    "too_much_beta_scale": 0.0,
    "energy_loss_weight": 0.01,
    "classification_loss_weight": 0.01,
    "position_loss_weight": 0.0,
    "timing_loss_weight": 0.0,
    "q_min": 1.0,
    "use_average_cc_pos": 0.9999,
}
BATCHNORM_OPTIONS = {
    "max_viscosity": 0.9999,
    "learn": False
}
DENSE_ACTIVATION = "elu"
DENSE_INIT = "he_normal"
DENSE_REGULARIZER_RATE = 1e-9
DENSE_REGULARIZER = tf.keras.regularizers.l2(DENSE_REGULARIZER_RATE)
DROPOUT = 1e-2
DISTANCE_SCALE = True
loss_layer = LLExtendedObjectCondensation3

TRAINING = {
  "stage_1": {
    "batch_size": 120000,
    "learning_rate": 0.001,
    "epochs": 5,
    },

  "stage_2": {
    "batch_size": 120000,
    "learning_rate": 0.0001,
    "epochs": 10,
    },

  "stage_3": {
    "batch_size": 120000,
    "learning_rate": 0.00001,
    "epochs": 20,
    },
}

wandb_config = {
    "gravnet_iterations"            :   GRAVNET_ITERATIONS,
    "gravnet_space_coordinates"     :   N_GRAVNET_SPACE_COORDINATES,
    "cluster_space_coordinates"     :   N_CLUSTER_SPACE_COORDINATES,
    "loss_energy_weight"            :   LOSS_OPTIONS['energy_loss_weight'],
    "loss_classification_weight"    :   LOSS_OPTIONS['classification_loss_weight'],
    "loss_qmin"                     :   LOSS_OPTIONS['q_min'],
    "loss_use_average_cc_pos"       :   LOSS_OPTIONS['use_average_cc_pos'],
    "loss_too_much_beta_scale"      :   LOSS_OPTIONS['too_much_beta_scale'],
    "loss_beta_scale"               :   LOSS_OPTIONS['beta_loss_scale'],
    "dense_activation"              :   DENSE_ACTIVATION,
    "dense_kernel_reg"              :   DENSE_REGULARIZER_RATE,
    "dense_dropout"                 :   DROPOUT,
    "distance_scale"                :   DISTANCE_SCALE,
}

for i in range(GRAVNET_ITERATIONS):
    wandb_config[f"gravnet_{i}_neighbours"] = NEIGHBOURS[i]
for i in range(1, len(TRAINING)+1):
    wandb_config[f"train_{i}_lr"] = TRAINING[f"stage_{i}"]['learning_rate']
    wandb_config[f"train_{i}_epochs"] = TRAINING[f"stage_{i}"]['epochs']
    wandb_config[f"train_{i}_batchsize"] = TRAINING[f"stage_{i}"]['batch_size']
    if i == 1:
        wandb_config[f"train_{i}+_max_visc"] = 0.999
        wandb_config[f"train_{i}+_fluidity_decay"] = 0.1


pre_args,_ = parser.parse_known_args() 

if not pre_args.no_wandb:
    wandb.init(
        project=pre_args.wandb_project,
        config={},
    )
    wandb.save(sys.argv[0]) # Save python file
else:
    wandb.active=False
#parses the rest of the arguments
train = training_base_hgcal.HGCalTraining(parser=parser)


PLOT_FREQUENCY = 400

###############################################################################
### Define Model ##############################################################
###############################################################################

from model_blocks import GravNet_plus_TEQMP, tree_condensation_block

def config_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False, new_prime=True)
    
    pre_processed['features'] = ProcessFeatures()(pre_processed['features'])    

    pre_processed, graph = tree_condensation_block(pre_processed)

    orig_input.update(pre_processed) #overwrite also the truth

    ############## preclus done #########

    prime_coords = pre_processed['prime_coords']

    c_coords = prime_coords
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    x = pre_processed['features']

    c_coords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name='input_c_coords',
        )([c_coords, energy, t_idx, rs])
    
    [cprime_hit], [cprime_track], rs_hit, rs_track = SplitOffTracks()([is_track, [prime_coords], rs])
    
    c_coords = extent_coords_if_needed(prime_coords, x, N_CLUSTER_SPACE_COORDINATES)

    x = Concatenate()([x, c_coords, is_track])
    x = Dense(64, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x)

    allfeat = []
    print("Available keys: ", pre_processed.keys())

    ###########################################################################
    ### Loop over GravNet Layers ##############################################
    ###########################################################################


    for i in range(GRAVNET_ITERATIONS):

        x = Dense(d_shape,activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(d_shape,activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        
        #x = LayerNormalization()(x)
        x_pre = x
        x = Concatenate()([c_coords, x])

        [x_hit, energy_hit, t_idx_hit], [x_track, energy_track, t_idx_track], rs_hit, rs_track = SplitOffTracks()(
                [is_track, [x, energy, t_idx], rs])


        x_hit = GravNet_plus_TEQMP(f'gnhit_{i}', x_hit, cprime_hit, energy_hit, t_idx_hit, rs_hit, 
                                   d_shape, NEIGHBOURS[i], debug_outdir, plot_debug_every)
        x_track = GravNet_plus_TEQMP(f'gntrack_{i}', x_track, cprime_track, energy_track, t_idx_track, rs_track, 
                                     d_shape, 16, debug_outdir, plot_debug_every)
        
        [xgn], rs  = ConcatRaggedTensors()([
            [x_track],
            [x_hit],
            rs_track, rs_hit])
        
        x = Concatenate()([xgn, x_pre])
        #x = LayerNormalization()(x)

        x = GravNet_plus_TEQMP(f'gncomb_{i}', x, prime_coords, energy, t_idx, rs, 
                               d_shape, NEIGHBOURS[i], debug_outdir, plot_debug_every, space_reg_strength=1e-2)

        x = Dense(d_shape,
                  name=f"dense_post_gravnet_1_iteration_{i}",
                  activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
                  kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(d_shape,
                  name=f"dense_post_gravnet_2_iteration_{i}",
                  activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
                  kernel_regularizer=DENSE_REGULARIZER)(x)
        

        #x = LayerNormalization()(x)
        allfeat.append(x)
        if len(allfeat) > 1:
            x = Concatenate()(allfeat)
        else:
            x = allfeat[0]


    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################


    x = Dense(128,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    #x = BatchNormalization()(x)
    

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x,
                n_ccoords=N_CLUSTER_SPACE_COORDINATES,
                fix_distance_scale=not DISTANCE_SCALE,
                is_track=is_track,
                set_track_betas_to_one=True)


    pred_beta = loss_layer(
            scale=1.,
            use_energy_weights=True,
            record_metrics=True,
            print_loss=train.args.no_wandb,
            name="ExtendedOCLoss",
            implementation = LOSS_IMPLEMENTATION,
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
        config_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        plot_debug_every=PLOT_FREQUENCY,
        )
    # load weights from $HGCALML/models/tree_condensation_block_noPU/KERAS_check_model_last.h5
    apply_weights_from_path(os.getenv("HGCALML") + "/models/tree_condensation_block_noPU/KERAS_check_model_last.h5", train.keras_model)
    train.setCustomOptimizer(tf.keras.optimizers.Adam())#clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]


cb = []#[NanSweeper()] #this takes a bit of time checking each batch but could be worth it


# cb += [wandbCallback()]

###############################################################################
### Actual Training ###########################################################
###############################################################################


N_TRAINING_STAGES = len(TRAINING)
for i in range(1, N_TRAINING_STAGES+1):
    print(f"Starting training stage {i}")
    learning_rate = TRAINING[f"stage_{i}"]['learning_rate']
    epochs = TRAINING[f"stage_{i}"]['epochs']
    batch_size = TRAINING[f"stage_{i}"]['batch_size']
    train.change_learning_rate(learning_rate)
    print(f"Training for {epochs} epochs")
    print(f"Learning rate set to {learning_rate}")
    print(f"Batch size: {batch_size}")

    if i == 2:
        # fix the keras batch norm 
        def fix_batchnorm(model):
            for layer in model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
        train.applyFunctionToAllModels(fix_batchnorm)
        #train.distributeModelToGPUs() #re-distribute changes
        train.compileModel(learningrate=learning_rate) # recompile models

    train.trainModel(
        nepochs=epochs,
        batchsize=batch_size,
        add_progbar=True,
        additional_callbacks=cb,
        collect_gradients = 4 #make it more smooth
        )
