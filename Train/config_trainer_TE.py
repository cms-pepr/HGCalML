"""
Flexible training script that should be mostly configured with a yaml config file
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

import wandb
from wandb_callback import wandbCallback
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout

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
from Layers import DictModel
from Layers import RaggedGlobalExchange
from Layers import SphereActivation
from Layers import Multi
from Layers import ShiftDistance
from Layers import LLRegulariseGravNetSpace
from Layers import SplitOffTracks, ConcatRaggedTensors
from Regularizers import AverageDistanceRegularizer
from model_blocks import tiny_pc_pool, condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from model_tools import apply_weights_from_path
from model_blocks import random_sampling_unit, random_sampling_block, random_sampling_block2
from noise_filter import noise_filter
from callbacks import plotClusteringDuringTraining
from callbacks import plotClusterSummary
from callbacks import NanSweeper, DebugPlotRunner


####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('configFile')
parser.add_argument('--run_name', help="wandb run name")
CONFIGFILE = sys.argv[1]
print(f"Using config File: \n{CONFIGFILE}")

with open(CONFIGFILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

N_CLUSTER_SPACE_COORDINATES = config['General']['n_cluster_space_coordinates']
N_GRAVNET_SPACE_COORDINATES = config['General']['n_gravnet_space_coordinates']
GRAVNET_ITERATIONS = len(config['General']['gravnet'])
LOSS_OPTIONS = config['LossOptions']
BATCHNORM_OPTIONS = config['BatchNormOptions']
DENSE_ACTIVATION = config['DenseOptions']['activation']
DENSE_REGULARIZER = tf.keras.regularizers.l2(config['DenseOptions']['kernel_regularizer_rate'])
DROPOUT = config['DenseOptions']['dropout']

wandb_config = {
    "loss_implementation"           :   config['General']['oc_implementation'],
    "gravnet_iterations"            :   GRAVNET_ITERATIONS,
    "gravnet_space_coordinates"     :   N_GRAVNET_SPACE_COORDINATES,
    "cluster_space_coordinates"     :   N_CLUSTER_SPACE_COORDINATES,
    "loss_energy_weight"            :   config['LossOptions']['energy_loss_weight'],
    "loss_classification_weight"    :   config['LossOptions']['classification_loss_weight'],
    "loss_qmin"                     :   config['LossOptions']['q_min'],
    "loss_use_average_cc_pos"       :   config['LossOptions']['use_average_cc_pos'],
    "loss_too_much_beta_scale"      :   config['LossOptions']['too_much_beta_scale'],
    "loss_beta_scale"               :   config['LossOptions']['beta_loss_scale'],
    "batch_max_viscosity"           :   config['BatchNormOptions']['max_viscosity'],
    "dense_activation"              :   config['DenseOptions']['activation'],
    "dense_kernel_reg"              :   config['DenseOptions']['kernel_regularizer_rate'] ,
    "dense_dropout"                 :   config['DenseOptions']['dropout'],
}

for i in range(GRAVNET_ITERATIONS):
    wandb_config[f"gravnet_{i}_neighbours"] =config['General']['gravnet'][i]['n']
for i in range(len(config['Training'])):
    wandb_config[f"train_{i}_lr"] = config['Training'][i]['learning_rate']
    wandb_config[f"train_{i}_epochs"] = config['Training'][i]['epochs']
    wandb_config[f"train_{i}_batchsize"] = config['Training'][i]['batch_size']
    if i == 1:
        wandb_config[f"train_{i}+_max_visc"] = 0.999
        wandb_config[f"train_{i}+_fluidity_decay"] = 0.1

wandb.init(
    project="playground",
    config=wandb_config,
)
wandb.save(sys.argv[0]) # Save python file
wandb.save(sys.argv[1]) # Save config file


###############################################################################
### Define Model ##############################################################
###############################################################################

def TEGN_block(x, rs, N_coords, K, messages):
    x_pre = x
    coords = Dense(N_coords)(x)
    nidx, distsq = KNN(K=K)([coords, rs])
    x = TranslationInvariantMP(messages, activation='elu')([x, nidx, distsq])
    x = Dense(x_pre.shape[-1], activation='elu')(x)
    # xgn, gncoords, gnnidx, gndist= RaggedGravNet...
    out = Add()([x, x_pre])
    return out, coords, nidx, distsq


def config_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
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

    c_coords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name='input_c_coords',
        # publish = publishpath
        )([c_coords, energy, t_idx, rs])
    c_coords = extent_coords_if_needed(prime_coords, x, N_CLUSTER_SPACE_COORDINATES)

    x = Concatenate()([x, c_coords, is_track])
    x = Dense(64, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x)

    allfeat = []
    print("Available keys: ", pre_processed.keys())

    ###########################################################################
    ### Loop over GravNet Layers ##############################################
    ###########################################################################


    for i in range(GRAVNET_ITERATIONS):

        d_shape = x.shape[1]//2

        if i in (0, 4):
            x = Dense(d_shape,activation=DENSE_ACTIVATION,
                kernel_regularizer=DENSE_REGULARIZER)(x)
            x = Dense(d_shape,activation=DENSE_ACTIVATION,
                kernel_regularizer=DENSE_REGULARIZER)(x)

            x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        x_pre = x
        # get indices of x

        if i == 0:
            x_hit, x_track, rs_hit, rs_track = SplitOffTracks()([is_track, [x], rs])
            x_hit = x_hit[0]
            x_track = x_track[0]


            # def TEGN_block(x, rs, N_coords, K, messages):
            xgn_hit, gncoords_hit, gnnidx_hit, gndist_hit = TEGN_block(
                    x_hit, rs_hit,
                    N_GRAVNET_SPACE_COORDINATES, config['General']['gravnet'][i]['n'],
                    [64, 32, 16])
            xgn_track, gncoords_track, gnnidx_track, gndist_track = TEGN_block(
                    x_track, rs_track,
                    N_GRAVNET_SPACE_COORDINATES, config['General']['gravnet'][i]['n'],
                    [64, 32, 16])

            # xgn_hit, gncoords_hit, gnnidx_hit, gndist_hit = RaggedGravNet(
                    # name = f"RSU_gravnet_{i}_hit", # 76929, 42625, 42625
                # n_neighbours=config['General']['gravnet'][i]['n'],
                # n_dimensions=N_GRAVNET_SPACE_COORDINATES,
                # n_filters=d_shape,
                # n_propagate=2*d_shape,
                # coord_initialiser_noise=None,
                # feature_activation='elu',
                # )([x_hit, rs_hit])
            # xgn_track, gncoords_track, gnnidx_track, gndist_track = RaggedGravNet(
                    # name = f"RSU_gravnet_{i}_track", # 76929, 42625, 42625
                # n_neighbours=16,
                # n_dimensions=N_GRAVNET_SPACE_COORDINATES,
                # n_filters=d_shape,
                # n_propagate=2*d_shape,
                # coord_initialiser_noise=None,
                # feature_activation='elu',
                # )([x_track, rs_track])


            # x_hit = DistanceWeightedMessagePassing([64, 32, 16], activation='elu')([x_hit, gnnidx_hit, gndist_hit])
            # x_track = DistanceWeightedMessagePassing([64, 32, 16], activation='elu')([x_track, gnnidx_track, gndist_track])
            # x_hit = TranslationInvariantMP([64, 32, 16], activation='elu')([x_hit, gnnidx_hit, gndist_hit])
            # x_track = TranslationInvariantMP([64, 32, 16], activation='elu')([x_track, gnnidx_track, gndist_track])

            [xgn, gncoords], rs  = ConcatRaggedTensors()([
                [xgn_track, gncoords_track],
                [xgn_hit, gncoords_hit],
                rs_track, rs_hit])

        else:
            xgn, gncoords, gnnidx, gndist= TEGN_block(
                    x, rs,
                    N_GRAVNET_SPACE_COORDINATES, config['General']['gravnet'][i]['n'],
                    [64, 32, 16])
            # xgn, gncoords, gnnidx, gndist = RaggedGravNet(
                    # name = f"RSU_gravnet_{i}", # 76929, 42625, 42625
                # n_neighbours=config['General']['gravnet'][i]['n'],
                # n_dimensions=N_GRAVNET_SPACE_COORDINATES,
                # n_filters=d_shape,
                # n_propagate=2*d_shape,
                # coord_initialiser_noise=None,
                # feature_activation='elu',
                # sumwnorm=True,
                # )([x, rs])

            gndist = LLRegulariseGravNetSpace(
                    scale=0.01,
                    record_metrics=True,
                    name=f'regularise_gravnet_{i}')([gndist, prime_coords, gnnidx])

            # x_rand = random_sampling_block(
                    # xgn, rs, gncoords, gnnidx, gndist, is_track,
                    # reduction=6, layer_norm=True, name=f"RSU_{i}")
            x_rand = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x_rand)

            gndist = AverageDistanceRegularizer(
                strength=1e-3,
                record_metrics=True
                )(gndist)
        # gndist = StopGradient()(gndist)
        gncoords = StopGradient()(gncoords)
        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i)
            )([gncoords, energy, t_idx, rs])

        if i == 0:
            x = Concatenate()([xgn, gncoords])
        else:
            x = Concatenate()([x_pre, xgn, x_rand, gncoords])
        x = Dense(d_shape,
                  name=f"dense_post_gravnet_1_iteration_{i}",
                  activation=DENSE_ACTIVATION,
                  kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(d_shape,
                  name=f"dense_post_gravnet_2_iteration_{i}",
                  activation=DENSE_ACTIVATION,
                  kernel_regularizer=DENSE_REGULARIZER)(x)

        x = ScaledGooeyBatchNorm2(
            name=f"batchnorm_loop1_iteration_{i}",
            **BATCHNORM_OPTIONS)(x)

        allfeat.append(x)

        if len(allfeat) > 1:
            x = Concatenate()(allfeat)
        else:
            x = allfeat[0]

    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################


    x = Dense(64,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = ScaledGooeyBatchNorm2(
        name=f"batchnorm_final",
        **BATCHNORM_OPTIONS)(x)

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES, fix_distance_scale=True)

    # pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)([pred_ccoords, rs, t_idx])

    if config['General']['oc_implementation'] == 'hinge':
        loss_implementation = 'hinge'
    else:
        loss_implementation = ''

    pred_beta = LLExtendedObjectCondensation(scale=1.,
                                             use_energy_weights=True,
                                             record_metrics=True,
                                             print_loss=True,
                                             name="ExtendedOCLoss",
                                             implementation = loss_implementation,
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

    return DictModel(inputs=Inputs, outputs=model_outputs)


###############################################################################
### Set up training ###########################################################
###############################################################################


train = training_base_hgcal.HGCalTraining(parser=parser)

if not train.modelSet():
    train.setModel(
        config_model,
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
cb += [
    plotClusteringDuringTraining(
        use_backgather_idx=8 + i,
        outputfile=train.outputDir + "/localclust/cluster_" + str(i) + '_',
        samplefile=samplepath,
        after_n_batches=500,
        on_epoch_end=False,
        publish=None,
        use_event=0
        )
    for i in [0, 2, 4]
    ]

cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics=[
            'ExtendedOCLoss_loss',
            'ExtendedOCLoss_dynamic_payload_scaling',
            'ExtendedOCLoss_attractive_loss',
            'ExtendedOCLoss_repulsive_loss',
            'ExtendedOCLoss_min_beta_loss',
            'ExtendedOCLoss_noise_loss',
            'ExtendedOCLoss_class_loss',
            'ExtendedOCLoss_energy_loss',
            'ExtendedOCLoss_energy_unc_loss',
            # 'ExtendedOCLoss_time_std',
            # 'ExtendedOCLoss_time_pred_std',
            '*regularise_gravnet_*',
            '*_gravReg*',
            ],
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    ]

cb += [


    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    ]

cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=1000
        )
    ]

cb += [wandbCallback()]

###############################################################################
### Actual Training ###########################################################
###############################################################################

shutil.copyfile(CONFIGFILE, os.path.join(sys.argv[3], "config.yaml"))

N_TRAINING_STAGES = len(config['Training'])
for i in range(N_TRAINING_STAGES):
    print(f"Starting training stage {i}")
    learning_rate = config['Training'][i]['learning_rate']
    epochs = config['Training'][i]['epochs']
    batch_size = config['Training'][i]['batch_size']
    train.change_learning_rate(learning_rate)
    print(f"Training for {epochs} epochs")
    print(f"Learning rate set to {learning_rate}")
    print(f"Batch size: {batch_size}")

    if i == 1:
        # change batchnorm
        for layer in train.keras_model.layers:
            if 'batchnorm' in layer.name:
                layer.max_viscosity = 0.999
                layer.fluidity_decay = 0.01
    model, history = train.trainModel(
        nepochs=epochs,
        batchsize=batch_size,
        additional_callbacks=cb
        )
