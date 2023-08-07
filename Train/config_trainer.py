"""
Flexible training script that should be mostly configured with a yaml config file
"""

import os
import sys
import yaml
import shutil
from argparse import ArgumentParser
import mlflow
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient

import training_base_hgcal
from Layers import ScaledGooeyBatchNorm2
from Layers import MixWhere
from Layers import RaggedGravNet
from Layers import PlotCoordinates
from Layers import DistanceWeightedMessagePassing
from Layers import LLFillSpace
from Layers import LLExtendedObjectCondensation
from Layers import DictModel
from Layers import RaggedGlobalExchange
from Layers import SphereActivation
from Layers import Multi
from Layers import ShiftDistance
from Regularizers import AverageDistanceRegularizer
from model_blocks import tiny_pc_pool, condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from model_tools import apply_weights_from_path
from noise_filter import noise_filter
from callbacks import plotClusteringDuringTraining
from callbacks import plotClusterSummary
from callbacks import NanSweeper, DebugPlotRunner


###############################################################################
### Load Configuration ########################################################
###############################################################################

parser = ArgumentParser('training')
parser.add_argument('configFile')
# args = parser.parse_args()

CONFIGFILE = "/mnt/home/pzehetner/ML4Reco/Train/configuration/pre-pooled_config.yaml"
CONFIGFILE = "/mnt/home/pzehetner/ML4Reco/Train/configuration/pre-pooled.yaml"
CONFIGFILE = "/mnt/home/pzehetner/ML4Reco/Train/configuration/pooling_config.yaml"
CONFIGFILE = "/mnt/home/pzehetner/ML4Reco/Train/configuration/noise_config.yaml"
# CONFIGFILE = args.configFile
CONFIGFILE = sys.argv[1]

print(f"Using config File: \n{CONFIGFILE}")
os.environ['MLFLOW_ARTIFACT_URI'] = sys.argv[3]

with open(CONFIGFILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


N_CLUSTER_SPACE_COORDINATES = config['Architecture']['n_cluster_space_coordinates']
N_GRAVNET_SPACE_COORDINATES = config['Architecture']['n_gravnet_space_coordinates']
GRAVNET_ITERATIONS = len(config['Architecture']['gravnet'])
MESSAGE_PASSING_ITERATIONS = len(config['Architecture']['message_passing'])
PRE_GRAVNET_DENSE_ITERATIONS = len(config['Architecture']['dense_pre_gravnet'])
POST_MESSAGE_PASSING_DENSE_ITERATIONS = len(config['Architecture']['dense_post_message_passing'])
FINAL_DENSE_ITERATIONS = len(config['Architecture']['dense_final'])
LOSS_OPTIONS = config['LossOptions']
BATCHNORM_OPTIONS = config['BatchNormOptions']
DENSE_ACTIVATION = config['DenseOptions']['activation']
if config['DenseOptions']['kernel_regularizer'].lower() == 'l2':
    DENSE_REGULARIZER = tf.keras.regularizers.l2(config['DenseOptions']['kernel_regularizer_rate'])
elif config['DenseOptions']['kernel_regularizer'].lower() == 'l1':
    DENSE_REGULARIZER = tf.keras.regularizers.l1(config['DenseOptions']['kernel_regularizer_rate'])
else:
    DENSE_REGULARIZER = None
DROPOUT = config['DenseOptions']['dropout']
USE_FILL_SPACE = config['General']['use_fill_space']
USE_LAYER_NORMALIZATION = config['General']['use_layer_normalization']
SPHERE_ACTIVATION = config['General']['use_layer_normalization']

###############################################################################
### Define Model ##############################################################
###############################################################################

def config_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    """
    Function that defines the model to train
    """
    orig_input = td.interpretAllModelInputs(Inputs)
    embedded = False

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################
    if config['General']['pre-model-type'] == 'noise-filter':
        pre_processed = noise_filter(orig_input,
                                     trainable=False,
                                     pass_through=False)
        x_in = Concatenate(name='concat_noise_filter')(
            [pre_processed['coords'],
             pre_processed['features']])
    elif config['General']['pre-model-type'] == 'pre-pooling':
        embedded = True # Tracks and hits are already embedded after pre-pooling
        pre_processed = condition_input(orig_input, no_scaling=True)
        trans, pre_processed = tiny_pc_pool(pre_processed,
                                            record_metrics=True,
                                            trainable=False)
        x_in = Concatenate(name='concat_pre_pooling')(
            [pre_processed['prime_coords'],
             pre_processed['features']])
    elif config['General']['pre-model-type'] == 'pre-pooled':
        print("Assuming dataset is already pre-pooled")
        embedded = True
        pre_processed = condition_input(orig_input, no_scaling=True)
        trans, pre_processed = tiny_pc_pool(pre_processed,
                                            record_metrics=True,
                                            pass_through=True)
        x_in = Concatenate(name='concat_pre_pooling')(
            [pre_processed['prime_coords'],
             pre_processed['features']])
    elif config['General']['pre-model-type'] == 'pre-clustering':
        print("Not yet implemented")
        raise NotImplementedError
    elif config['General']['pre-model-type'] == 'none':
        pre_processed = orig_input
    elif config['General']['pre-model-type'] == 'pass-through':
        pre_processed = orig_input
        pre_processed = noise_filter(orig_input,
                                     trainable=False,
                                     pass_through=True)
        x_in = Concatenate(name='concat_noise_filter')(
            [pre_processed['coords'],
             pre_processed['features']])
    else:
        print("Unknown pre-model-type")
        raise ValueError

    c_coords = pre_processed['coords']
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']

    if SPHERE_ACTIVATION:
        x_in = Concatenate()([x_in, is_track, SphereActivation()(x_in)])

    c_coords = extent_coords_if_needed(c_coords, x_in, N_CLUSTER_SPACE_COORDINATES)
    c_coords = ScaledGooeyBatchNorm2(
        name='batchnorm_ccoords', **BATCHNORM_OPTIONS)(c_coords)

    if not embedded:
        x_hit = Dense(64,
                    name='dense_embedding_hit',
                    activation=DENSE_ACTIVATION,
                    kernel_regularizer=DENSE_REGULARIZER)(x_in)
        x_hit = Dropout(DROPOUT)(x_hit)
        x_track = Dense(64,
                        name='dense_embedding_track',
                        activation=DENSE_ACTIVATION,
                        kernel_regularizer=DENSE_REGULARIZER)(x_in)
        x_track = Dropout(DROPOUT)(x_track)
        x = MixWhere(name='mix_hits_tracks')([is_track, x_hit, x_track])
    else:
        x = x_in

    x = ScaledGooeyBatchNorm2(name="batchnorm_0", **BATCHNORM_OPTIONS)(x)
    x = Dense(128, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x_in)
    allfeat = []
    print("Available keys: ", pre_processed.keys())

    ###########################################################################
    ### Loop over GravNet Layers ##############################################
    ###########################################################################

    for i in range(GRAVNET_ITERATIONS):
        
        x = RaggedGlobalExchange()([x, rs])
        x, norm = SphereActivation(return_norm=True)(x)

        for j in range(PRE_GRAVNET_DENSE_ITERATIONS):
            n = config['Architecture']['dense_pre_gravnet'][j]['n']
            x = Dense(n,
                      name=f"dense_pre_gravnet{j}_iteration_{i}",
                      activation=DENSE_ACTIVATION,
                      kernel_regularizer=DENSE_REGULARIZER)(x)
            if not j == PRE_GRAVNET_DENSE_ITERATIONS - 1:
                x = Dropout(DROPOUT, name=f"droput_pre_gravnet_{j}_iteration_{i}")(x)

        x = ScaledGooeyBatchNorm2(
            name=f"batchnorm_1_iteration_{i}",
            **BATCHNORM_OPTIONS)(x)
        x = Concatenate(name=f"concat_ccoords_iteration_{i}")([c_coords,x])

        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            name = f"gravnet_{i}",
            n_neighbours=config['Architecture']['gravnet'][i]['n'],
            n_dimensions=N_GRAVNET_SPACE_COORDINATES,
            n_filters=64,
            n_propagate=64,
            record_metrics=True,
            coord_initialiser_noise=1e-2,
            use_approximate_knn=False
            )([x, rs])
        x = Concatenate(name=f"concat_xgn_iteration_{i}")([x, xgn])

        gndist = AverageDistanceRegularizer(
            strength=1e-3,
            record_metrics=True
            )(gndist)

        gndist = StopGradient()(gndist)

        gncoords = StopGradient()(gncoords)
        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i)
            )([gncoords, energy, t_idx, rs])
        x = Concatenate()([gncoords,x])
        x = ScaledGooeyBatchNorm2(
            name=f"batchnorm_2_iteration_{i}",
            **BATCHNORM_OPTIONS)(x)

        for j in range(MESSAGE_PASSING_ITERATIONS):
            n = config['Architecture']['message_passing'][j]['n']
            shift = config['Architecture']['message_passing'][j]['shift']
            gndist = ShiftDistance(shift=shift)(gndist)
            x = DistanceWeightedMessagePassing(
                name=f"message_passing_{j}_iteration_{i}",
                n_feature_transformation = [n],
                activation=DENSE_ACTIVATION,
            )([x, gnnidx, gndist])
            if SPHERE_ACTIVATION:
                x = SphereActivation()(x)
            else:
                x = ScaledGooeyBatchNorm2(
                    name=f"batchnorm_message_passing_{j}_iteration_{i}",
                    **BATCHNORM_OPTIONS)(x)

        for j in range(POST_MESSAGE_PASSING_DENSE_ITERATIONS):
            n = config['Architecture']['dense_post_message_passing'][j]['n']
            x = Dense(n,
                      name=f"dense_post_gravnet_{j}_iteration_{i}",
                      activation=DENSE_ACTIVATION,
                      kernel_regularizer=DENSE_REGULARIZER)(x)
            if not j == POST_MESSAGE_PASSING_DENSE_ITERATIONS -1:
                x = Dropout(DROPOUT, name=f"droput_{j}_iteration_{i}")(x)

        x = Multi()([x,norm])
        x = ScaledGooeyBatchNorm2(
            name=f"batchnorm_3_iteration_{i}",
            **BATCHNORM_OPTIONS)(x)

        allfeat.append(x)

    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################

    x = Concatenate()(allfeat)

    for j in range(FINAL_DENSE_ITERATIONS):
        n = config['Architecture']['dense_final'][j]['n']
        x = Dense(n,
                  name=f"dense_final_{j}",
                  activation=DENSE_ACTIVATION,
                  kernel_regularizer=DENSE_REGULARIZER)(x)
        if not j == FINAL_DENSE_ITERATIONS -1:
            x = Dropout(DROPOUT, name=f"droput_final_{j}")(x)

    x = Dense(64, name='dense_very_final', activation=DENSE_ACTIVATION)(x)
    x = ScaledGooeyBatchNorm2(
        name=f"batchnorm_final",
        **BATCHNORM_OPTIONS)(x)

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES, fix_distance_scale=True)

    if USE_FILL_SPACE:
        pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)(
                [pred_ccoords, rs, t_idx])


    pred_beta = LLExtendedObjectCondensation(scale=1.,
                                             use_energy_weights=True,
                                             record_metrics=True,
                                             print_loss=True,
                                             name="ExtendedOCLoss",
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

    if config['General']['pre-model-type'] == 'noise-filter':
        model_outputs['no_noise_sel'] = pre_processed['no_noise_sel']
        model_outputs['no_noise_rs'] = pre_processed['no_noise_rs']

    if (config['General']['pre-model-type'] == 'pre-pooling') or (config['General']['pre-model-type'] == 'pre-pooled'):
        model_outputs['no_noise_sel'] = trans['sel_idx_up']
        model_outputs['no_noise_rs'] = trans['rs_down']
        model_outputs['sel_idx'] = trans['sel_idx_up']
        model_outputs['sel_t_idx'] = pre_processed['t_idx']

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
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.,epsilon=1e-2))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

    use_noise_filter = config['General']['pre-model-type'] == 'noise-filter'
    use_pre_pooling = config['General']['pre-model-type'] == 'pre-pooling'
    model_path = config['General']['pre-model-path']

    if use_noise_filter or use_pre_pooling:
        apply_weights_from_path(model_path, train.keras_model)


###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]
RECORD_FREQUENCY = config['Plotting']['record_frequency']
PLOT_FREQUENCY = config['Plotting']['plot_frequency']

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
        select_metrics=['ExtendedOCLoss*','FullOCLoss_*loss'],
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),

    simpleMetricsCallback(
        output_file=train.outputDir+'/time_pred.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics=['FullOCLoss_*time_std','FullOCLoss_*time_pred_std'],
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),

    simpleMetricsCallback(
        output_file=train.outputDir+'/gooey_metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics='*gooey_*',
        publish=PUBLISHPATH
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/latent_space_metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics='average_distance_*',
        publish=PUBLISHPATH
        ),

    simpleMetricsCallback(
        output_file=train.outputDir+'/non_amb_truth_fraction.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics='*_non_amb_truth_fraction',
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),

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

###############################################################################
### Actual Training ###########################################################
###############################################################################

shutil.copyfile(CONFIGFILE, os.path.join(sys.argv[3], "config.yaml"))

with mlflow.start_run():
    mlflow.tensorflow.autolog()

    for key, value in config.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                mlflow.log_param(key2, value2)
        else:
            mlflow.log_param(key, value)

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
                    layer.fluidity_decay = 0.1
        model, history = train.trainModel(
            nepochs=epochs,
            batchsize=batch_size,
            additional_callbacks=cb
            )
