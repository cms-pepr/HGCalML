'''
Intended to be used on toy data set found on FI
/eos/home-p/phzehetn/ML4Reco/Data/V4/Train_cut11/dataCollection.djcdc

As of November 10th, 2022 both classification loss and timing loss do not
work and should be left at 0.0 in the LOSS_OPTIONS
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate

import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from datastructures import TrainData_PreselectionNanoML

from Layers import RaggedGravNet
from Layers import DistanceWeightedMessagePassing
from Layers import DictModel
from Layers import CastRowSplits, PlotCoordinates, LLExtendedObjectCondensation
from Layers import ScaledGooeyBatchNorm2
from Layers import LLFillSpace
from Regularizers import AverageDistanceRegularizer
from model_blocks import create_outputs
from model_blocks import extent_coords_if_needed
from noise_filter import noise_filter
from model_tools import apply_weights_from_path
from callbacks import plotEventDuringTraining, plotClusteringDuringTraining
from callbacks import plotClusterSummary


###############################################################################
### Configure model and training here #########################################
###############################################################################

LOSS_OPTIONS = {
    'energy_loss_weight': .05,
    'q_min': 0.5,
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0.05,
    'too_much_beta_scale': 0.0,
    'position_loss_weight':0.0,
    'timing_loss_weight':0.0,
    'beta_loss_scale':0.1, #1.0
    'beta_push': 0.00,#0.01 or 0.00 #push betas gently up at low values to not lose the gradients
    }

# Configuration for model
PRESELECTION_PATH = '/mnt/home/pzehetner/Models_Good/NoiseTraining0/KERAS_check_best_model.h5'

# Configuration for plotting
RECORD_FREQUENCY = 5
PLOT_FREQUENCY = 10 #plots every 1k batches
PUBLISHPATH = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/July2022_jk/"
PUBLISHPATH = ""

# Configuration for training
DENSE_ACTIVATION='elu'
LEARNINGRATE = 1e-4
NBATCH = 200000
DENSE_REGULARIZER = tf.keras.regularizers.L2(l2=1e-5)
DENSE_REGULARIZER = None

# Configuration of GravNet Blocks
N_NEIGHBOURS = [64, 64]
TOTAL_ITERATIONS = len(N_NEIGHBOURS)
N_CLUSTER_SPACE_COORDINATES = 4
N_GRAVNET = 6
CLUSTER_TRAINABLE = False
EXTENSION_TRAINABLE = True

###############################################################################
### Define model ##############################################################
###############################################################################

def gravnet_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    ############################################################################
    ##################### Input processing, no need to change much here ########
    ############################################################################

    is_preselected = isinstance(td, TrainData_PreselectionNanoML)
    pre_selection = td.interpretAllModelInputs(Inputs, returndict=True)

    #can be loaded - or use pre-selected dataset (to be made)
    if not is_preselected:
        pre_selection = noise_filter(
            pre_selection,
            trainable=False,
            pass_through=False)
    else:
        pre_selection['row_splits'] = CastRowSplits()(pre_selection['row_splits'])
        print(">> preselected dataset will omit pre-selection step")

    #just for info what's available
    print('available pre-selection outputs',list(pre_selection.keys()))

    rs = pre_selection['row_splits']
    is_track = pre_selection['is_track']

    x_in = Concatenate()([pre_selection['coords'],
                          pre_selection['features']])
    x = x_in
    energy = pre_selection['rechit_energy']
    c_coords = pre_selection['coords']#pre-clustered coordinates
    t_idx = pre_selection['t_idx']

    ############################################################################
    ##################### now the actual model goes below ######################
    ############################################################################

    allfeat = []

    #extend coordinates already here if needed
    c_coords = extent_coords_if_needed(c_coords, x, N_CLUSTER_SPACE_COORDINATES)
    x_track = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
    x_hit = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
    is_track_bool = tf.cast(is_track, tf.bool)
    x = tf.where(is_track_bool, x_track, x_hit)

    for i in range(TOTAL_ITERATIONS):

        # x = RaggedGlobalExchange()([x, rs])
        x = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
        x = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
        x = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
        x = ScaledGooeyBatchNorm2(trainable=CLUSTER_TRAINABLE)(x)
        x = Concatenate()([c_coords,x])

        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=N_NEIGHBOURS[i],
            n_dimensions=N_GRAVNET,
            n_filters=64,
            n_propagate=64,
            record_metrics=True,
            coord_initialiser_noise=1e-2,
            use_approximate_knn=False, #weird issue with that for now
            trainable=CLUSTER_TRAINABLE,
            )([x, rs])
        x = Concatenate()([x, xgn])

        gndist = AverageDistanceRegularizer(
            strength=1e-4,
            record_metrics=True,
            trainable=CLUSTER_TRAINABLE,
            )(gndist)

        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i),
            trainable=CLUSTER_TRAINABLE,
            )([gncoords, energy, t_idx, rs])
        x = Concatenate()([gncoords,x])

        x = DistanceWeightedMessagePassing(
            [64,64,32,32,16,16],
            activation=DENSE_ACTIVATION,
            trainable=CLUSTER_TRAINABLE,
            )([x, gnnidx, gndist])

        x = ScaledGooeyBatchNorm2(trainable=CLUSTER_TRAINABLE)(x)

        x = Dense(64,
            name='dense_past_mp_'+str(i),
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
        x = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)
        x = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER,
            trainable=CLUSTER_TRAINABLE)(x)

        x = ScaledGooeyBatchNorm2(trainable=CLUSTER_TRAINABLE)(x)

        allfeat.append(x)

    x = Concatenate()(allfeat)
    x = Dense(64,
        name='Last_Dense_1',
        activation=DENSE_ACTIVATION,
        trainable=CLUSTER_TRAINABLE)(x)
    x = Dense(64,
        name='Last_Dense_2',
        activation=DENSE_ACTIVATION,
        trainable=CLUSTER_TRAINABLE)(x)
    x = Dense(64,
        name='Last_Dense_3',
        activation=DENSE_ACTIVATION,
        trainable=CLUSTER_TRAINABLE)(x)
    x = ScaledGooeyBatchNorm2(trainable=CLUSTER_TRAINABLE)(x)

    y = Concatenate()(allfeat)
    y = tf.stop_gradient(y)
    y = Dense(64,
        name='Last_Dense_1_energy',
        activation=DENSE_ACTIVATION,
        trainable=EXTENSION_TRAINABLE)(y)
    y = Dense(64,
        name='Last_Dense_2_energy',
        activation=DENSE_ACTIVATION,
        trainable=EXTENSION_TRAINABLE)(y)
    y = Dense(64,
        name='Last_Dense_3_energy',
        activation=DENSE_ACTIVATION,
        trainable=EXTENSION_TRAINABLE)(y)
    y = ScaledGooeyBatchNorm2(trainable=EXTENSION_TRAINABLE)(y)


    ###########################################################################
    ########### the part below should remain almost unchanged #################
    ########### of course with the exception of the OC loss   #################
    ########### weights                                       #################
    ###########################################################################

    pred_beta, pred_ccoords, pred_dist, \
        _, _, _, \
        pred_pos, pred_time, pred_time_unc, _ = \
        create_outputs(x,
            n_ccoords=N_CLUSTER_SPACE_COORDINATES,
            fix_distance_scale=True,
            trainable=CLUSTER_TRAINABLE,)

    _, _, _, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        _, _, _, pred_id = \
        create_outputs(y,
            n_ccoords=N_CLUSTER_SPACE_COORDINATES,
            fix_distance_scale=True,
            trainable=EXTENSION_TRAINABLE)

    # pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)([pred_ccoords, rs, t_idx])

    # loss
    pred_beta = LLExtendedObjectCondensation(
        scale=1.,
        use_energy_weights=True,
        record_metrics=True,
        print_loss=True,
        name="ExtendedOCLoss",
        **LOSS_OPTIONS
        )( # oc output and payload
            [pred_beta,
             pred_ccoords,
             pred_dist,
             pred_energy_corr,
             pred_energy_low_quantile,
             pred_energy_high_quantile,
             pred_pos,
             pred_time,
             pred_time_unc,
             pred_id] +
            [energy] +
            # truth information
            [pre_selection['t_idx'] ,
             pre_selection['t_energy'] ,
             pre_selection['t_pos'] ,
             pre_selection['t_time'] ,
             pre_selection['t_pid'] ,
             pre_selection['t_spectator_weight'],
             pre_selection['t_fully_contained'],
             pre_selection['t_rec_energy'],
             pre_selection['t_is_unique'],
             pre_selection['row_splits'] ]
            )

    #fast feedback
    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation'
        )([pred_ccoords, pred_beta,pre_selection['t_idx'], rs])
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
            'row_splits': pre_selection['row_splits'],
            'no_noise_sel': pre_selection['no_noise_sel'],
            'no_noise_rs': pre_selection['no_noise_rs'],
            # 'noise_backscatter': pre_selection['noise_backscatter'],
            }

    return DictModel(inputs=Inputs, outputs=model_outputs)

###############################################################################
### Model defined, set up training ############################################
###############################################################################

train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(
        gravnet_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        )
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.,epsilon=1e-2))
    train.compileModel(learningrate=LEARNINGRATE)
    train.keras_model.summary()

    if not isinstance(train.train_data.dataclass(), TrainData_PreselectionNanoML):
        apply_weights_from_path(PRESELECTION_PATH, train.keras_model)

###############################################################################
### Create Callbacks ##########################################################
###############################################################################

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]
cb = []
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
### Start training ############################################################
###############################################################################

print("Batch size: ", NBATCH)
train.change_learning_rate(LEARNINGRATE)
model, history = train.trainModel(
    nepochs=100,
    batchsize=NBATCH,
    additional_callbacks=cb
    )

print("freeze BN")
# Note the submodel here its not just train.keras_model
# for l in train.keras_model.layers:
#   if 'FullOCLoss' in l.name:
#       l.q_min/=2.

train.change_learning_rate(LEARNINGRATE/2.)
model, history = train.trainModel(
    nepochs=100,
    batchsize=NBATCH,
    additional_callbacks=cb
    )

train.change_learning_rate(LEARNINGRATE/2.)
model, history = train.trainModel(
    nepochs=100,
    batchsize=NBATCH,
    additional_callbacks=cb
    )
model, history = train.trainModel(
    nepochs=100,
    batchsize=NBATCH,
    additional_callbacks=cb
    )
