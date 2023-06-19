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
from Layers import CastRowSplits, PlotCoordinates
from Layers import LLFullObjectCondensation as  LLExtendedObjectCondensation
from Layers import ScaledGooeyBatchNorm2
from Layers import LLFillSpace
from Regularizers import AverageDistanceRegularizer
from model_blocks import create_outputs
from model_blocks import extent_coords_if_needed
from model_blocks import tiny_pc_pool, condition_input
from model_tools import apply_weights_from_path
from callbacks import plotEventDuringTraining, plotClusteringDuringTraining
from callbacks import plotClusterSummary
import os


###############################################################################
### Configure model and training here #########################################
###############################################################################

LOSS_OPTIONS = {
    'energy_loss_weight': .0001,
    'q_min': 0.5,
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0., # to make it work0.5,
    'too_much_beta_scale': 0.0,
    'position_loss_weight':0.0,
    'timing_loss_weight':0.0,
    'beta_loss_scale':1., #2.0
    'beta_push': 0.00,#0.01 or 0.00 #push betas gently up at low values to not lose the gradients
    }

# Configuration for model
PRESELECTION_PATH = os.getenv("HGCALML")+'/models/tiny_pc_pool/model.h5'

PRESELECTION_PATH = '/afs/cern.ch/user/j/jkiesele/Cernbox/www/files/temp/June2023/tiny_again2/KERAS_check_model_last.h5'

# Configuration for plotting
RECORD_FREQUENCY = 10
PLOT_FREQUENCY = 10 #plots every 600 batches -> roughly 10 minutes
PUBLISHPATH = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/June2023/"
PUBLISHPATH = None

# Configuration for training
DENSE_ACTIVATION='elu'
LEARNINGRATE = 1e-4
NBATCH = 70000#200000
DENSE_REGULARIZER = tf.keras.regularizers.L2(l2=1e-5)
DENSE_REGULARIZER = None


# Configuration of GravNet Blocks
N_NEIGHBOURS = [64, 64]
TOTAL_ITERATIONS = len(N_NEIGHBOURS)
N_CLUSTER_SPACE_COORDINATES = 4
N_GRAVNET = 6

###############################################################################
### Define model ##############################################################
###############################################################################

def gravnet_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    ############################################################################
    ##################### Input processing, no need to change much here ########
    ############################################################################

    pre_selection = td.interpretAllModelInputs(Inputs, returndict=True)

    pre_selection = condition_input(pre_selection, no_scaling=True)
    trans, pre_selection = tiny_pc_pool(
        pre_selection,
        record_metrics=True,
        #trainable=True
        )#train in one go.. what is up with the weight loading?
    
    #just for info what's available
    print('available pre-selection outputs',list(pre_selection.keys()))

    rs = pre_selection['row_splits']
    is_track = pre_selection['is_track']

    x_in = Concatenate()([pre_selection['prime_coords'],
                          pre_selection['features']])
    
    x_in = ScaledGooeyBatchNorm2()(x_in)
    x = x_in
    energy = pre_selection['rechit_energy']
    c_coords = pre_selection['prime_coords']#pre-clustered coordinates
    t_idx = pre_selection['t_idx']

    ############################################################################
    ##################### now the actual model goes below ######################
    ############################################################################

    allfeat = []

    #extend coordinates already here if needed
    c_coords = extent_coords_if_needed(c_coords, x, N_CLUSTER_SPACE_COORDINATES)
    x_track = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
    x_hit = Dense(64,
            activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
    is_track_bool = tf.cast(is_track, tf.bool)
    x = tf.where(is_track_bool, x_track, x_hit)

    for i in range(TOTAL_ITERATIONS):

        # x = RaggedGlobalExchange()([x, rs])
        x = Dense(64, activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = ScaledGooeyBatchNorm2()(x)
        x = Concatenate()([c_coords,x])

        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=N_NEIGHBOURS[i],
            n_dimensions=N_GRAVNET,
            n_filters=64,
            n_propagate=64,
            record_metrics=True,
            coord_initialiser_noise=1e-2,
            use_approximate_knn=False #weird issue with that for now
            )([x, rs])
        x = Concatenate()([x, xgn])

        gndist = AverageDistanceRegularizer(
            strength=1e-4,
            record_metrics=True
            )(gndist)

        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i)
            )([gncoords, energy, t_idx, rs])
        x = Concatenate()([gncoords,x])

        x = DistanceWeightedMessagePassing(
            [64,64,32,32,16,16],
            activation=DENSE_ACTIVATION
            )([x, gnnidx, gndist])

        x = ScaledGooeyBatchNorm2()(x)

        x = Dense(64,name='dense_past_mp_'+str(i),activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)

        x = ScaledGooeyBatchNorm2()(x)

        allfeat.append(x)

    x = Concatenate()(allfeat)
    x = Dense(64, name='Last_Dense_1', activation=DENSE_ACTIVATION)(x)
    x = Dense(64, name='Last_Dense_2', activation=DENSE_ACTIVATION)(x)
    x = Dense(64, name='Last_Dense_3', activation=DENSE_ACTIVATION)(x)


    ###########################################################################
    ########### the part below should remain almost unchanged #################
    ########### of course with the exception of the OC loss   #################
    ########### weights                                       #################
    ###########################################################################

    x = ScaledGooeyBatchNorm2()(x)
    # x = Concatenate()([x])

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES, fix_distance_scale=True)

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
            'row_splits': pre_selection['row_splits'], #are these the selected ones or not?
            'no_noise_sel': trans['sel_idx_up'],
            'no_noise_rs': trans['rs_down'], #unclear what that actually means?
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
        train.keras_model = apply_weights_from_path(PRESELECTION_PATH, train.keras_model)
        
    exit()

###############################################################################
### Create Callbacks ##########################################################
###############################################################################

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
if PUBLISHPATH is not None:
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

    # collect all pre pooling metrics here
    simpleMetricsCallback(
        output_file=train.outputDir+'/pgp_metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics='*pre_graph_pool*',
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