'''
Intended to be used on toy data set found on FI
/eos/home-p/phzehetn/ML4Reco/Data/V4/Train_cut11/dataCollection.djcdc

As of November 10th, 2022 both classification loss and timing loss do not
work and should be left at 0.0 in the LOSS_OPTIONS
'''


import globals
if True: #for testing
    #globals.acc_ops_use_tf_gradients = True 
    globals.knn_ops_use_tf_gradients = True
    
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Dropout

import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient
from datastructures import TrainData_PreselectionNanoML

from Layers import RaggedGravNet, RaggedGlobalExchange
from Layers import DistanceWeightedMessagePassing
from Layers import DictModel, SphereActivation, Multi
from Layers import CastRowSplits, PlotCoordinates
from Layers import LLFullObjectCondensation as  LLExtendedObjectCondensation
from Layers import ScaledGooeyBatchNorm2, Sqrt
from Layers import LLFillSpace, SphereActivation
from Regularizers import AverageDistanceRegularizer
from model_blocks import create_outputs
from model_blocks import extent_coords_if_needed
from model_blocks import tiny_pc_pool, condition_input
from model_tools import apply_weights_from_path

from callbacks import NanSweeper, DebugPlotRunner
from Layers import layernorm
import os


###############################################################################
### Configure model and training here #########################################
###############################################################################

LOSS_OPTIONS = {
    'energy_loss_weight': 1e-6,
    'q_min': 0.5,
    'use_average_cc_pos': 0.5,
    'classification_loss_weight':0., # to make it work0.5,
    'too_much_beta_scale': 0.0,
    'position_loss_weight':0.,
    'timing_loss_weight':0.0,
    'beta_loss_scale':1., #2.0
    'implementation': 'hinge' #'hinge_manhatten'#'hinge'#old school
    }

BATCHNORM_OPTIONS = {
    'max_viscosity': 0.5 #keep very batchnorm like
    }

# Configuration for model
PRESELECTION_PATH = os.getenv("HGCALML")+'/models/tiny_pc_pool/model_no_nan.h5'#model.h5'

# Configuration for plotting
RECORD_FREQUENCY = 20
PLOT_FREQUENCY = 10 #plots every 200 batches -> roughly 3 minutes
PUBLISHPATH = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/June2023/"
#PUBLISHPATH = None

# Configuration for training
DENSE_ACTIVATION='elu' #layernorm #'elu'
LEARNINGRATE = 5e-3
LEARNINGRATE2 = 1e-3
LEARNINGRATE3 = 1e-4
NBATCH = 200000#200000
DENSE_REGULARIZER = tf.keras.regularizers.L2(l2=1e-5)
DENSE_REGULARIZER = None

# Configuration of GravNet Blocks
N_NEIGHBOURS = [256, 256, 256]
TOTAL_ITERATIONS = len(N_NEIGHBOURS)
N_CLUSTER_SPACE_COORDINATES = 3
N_GRAVNET = 3

###############################################################################
### Define model ##############################################################
###############################################################################

def gravnet_model(Inputs, td, debug_outdir=None, 
                  plot_debug_every=RECORD_FREQUENCY*PLOT_FREQUENCY,
                  publish = None):
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
    
    #x_in = Concatenate()([x_in, is_track, SphereActivation()(x_in)])
    x_in = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x_in)
    x_in = Dense(128, activation=DENSE_ACTIVATION)(x_in)
    x_in = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x_in)
    x = x_in
    energy = pre_selection['rechit_energy']
    c_coords = pre_selection['prime_coords']#pre-clustered coordinates
    c_coords = ScaledGooeyBatchNorm2(
        fluidity_decay=0.5, #can freeze almost immediately
        )(c_coords)
    t_idx = pre_selection['t_idx']

    c_coords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='input_c_coords',
            publish = publish
            )([c_coords, energy, t_idx, rs])

    ############################################################################
    ##################### now the actual model goes below ######################
    ############################################################################

    allfeat = []

    #extend coordinates already here if needed, starting point for gravnet
    
    c_coords = extent_coords_if_needed(c_coords, x, N_GRAVNET)
    
    ## not needed, embedding already done in the pre-pooling
    #x_track = Dense(64,
    #        activation=DENSE_ACTIVATION,
    #        kernel_regularizer=DENSE_REGULARIZER)(x)
    #x_hit = Dense(64,
    #        activation=DENSE_ACTIVATION,
    #        kernel_regularizer=DENSE_REGULARIZER)(x)
    #is_track_bool = tf.cast(is_track, tf.bool)
    #x = tf.where(is_track_bool, x_track, x_hit)

    for i in range(TOTAL_ITERATIONS):

        #x,n = SphereActivation(return_norm=True)(x)
        
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        #x = Dropout(0.1)(x)
        x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
          
        x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        #x = Dropout(0.1)(x) 
        
        x = Concatenate()([c_coords,x]) 

        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=N_NEIGHBOURS[i],
            n_dimensions=N_GRAVNET,
            n_filters=64,
            n_propagate=64,
            coord_initialiser_noise=1e-5,
            #sumwnorm = True,
            )([x, rs])
            
        
        xgn = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(xgn)
        
        gndist = AverageDistanceRegularizer(
            strength=1e-2,
            record_metrics=True
            )(gndist)

        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i),
            publish = publish
            )([gncoords, energy, t_idx, rs])
        gncoords = StopGradient()(gncoords)
        
        x = Concatenate()([gncoords,xgn,x])
        
        #does the same but with batch norm
        for nn in [64,64,64,64]:
            
            #d_mult = ScalarMultiply(2.)(Dense(1,activation='sigmoid')(x))
            #gndist = Multi()([gndist,d_mult])#scale distances here dynamically
            x = SphereActivation()(x)
            x = DistanceWeightedMessagePassing(
                [nn],
                activation=DENSE_ACTIVATION,
                #sumwnorm = True,
                )([x, gnnidx, gndist])
            x = Dense(128,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
            #x = Dropout(0.1)(x)
            #
        
        x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        x = Dense(64,name='dense_past_mp_'+str(i),activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        #x = Dropout(0.25)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        
        #x = Multi()([x,n]) #back to full space
        x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)

        allfeat.append(x)

    x = Concatenate()(allfeat)
    x = RaggedGlobalExchange()([x,rs])
    #x = Concatenate()([x,SphereActivation()(x)])
    x = Dense(64, name='Last_Dense_1', activation=DENSE_ACTIVATION)(x)
    x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
    x = Dense(64, name='Last_Dense_2', activation=DENSE_ACTIVATION)(x)
    x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
    #x = Dropout(0.1)(x)
    x = Dense(64, name='Last_Dense_3', activation=DENSE_ACTIVATION)(x)#we want this to be not bounded
    #x = Dropout(0.1)(x)
    ###########################################################################
    ########### the part below should remain almost unchanged #################
    ########### of course with the exception of the OC loss   #################
    ########### weights                                       #################
    ###########################################################################
    
    x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
    # x = Concatenate()([x])

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES, fix_distance_scale=True)

    pred_ccoords = LLFillSpace(maxhits=1000, runevery=1, 
                               scale=0.01,
                               record_metrics=True,
                               print_loss=True,
                               print_batch_time=True)([pred_ccoords, rs, 
                                                                       pre_selection['t_idx']])


    # loss
    pred_beta = LLExtendedObjectCondensation(
        scale=1.,
        use_energy_weights=False,#well distributed anyways
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
        name='condensation',
        publish = publish
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
            'sel_idx': trans['sel_idx_up'], #just a duplication but more intuitive to understand
            'sel_t_idx': pre_selection['t_idx'] #for convenience
            # 'noise_backscatter': pre_selection['noise_backscatter'],
            }

    return DictModel(inputs=Inputs, outputs=model_outputs)

###############################################################################
### Model defined, set up training ############################################
###############################################################################

train = training_base_hgcal.HGCalTraining()

if PUBLISHPATH is not None:
    PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]
    
if not train.modelSet():
    train.setModel(
        gravnet_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        publish = PUBLISHPATH
        )
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=2.,
                                                       epsilon=1e-2))
    train.compileModel(learningrate=LEARNINGRATE)
    train.keras_model.summary()

    if not isinstance(train.train_data.dataclass(), TrainData_PreselectionNanoML):
        train.keras_model = apply_weights_from_path(PRESELECTION_PATH, train.keras_model)
        
    #exit()

###############################################################################
### Create Callbacks ##########################################################
###############################################################################

val_samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
cb = []


cb += [
    
    NanSweeper(),#this takes a bit of time checking each batch but could be worth it
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics=['ExtendedOCLoss*','FullOCLoss_*loss','*ll_fill_space*'],
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/gndist.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics=['*average_distance*'],
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
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    
    #triggers debug plots within the model on a specific sample
    DebugPlotRunner(
        plot_frequency = 2, #testing
        sample = val_samplepath
        )
    ]


###############################################################################
### Start training ############################################################
###############################################################################

print("Batch size: ", NBATCH)
train.change_learning_rate(LEARNINGRATE)
model, history = train.trainModel(
    nepochs=2,
    batchsize=NBATCH,
    additional_callbacks=cb
    )

train.change_learning_rate(LEARNINGRATE2)
model, history = train.trainModel(
    nepochs=4,
    batchsize=NBATCH,
    additional_callbacks=cb
    )


train.change_learning_rate(LEARNINGRATE3)
model, history = train.trainModel(
    nepochs=6,
    batchsize=NBATCH,
    additional_callbacks=cb
    )

for l in train.keras_model.layers:
    if isinstance(l, ScaledGooeyBatchNorm2):
        l.trainable = False
        
train.compileModel(learningrate=LEARNINGRATE3)

model, history = train.trainModel(
    nepochs=8,
    batchsize=NBATCH,
    additional_callbacks=cb
    )

exit()
