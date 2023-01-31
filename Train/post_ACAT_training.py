'''
Intended to be used on toy data set found on FI
/mnt/ceph/users/sqasim/Datasets/b4toys/run2/ready/pu200_plus_60_phase_cut_30/train/dataCollection.djcdc

In principle this can be used with data sets newer than January 2022,
but you might want to retrain the preselection model if using a different
data set

As of November 10th, 2022 both classification loss and timing loss do not
work and should be left at 0.0 in the LOSS_OPTIONS
'''
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate

import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from datastructures import TrainData_NanoML, TrainData_PreselectionNanoML

from Layers import RaggedGravNet, DistanceWeightedMessagePassing
from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing 
from Layers import DictModel, ScaledGooeyBatchNorm, LLFullObjectCondensation 
from Layers import CastRowSplits, PlotCoordinates
from Regularizers import AverageDistanceRegularizer
from model_blocks import pre_selection_model, create_outputs
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits
from model_tools import apply_weights_from_path
from callbacks import plotEventDuringTraining, plotClusteringDuringTraining
from callbacks import plotClusterSummary
import globals


###############################################################################
### Configure model and training here #########################################
###############################################################################

BATCHNORM_OPTIONS = {
    'viscosity': 0.1,
    'fluidity_decay': 1e-4,
    'max_viscosity': 0.95,
    'soft_mean': False,
    'variance_only': False,
    'record_metrics': True
    }

LOSS_OPTIONS = {
    'energy_loss_weight': .25,
    'q_min': 1.5,
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0.0,
    'too_much_beta_scale': 1e-5 ,
    'position_loss_weight':1e-5,
    'timing_loss_weight':0.0,
    'beta_loss_scale':2.,
    'beta_push': 0#0.01 #push betas gently up at low values to not lose the gradients
    }

# Configuration for model
PRESELECTION = True
PASS_THROUGH = not PRESELECTION
PRESELECTION_PATH = os.getenv("HGCALML") + '/models/pre_selection_toydetector_ACAT/KERAS_check_best_model.h5'

# Configuration for plotting
RECORD_FREQUENCY = 20
PLOT_FREQUENCY = 50 #plots every 1k batches
PUBLISHPATH = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/July2022_jk/"

# Configuration for training
DENSE_ACTIVATION='elu'
LEARNINGRATE = 4e-5
NBATCH = 100000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    NBATCH = int(NBATCH / 2)

# Configuration of GravNet Blocks
N_NEIGHBOURS = [64, 64, 64]
TOTAL_ITERATIONS = len(N_NEIGHBOURS)
N_CLUSTER_SPACE_COORDINATES = 6

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
        pre_selection = pre_selection_model(
            pre_selection, 
            trainable=False, 
            pass_through=PASS_THROUGH)
    else:
        pre_selection['row_splits'] = CastRowSplits()(pre_selection['row_splits'])
        print(">> preselected dataset will omit pre-selection step")
    
    #just for info what's available
    print('available pre-selection outputs',[k for k in pre_selection.keys()])
                                          
    
    t_spectator_weight = pre_selection['t_spectator_weight']
    rs = pre_selection['row_splits']
                               
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

    for i in range(TOTAL_ITERATIONS):

        x = RaggedGlobalExchange()([x, rs])
        x = Dense(64,activation=DENSE_ACTIVATION)(x)
        x = Dense(64,activation=DENSE_ACTIVATION)(x)
        x = Dense(64,activation=DENSE_ACTIVATION)(x)
        x = ScaledGooeyBatchNorm(**BATCHNORM_OPTIONS)(x)
        x = Concatenate()([c_coords,x])
        
        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=N_NEIGHBOURS[i], 
            n_dimensions=N_CLUSTER_SPACE_COORDINATES, 
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
            
        x = ScaledGooeyBatchNorm(**BATCHNORM_OPTIONS)(x)
        
        x = Dense(64,name='dense_past_mp_'+str(i),activation=DENSE_ACTIVATION)(x)
        x = Dense(64,activation=DENSE_ACTIVATION)(x)
        x = Dense(64,activation=DENSE_ACTIVATION)(x)
        
        x = ScaledGooeyBatchNorm(**BATCHNORM_OPTIONS)(x)
        
        allfeat.append(x)
        
    x = Concatenate()([c_coords] + allfeat)
    x = Dense(64,activation=DENSE_ACTIVATION)(x)
    x = Dense(64,activation=DENSE_ACTIVATION)(x)
    x = Dense(64,activation=DENSE_ACTIVATION)(x)
    
    
    ###########################################################################
    ########### the part below should remain almost unchanged #################
    ########### of course with the exception of the OC loss   #################
    ########### weights                                       #################
    ###########################################################################
    
    x = ScaledGooeyBatchNorm(**BATCHNORM_OPTIONS)(x)
    x = Concatenate()([c_coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES)
    
    # loss
    pred_beta = LLFullObjectCondensation(
        scale=1., 
        use_energy_weights=True, 
        record_metrics=True, 
        print_loss=True, 
        name="FullOCLoss", 
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

    model_outputs = re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=True,
        is_preselected_dataset=is_preselected
        )
    
    return DictModel(inputs=Inputs, outputs=model_outputs)
    
###############################################################################
### Model defined, set up training ############################################
###############################################################################

train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(
        gravnet_model, 
        td=train.train_data.dataclass(), 
        debug_outdir=train.outputDir+'/intplots'
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
    plotEventDuringTraining(
       outputfile=train.outputDir + "/condensation/c_"+str(i),
       samplefile=samplepath,
       after_n_batches=2*PLOT_FREQUENCY,
       batchsize=200000,
       on_epoch_end=False,
       publish=None,
       use_event=i)
    for i in range(5)
    ]



cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics='FullOCLoss_*loss',
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

train.change_learning_rate(LEARNINGRATE)
model, history = train.trainModel(
    nepochs=3, 
    batchsize=NBATCH, 
    additional_callbacks=cb
    )

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.layers:
    if 'FullOCLoss' in l.name:
        l.q_min/=2.

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
