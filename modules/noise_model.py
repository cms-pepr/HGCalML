import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Concatenate

import training_base_hgcal
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from datastructures import TrainData_NanoML, TrainData_PreselectionNanoML

from Layers import RaggedGravNet, DistanceWeightedMessagePassing
from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing 
from Layers import DictModel
from Layers import ConditionalBatchNorm, ConditionalBatchEmbedding
from Layers import CastRowSplits, PlotCoordinates, LLFullObjectCondensation
from Layers import ScaledGooeyBatchNorm2, ConditionalScaledGooeyBatchNorm
from Regularizers import AverageDistanceRegularizer
from model_blocks import pre_selection_model, create_outputs
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits
from noise_filter import noise_filter
from model_tools import apply_weights_from_path
from callbacks import plotEventDuringTraining, plotClusteringDuringTraining
from callbacks import plotClusterSummary
import globals


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
    print('available pre-selection outputs',[k for k in pre_selection.keys()])

    t_spectator_weight = pre_selection['t_spectator_weight']
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
            
        x = ScaledGooeyBatchNorm2()(x)
        
        x = Dense(64,name='dense_past_mp_'+str(i),activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        x = Dense(64,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        
        x = ScaledGooeyBatchNorm2()(x)
        
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
    
    x = ScaledGooeyBatchNorm2()(x)
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
            # 'no_noise_sel': pre_selection['no_noise_sel'],
            # 'no_noise_rs': pre_selection['no_noise_rs'],
            # 'noise_backscatter': pre_selection['noise_backscatter'],
            }
    
    return DictModel(inputs=Inputs, outputs=model_outputs)

