'''

Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

import tensorflow as tf

from tensorflow.keras.layers import Dense, Concatenate

from DeepJetCore.DJCLayers import StopGradient

from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing, DictModel
from Layers import RaggedGravNet, ScaledGooeyBatchNorm2 
from Regularizers import AverageDistanceRegularizer
from LossLayers import LLExtendedObjectCondensation
from DebugLayers import PlotCoordinates

from model_blocks import condition_input, extent_coords_if_needed, create_outputs, re_integrate_to_full_hits

from callbacks import plotClusterSummary

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
import os

from model_blocks import tiny_pc_pool

from model_tools import apply_weights_from_path

#loss options:
loss_options={
    # here and in the following energy = momentum
    'energy_loss_weight': 0.,
    'q_min': 1.,
    # addition to original OC, adds average position for clusterin
    # usually 0.5 is a reasonable value to break degeneracies 
    # and keep training smooth enough
    'use_average_cc_pos': 0.5, 
    'classification_loss_weight':0.0,
    'position_loss_weight':0.,
    'timing_loss_weight':0.,
    'beta_loss_scale':1.,
    # these weights will downweight low energies, for a 
    # training sample with a good energy distribution,
    # this won't be needed.
    'use_energy_weights': False,
    # this is the standard repulsive hinge loss from the paper
    'implementation': 'hinge' 
    }


# elu behaves well, likely fine
dense_activation='elu'

# record internal metrics every N batches
record_frequency=10
# plot every M times, metrics were recorded. In other words, 
# plotting will happen every M*N batches
plotfrequency=50 

learningrate = 1e-4

# this is the maximum number of hits (points) per batch,
# not the number of events (samples). This is safer w.r.t. 
# memory
nbatch = 170000


#iterations of gravnet blocks
n_neighbours=[64,64]

# 3 is a bit low but nice in the beginning since it can be plotted
n_cluster_space_coordinates = 3
n_gravnet_dims = 3

do_presel = True
PRESELECTION_PATH = os.getenv("HGCALML")+'/models/tiny_pc_pool/model.h5'

def gravnet_model(Inputs,
                  td,
                  debug_outdir=None,
                  plot_debug_every=record_frequency*plotfrequency,
                  ):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    input_list = td.interpretAllModelInputs(Inputs,returndict=True)
    input_list = condition_input(input_list, no_scaling=True)
    
    trans, input_list = tiny_pc_pool(input_list, pass_through=not do_presel)
    
                              
    #just for info what's available, prints once
    print('available inputs',[k for k in input_list.keys()])
                                          
    rs = input_list['row_splits']
    t_idx = input_list['t_idx']
    energy = input_list['rechit_energy']
    c_coords = input_list['coords']
    
    ## build inputs
                               
    x_in = Concatenate()([input_list['coords'],
                          input_list['features']])
    
    x_in = ScaledGooeyBatchNorm2(
        fluidity_decay=0.1 #freeze out quickly, just to get good input preprocessing
        )(x_in)
                    
    x = x_in       
    
    c_coords = ScaledGooeyBatchNorm2(
        fluidity_decay=0.1 #same here
        )(c_coords)
        
    
    ####################################################################################
    ##################### now the actual model goes below ##############################
    ####################################################################################
    
    # output of each iteration will be concatenated
    allfeat = []
    
    # extend coordinates already here if needed, just as a good starting point
    c_coords = extent_coords_if_needed(c_coords, x, n_gravnet_dims)

    for i in range(len(n_neighbours)):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Concatenate()([c_coords,x]) #give a good starting point
        x = ScaledGooeyBatchNorm2()(x)
        
        xgn, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=n_neighbours[i],
                                                 n_dimensions=n_gravnet_dims,
                                                 n_propagate=64, #this is the number of features that are exchanged
                                                 n_filters=64, #output dense
                                                 feature_activation = 'elu',
                                                 )([x, rs])
        
        x = Concatenate()([x,xgn])    
                                     
        # mostly to record average distances etc. can be used to force coordinates
        # to be within reasonable range (but usually not needed)                                      
        gndist = AverageDistanceRegularizer(strength=1e-6,
                                            record_metrics=True
                                            )(gndist)
        
        #for information / debugging, can also be safely removed                                    
        gncoords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
                                   name='gn_coords_'+str(i))([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs]) 
        # we have to pass them downwards, otherwise the layer above gets optimised away
        # but we don't want the gradient to be disturbed, so it gets stopped
        gncoords = StopGradient()(gncoords)
        x = Concatenate()([gncoords,x])           
        
        # this repeats the distance weighted message passing step from gravnet
        # on the same graph topology
        x = DistanceWeightedMessagePassing([64,64],
                                           activation=dense_activation
                                           )([x,gnnidx,gndist])
            
        x = ScaledGooeyBatchNorm2()(x)
        
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        
        x = ScaledGooeyBatchNorm2()(x)
        
        allfeat.append(x)
        
        
    
    x = Concatenate()([c_coords]+allfeat)#gives a prior to the clustering coords
    #create one global feature vector
    xg = Dense(512,activation=dense_activation,name='glob_dense_'+str(i))(x)
    x = RaggedGlobalExchange()([xg, rs])
    x = Concatenate()([x,xg])
    # last part of network
    x = Dense(64,activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)
    x = Dense(64,activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)
    x = Dense(64,activation=dense_activation)(x)
    x = ScaledGooeyBatchNorm2()(x)
    
    
    #######################################################################
    ########### the part below should remain almost unchanged #############
    ########### of course with the exception of the OC loss   #############
    ########### weights                                       #############
    #######################################################################
    
    #use a standard batch norm at the last stage
    
    
    pred_beta, pred_ccoords, pred_dist,\
    pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile,\
    pred_pos, pred_time, pred_time_unc, pred_id = create_outputs(x, n_ccoords=n_cluster_space_coordinates)
    
    # loss
    pred_beta = LLExtendedObjectCondensation(scale=1.,
                                         record_metrics=True,
                                         print_loss=True,
                                         name="FullOCLoss",
                                         **loss_options
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy_corr,pred_energy_low_quantile,pred_energy_high_quantile,
         pred_pos, pred_time, pred_time_unc,
         pred_id] +
        [energy]+
        # truth information
        [input_list['t_idx'] ,
         input_list['t_energy'] ,
         input_list['t_pos'] ,
         input_list['t_time'] ,
         input_list['t_pid'] ,
         input_list['t_spectator_weight'],
         input_list['t_fully_contained'],
         input_list['t_rec_energy'],
         input_list['t_is_unique'],
         input_list['row_splits']])
                                         
    # fast feedback
    pred_ccoords = PlotCoordinates(plot_every=plot_debug_every, outdir = debug_outdir,
                    name='condensation_coords')([pred_ccoords, pred_beta,input_list['t_idx'],
                                          rs])                                    

    # just to have a defined output, only adds names
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
            'row_splits': input_list['row_splits'], #are these the selected ones or not?
            'no_noise_sel': trans['sel_idx_up'],
            'no_noise_rs': trans['rs_down'], #unclear what that actually means?
            'sel_idx': trans['sel_idx_up'], #just a duplication but more intuitive to understand
            'sel_t_idx': input_list['t_idx'] #for convenience
            # 'noise_backscatter': pre_selection['noise_backscatter'],
            }
    
    return DictModel(inputs=Inputs, outputs=model_outputs)
    


import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.,epsilon=1e-2))
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
    if do_presel:
        train.keras_model = apply_weights_from_path(PRESELECTION_PATH, train.keras_model)
    

verbosity = 2
import os

publishpath = None #this can be an ssh reachable path (be careful: needs tokens / keypairs)

# establish callbacks


cb = [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='FullOCLoss_*loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/latent_space_metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='average_distance_*',
        publish=publishpath
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    
    
    ]


cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=200
        )
    ]

#cb=[]

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=3,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
#for l in train.keras_model.layers:
#    if 'FullOCLoss' in l.name:
#        l.q_min/=2.

train.change_learning_rate(learningrate/2.)


model, history = train.trainModel(nepochs=121,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)
    



