'''

Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''
from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import Reshape,BatchNormalization, Dropout, Add

from GravNetLayersRagged import MultiAttentionGravNetAdd,WeightFeatures,WeightedNeighbourMeans,DownSample, CreateIndexFromMajority, ProcessFeatures, SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing

from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from datastructures import TrainData_NanoML

from callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining, plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR

from model_blocks import create_outputs
from GravNetLayersRagged import MultiBackScatter,EdgeCreator, EdgeSelector
from GravNetLayersRagged import GroupScoreFromEdgeScores,NoiseFilter
from GravNetLayersRagged import ProcessFeatures,SoftPixelCNN, RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing,MultiBackScatterOrGather

from GravNetLayersRagged import NeighbourGroups,AccumulateNeighbours,SelectFromIndices
from GravNetLayersRagged import RecalcDistances, ElementScaling, RemoveSelfRef, CastRowSplits

from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CreateGlobalIndices, SelectFromIndices, MultiBackScatter, KNN, MessagePassing, DictModel
from Layers import GausActivation,GooeyBatchNorm, ScaledGooeyBatchNorm2 #make a new line
from model_blocks import create_outputs
from Regularizers import AverageDistanceRegularizer

from model_blocks import pre_selection_model
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits

from LossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from LossLayers import LLFullObjectCondensation, LLClusterCoordinates,LLEdgeClassifier

from DebugLayers import PlotCoordinates

from datastructures import TrainData_PreselectionNanoML

from GravNetLayersRagged import CastRowSplits


import globals
if False: #for testing
    globals.acc_ops_use_tf_gradients = True 
    globals.knn_ops_use_tf_gradients = True
'''

make this about coordinate shifts


'''


#loss options:
loss_options={
    'energy_loss_weight': .25,
    'q_min': 1.5,
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0.0,
    'too_much_beta_scale': 1e-5 ,
    'position_loss_weight':1e-5,
    'timing_loss_weight':0.1,
    'beta_loss_scale':2.,
    'beta_push': 0#0.01 #push betas gently up at low values to not lose the gradients
    }


#elu behaves much better when training
dense_activation='elu'

record_frequency=20
plotfrequency=50 #plots every 1k batches

learningrate = 1e-6
nbatch = 100000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

#iterations of gravnet blocks
n_neighbours=[64,64]
total_iterations = len(n_neighbours)

n_cluster_space_coordinates = 3


def gravnet_model(Inputs,
                  td,
                  debug_outdir=None,
                  plot_debug_every=2000,
                  ):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    is_preselected = isinstance(td, TrainData_PreselectionNanoML)

    pre_selection = td.interpretAllModelInputs(Inputs,returndict=True)
                                                
    #can be loaded - or use pre-selected dataset (to be made)
    if not is_preselected:
        pre_selection = pre_selection_model(pre_selection,trainable=False,pass_through=False)
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
    
    ####################################################################################
    ##################### now the actual model goes below ##############################
    ####################################################################################
    
    allfeat = []
    
    
    
    #extend coordinates already here if needed
    c_coords = extent_coords_if_needed(c_coords, x, n_cluster_space_coordinates)
        

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = ScaledGooeyBatchNorm2()(x)
        ### reduction done
        
        n_dims = 6
        #exchange information, create coordinates
        x = Concatenate()([c_coords,x])
        xgn, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=n_neighbours[i],
                                                 n_dimensions=n_dims,
                                                 n_filters=64,
                                                 n_propagate=64,
                                                 record_metrics=True,
                                                 coord_initialiser_noise=1e-2,
                                                 use_approximate_knn=False #weird issue with that for now
                                                 )([x, rs])
        
        x = Concatenate()([x,xgn])                                                      
        #just keep them in a reasonable range  
        #safeguard against diappearing gradients on coordinates                                       
        gndist = AverageDistanceRegularizer(strength=1e-4,
                                            record_metrics=True
                                            )(gndist)
                                            
        gncoords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
                                   name='gn_coords_'+str(i))([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs]) 
        x = Concatenate()([gncoords,x])           
        
        x = DistanceWeightedMessagePassing([64,64,32,32,16,16],
                                           activation=dense_activation
                                           )([x,gnnidx,gndist])
            
        x = ScaledGooeyBatchNorm2()(x)
        
        x = Dense(64,name='dense_past_mp_'+str(i),activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        
        x = ScaledGooeyBatchNorm2()(x)
        
        
        allfeat.append(x)
        
        
    
    x = Concatenate()([c_coords]+allfeat)
    #do one more exchange with all
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    
    
    #######################################################################
    ########### the part below should remain almost unchanged #############
    ########### of course with the exception of the OC loss   #############
    ########### weights                                       #############
    #######################################################################
    
    #use a standard batch norm at the last stage
    x = ScaledGooeyBatchNorm2()(x)
    x = Concatenate()([c_coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist,\
    pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile,\
    pred_pos, pred_time, pred_time_unc, pred_id = create_outputs(x, n_ccoords=n_cluster_space_coordinates)
    
    # loss
    pred_beta = LLFullObjectCondensation(scale=1.,
                                         use_energy_weights=True,
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
        [pre_selection['t_idx'] ,
         pre_selection['t_energy'] ,
         pre_selection['t_pos'] ,
         pre_selection['t_time'] ,
         pre_selection['t_pid'] ,
         pre_selection['t_spectator_weight'],
         pre_selection['t_fully_contained'],
         pre_selection['t_rec_energy'],
         pre_selection['t_is_unique'],
         pre_selection['row_splits']])
                                         
    #fast feedback
    pred_ccoords = PlotCoordinates(plot_every=plot_debug_every, outdir = debug_outdir,
                    name='condensation')([pred_ccoords, pred_beta,pre_selection['t_idx'],
                                          rs])                                    

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
    
    if not isinstance(train.train_data.dataclass(), TrainData_PreselectionNanoML):
        from model_tools import apply_weights_from_path
        import os
        path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_june22/KERAS_model.h5'
        apply_weights_from_path(path_to_pretrained,train.keras_model)
    

verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


# publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/July2022_jk/"
publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/July2022_pz/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

cb = []


#cb += [plotClusteringDuringTraining(
#    use_backgather_idx=8 + i,
#    outputfile=train.outputDir + "/localclust/cluster_" + str(i) + '_',
#    samplefile=samplepath,
#    after_n_batches=500,
#    on_epoch_end=False,
#    publish=None,
#    use_event=0)
#    for i in [0, 2, 4]]
#
#cb += [
#    plotEventDuringTraining(
#        outputfile=train.outputDir + "/condensation/c_"+str(i),
#        samplefile=samplepath,
#        after_n_batches=2*plotfrequency,
#        batchsize=200000,
#        on_epoch_end=False,
#        publish=None,
#        use_event=i)
#for i in range(5)
#]
#


from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='FullOCLoss_*loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/time_pred.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics=['FullOCLoss_*time_std','FullOCLoss_*time_pred_std'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/gooey_metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='*gooey_*',
        publish=publishpath
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/latent_space_metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='average_distance_*',
        publish=publishpath
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/non_amb_truth_fraction.html',
        record_frequency= record_frequency,
        plot_frequency = plotfrequency,
        select_metrics='*_non_amb_truth_fraction',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    
    
    #if approxime knn is used
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/slicing_knn_metrics.html',
    #    record_frequency= record_frequency,
    #    plot_frequency = plotfrequency,
    #    publish=publishpath,
    #    select_metrics='*_bins'
    #),
    
    
    ]

from callbacks import plotClusterSummary

cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=1000
        )
    ]

#cb=[]

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=3,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.layers:
    if 'FullOCLoss' in l.name:
        l.q_min/=2.

train.change_learning_rate(learningrate/2.)
nbatch = 160000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

model, history = train.trainModel(nepochs=121,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)
    



