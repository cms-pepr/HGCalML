'''

Compatible with the dataset here:
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Jan2022_production_3

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Jan2022_production_3

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
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer

from model_blocks import first_coordinate_adjustment, reduce, pre_selection_model_full
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits

from LossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from LossLayers import LLFullObjectCondensation, LLClusterCoordinates,LLEdgeClassifier

from DebugLayers import PlotCoordinates

'''

make this about coordinate shifts


'''


batchnorm_options={
    'viscosity': 0.2,
    'fluidity_decay': 5e-4,
    'max_viscosity': 0.95,
    'soft_mean': True,
    'variance_only': False,
    'record_metrics': True,
    'soft_mean_turn_on': 10.,
    }

#loss options:
loss_options={
    'q_min': 2.5,
    'alt_energy_weight': False,
    'use_average_cc_pos': 0.99
    }



def gravnet_model(Inputs,
                  td,
                  use_multigrav = True,
                  total_iterations=2,
                  debug_outdir=None,
                  plot_debug_every=1000,
                  ):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    
    
    orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([orig_inputs['t_spectator'], 
                                                        orig_inputs['t_idx']])
                                                     
    orig_inputs['t_spectator_weight'] = orig_t_spectator_weight                                                 
    #can be loaded - or use pre-selected dataset (to be made)
    pre_selection = pre_selection_model_full(orig_inputs,trainable=False)
    
    #just for info what's available
    print([k for k in pre_selection.keys()])
                                          
    
    t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([pre_selection['t_spectator'], 
                                                        pre_selection['t_idx']])
    rs = pre_selection['rs']
                               
    x_in = Concatenate()([pre_selection['coords'],
                          pre_selection['features'],
                          pre_selection['addfeat']])
                           
    x = x_in
    energy = pre_selection['energy']
    coords = pre_selection['phys_coords']#physical coordinates
    c_coords = pre_selection['coords']#pre-clustered coordinates
    t_idx = pre_selection['t_idx']
    
    ####################################################################################
    ##################### now the actual model goes below ##############################
    ####################################################################################
    
    allfeat = []
    
    n_cluster_space_coordinates = 3
    
    #extend coordinates already here if needed
    c_coords = extent_coords_if_needed(c_coords, x, n_cluster_space_coordinates)
        

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = GooeyBatchNorm(**batchnorm_options)(x)
        ### reduction done
        
        n_dims = 6
        #exchange information, create coordinates
        x = Concatenate()([coords,coords,c_coords,x])
        x, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=64,
                                                 n_dimensions=n_dims,
                                                 n_filters=64,
                                                 n_propagate=64,
                                                 record_metrics=True,
                                                 use_approximate_knn=False #weird issue with that for now
                                                 )([x, rs])
        
        
        gncoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir,
                                   name='gn_coords_'+str(i))([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs])                                       
        #just keep them in a reasonable range  
        #safeguard against diappearing gradients on coordinates                                       
        gndist = AverageDistanceRegularizer(strength=0.01,
                                            record_metrics=True
                                            )(gndist)
                            
        x = DistanceWeightedMessagePassing([32,32])([x,gnnidx,gndist])                
        x_matt = Dense(16,activation='relu')(x)
        x_matt = MultiAttentionGravNetAdd(n_dims+1)([x,x_matt,gncoords,gnnidx])
        x = Concatenate()([x,x_matt])
        x = DistanceWeightedMessagePassing([32,32])([x,gnnidx,gndist])   
            
        x = GooeyBatchNorm(**batchnorm_options)(x)
        
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        
        x = GooeyBatchNorm(**batchnorm_options)(x)
        
        allfeat.append(x)
        
        
    
    x = Concatenate()([c_coords]+allfeat+[pre_selection['not_noise_score']])
    #do one more exchange with all
    x = Dense(64,activation='elu')(x)
    x = Dense(64,activation='elu')(x)
    x = Dense(64,activation='elu')(x)
    
    
    #######################################################################
    ########### the part below should remain almost unchanged #############
    ########### of course with the exception of the OC loss   #############
    ########### weights                                       #############
    #######################################################################
    
    x = GooeyBatchNorm(**batchnorm_options)(x)
    x = Concatenate()([c_coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist, pred_energy_corr, \
    pred_pos, pred_time, pred_id = create_outputs(x, pre_selection['unproc_features'], 
                                                  n_ccoords=n_cluster_space_coordinates)
    
    # loss
    pred_beta = LLFullObjectCondensation(scale=1.,
                                         energy_loss_weight=2.,
                                         #print_batch_time=True,
                                         position_loss_weight=1e-5,
                                         timing_loss_weight=1e-5,
                                         classification_loss_weight=1e-5,
                                         beta_loss_scale=1.,
                                         too_much_beta_scale=1e-4,
                                         use_energy_weights=True,
                                         record_metrics=True,
                                         name="FullOCLoss",
                                         **loss_options
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy_corr, pred_pos, pred_time, pred_id] +
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
         pre_selection['rs']])
                                         
    #fast feedback
    pred_ccoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir,
                    name='condensation')([pred_ccoords, pred_beta,pre_selection['t_idx'],
                                          rs])                                    

    model_outputs = re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=True
        )
    
    return DictModel(inputs=Inputs, outputs=model_outputs)
    


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(redirect_stdout=True)

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
    
    from model_tools import apply_weights_from_path
    import os
    path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    


verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

plotfrequency=200

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Jan2022/"
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
cb += [
    plotEventDuringTraining(
        outputfile=train.outputDir + "/condensation/c_"+str(i),
        samplefile=samplepath,
        after_n_batches=plotfrequency,
        batchsize=200000,
        on_epoch_end=False,
        publish=None,
        use_event=i)
for i in range(5)
]



from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= 2,
        plot_frequency = plotfrequency,
        select_metrics='FullOCLoss_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/gooey_metrics.html',
        record_frequency= 2,
        plot_frequency = plotfrequency,
        select_metrics='gooey_*',
        publish=publishpath
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/latent_space_metrics.html',
        record_frequency= 2,
        plot_frequency = plotfrequency,
        select_metrics='average_distance_*',
        publish=publishpath
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/slicing_knn_metrics.html',
        record_frequency= 2,
        plot_frequency = plotfrequency,
        publish=publishpath,
        select_metrics='*_bins'
    ),
    
    
    
    
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/global_metrics.html',
    #    record_frequency= 1,
    #    plot_frequency = 200,
    #    select_metrics='process*'
    #    ),
    ]

cb += build_callbacks(train)

#cb=[]
learningrate = 5e-5
nbatch = 220000

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=2,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.99
        l.fluidity_decay= 1e-4 #reaches constant 1 after about one epoch
    
#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=5.

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=121,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)


