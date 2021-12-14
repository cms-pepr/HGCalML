'''

Compatible with the dataset here:
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoML
and (soon)
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoMLTracks

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/OctProd/NanoML

not compatible with datasets before end of October 2021

'''
from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import Reshape,BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import WeightFeatures,WeightedNeighbourMeans,DownSample, CreateIndexFromMajority, ProcessFeatures, SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from initializers import EyeInitializer
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from datastructures import TrainData_NanoML

from plotting_callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining, plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR

from model_blocks import create_outputs
from GravNetLayersRagged import MultiBackScatter,EdgeCreator, EdgeSelector
from GravNetLayersRagged import GroupScoreFromEdgeScores,NoiseFilter
from GravNetLayersRagged import ProcessFeatures,SoftPixelCNN, RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing,MultiBackScatterOrGather

from GravNetLayersRagged import NeighbourGroups,AccumulateNeighbours,SelectFromIndices
from GravNetLayersRagged import RecalcDistances, ElementScaling, RemoveSelfRef, CastRowSplits

from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CreateGlobalIndices, SelectFromIndices, MultiBackScatter, KNN, MessagePassing, RobustModel
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer

from model_blocks import first_coordinate_adjustment, reduce, pre_selection_model_full
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits

from lossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates,LLEdgeClassifier
from lossLayers import LLFillSpace

from debugLayers import PlotCoordinates

'''

make this about coordinate shifts


'''




def gravnet_model(Inputs,
                  td,
                  viscosity=0.2,
                  print_viscosity=False,
                  fluidity_decay=1e-3,  # reaches after about 7k batches
                  max_viscosity=0.99,
                  debug_outdir=None,
                  plot_debug_every=1000,
                  ):
    # Input preprocessing below. Not much to change here

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    
    
    orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([orig_inputs['t_spectator'], 
                                                        orig_inputs['t_idx']])
    orig_inputs['t_spectator_weight'] = orig_t_spectator_weight                                                 
    #can be loaded - or use pre-selected dataset (to be made)
    pre_selection = pre_selection_model_full(orig_inputs,
                             debug_outdir,
                             reduction_threshold=0.5,
                             use_edges=True,
                             trainable=False, #use this as a static model
                             debugplots_after=-1, #no debug plots
                             omit_reduction=False
                             )
    
    '''
    pre_selection has the following dict items:
    
    ['features'] = selfeat
    ['coords'] = coords
    ['addfeat'] = x (pre-processed features)
    ['energy'] = energy (energy sums in case there was a grouping)
    
    ['group_backgather']=group_backgather
    ['noise_backscatter_N']=noise_backscatter[0]
    ['noise_backscatter_idx']=noise_backscatter[1]
    
    ['rs']=rs
    
    Selected truth information:
    
    ['t_idx'], 
    ['t_energy'], 
    ['t_pos'], 
    ['t_time'], 
    ['t_pid'], 
    ['t_spectator'], 
    ['t_fully_contained'] 
    
    full N_hits dimension!! 
    out['orig_t_idx'] = orig_inputs['t_idx']
    out['orig_t_energy'] = orig_inputs['t_energy'] #for validation
    '''                                                 
    
    ########## from here on everything is based on the pre-selection; only extend at the very end for the loss
    
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
    coords = pre_selection['coords']
    t_idx = pre_selection['t_idx']
    
                   
    #scatterids = [pre_selection['group_backgather'], [
    #    pre_selection['noise_backscatter_N'],pre_selection['noise_backscatter_idx']
    #    ]] #add them here directly
        
    allfeat = []
    
    n_cluster_space_coordinates = 3
    
    #extend coordinates already here if needed
    coords = extent_coords_if_needed(coords, x, n_cluster_space_coordinates)
    
    pre_coords = coords
    total_iterations=2

    for i in range(total_iterations):
            
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, 
                           max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay)(x)
        ### reduction done
        
        n_dimensions = 6
        #in standard configuration with i<2
        n_neighbours = 64
        
        #exchange information, create coordinates
        x = Concatenate()([coords,coords,x])
        x, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=n_neighbours,
                                                 n_dimensions=n_dimensions,
                                                 n_filters=128,
                                                 n_propagate=64,
                                                 coord_initialiser_noise=1e-3,
                                                 use_approximate_knn=False #faster on reduced data for now
                                                 )([x, rs])
                                                 
        #just keep them in a reasonable range  
        #safeguard against diappearing gradients on coordinates                                       
        gndist = AverageDistanceRegularizer(strength=0.01,
                                            printout=True
                                            )(gndist)
        
        #enforce dimensions to be used and avoid (hyper) planes                                   
        gncoords = LLFillSpace(print_loss = True,
                               active = True,
                               scale=0.005,#just mild
                               #runevery=1, 
                               )([gncoords,rs]) 
        
        gncoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir)([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs])
        x = Concatenate()([StopGradient()(ScalarMultiply(1e-6)(gncoords)),x])

        if i > 1:
            for _ in range(4): #consider different ranges
                dscaling = Dense(1)(x)
                ld = LocalDistanceScaling()([gndist,dscaling])
                x = DistanceWeightedMessagePassing([32,16])([x,gnnidx,ld])
            
        else: #in standard configuration with i<2
            x = DistanceWeightedMessagePassing([64,64,32,32,16,16,8,8])([x,gnnidx,gndist])
            
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        #                     
        #compress output
        x = Dense(96,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        #allfeat.append(MultiBackScatterOrGather()([x, scatterids]))
        allfeat.append(x)
        
    
    ## add coordinate differences
    #coords = Add()([coords, ScalarMultiply(-1.)(pre_coords)])
    
    
    ####### back to non-reduced space
    
    #extend coordinate list
    #coords = MultiBackScatterOrGather()([coords, scatterids])
    #pre_coords = extent_coords_if_needed(pre_selection['orig_dim_coords'], x, n_cluster_space_coordinates)
    #coords = Add()([coords,pre_coords])
    
    x = Concatenate()(allfeat+[pre_selection['not_noise_score']])
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    #do one more exchange with all
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    x = Concatenate()([coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist, pred_energy_corr, \
    pred_pos, pred_time, pred_id = create_outputs(x, pre_selection['features'], 
                                                  fix_distance_scale=False,
                                                  scale_energy=False,
                                                  energy_factor=True,
                                                  wide_distance_scale=True,
                                                  n_ccoords=n_cluster_space_coordinates)
    
    # loss
    pred_beta = LLFullObjectCondensation(print_loss=True, scale=.8,
                                         energy_loss_weight=2.2,
                                         position_loss_weight=1e-1,
                                         timing_loss_weight=1e-2,
                                         classification_loss_weight=1e-4,
                                         beta_loss_scale=3.,
                                         too_much_beta_scale=.0001,
                                         use_energy_weights=True,
                                         alt_energy_weight=True,
                                         q_min=2.5,
                                         #div_repulsion=True,
                                         # cont_beta_loss=True,
                                         # beta_gradient_damping=0.999,
                                         # phase_transition=1,
                                         #huber_energy_scale=0.1,
                                         use_average_cc_pos=0.75,  # smoothen it out a bit
                                         name="FullOCLoss"
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
         pre_selection['t_spectator_weight'] ,
         pre_selection['rs']])
                                         
    #fast feedback
    pred_ccoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir,
                    name='condensation')([pred_ccoords, pred_beta,pre_selection['t_idx'],
                                          rs])                                    

    #model_outputs = [('pred_beta', pred_beta), 
    #                 ('pred_ccoords', pred_ccoords),
    #                 ('pred_energy_corr_factor', pred_energy_corr),
    #                 ('pred_pos', pred_pos),
    #                 ('pred_time', pred_time),
    #                 ('pred_id', pred_id),
    #                 ('pred_dist', pred_dist),
    #                 ('row_splits', row_splits)]
    #
    
    model_outputs = re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist
        )
    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(
        #larger->slower forgetting
        #beta_1: linear
        #beta_2: sq
        #make it slower for our weird fluctuating batches
        #beta_1=0.99, #0.9
        #beta_2=0.99999 #0.999
        #clipnorm=0.001
        #amsgrad=True,
        #epsilon=1e-2
        ))

    #get pretrained preselection weights
    
    from model_tools import apply_weights_from_path
    import os
    path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection/KERAS_check_model_last.h5'
    train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    
    #
    train.compileModel(learningrate=1e-4,
                       loss=None)


verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


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
        after_n_batches=500,
        batchsize=200000,
        on_epoch_end=False,
        publish=None,
        use_event=i)
for i in range(5)
]

#cb += [
#    plotGravNetCoordsDuringTraining(
#        outputfile=train.outputDir + "/localcoords/coord_" + str(i),
#        samplefile=samplepath,
#        after_n_batches=500,
#        batchsize=200000,
#        on_epoch_end=False,
#        publish=None,
#        use_event=0,
#        use_prediction_idx=8+i,
#    )
#    for i in [1,3,5]  # between 16 and 21
#]
#


cb += build_callbacks(train)

#cb=[]
learningrate = 4e-5
nbatch = 360000

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)

print("ALMOST freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.99
        #l.fluidity_decay= 1e-4 #reaches constant 1 after about one epoch
    
#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=5.

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=121,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)
#

