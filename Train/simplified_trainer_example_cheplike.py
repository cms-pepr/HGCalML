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

from lossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates,LLEdgeClassifier

from debugLayers import PlotCoordinates

'''

make this about coordinate shifts


'''




def gravnet_model(Inputs,
                  td,
                  viscosity=0.8,
                  print_viscosity=False,
                  fluidity_decay=5e-4,  # reaches after about 7k batches
                  max_viscosity=0.95,
                  debug_outdir=None
                  ):
    # Input preprocessing below. Not much to change here

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    
    
    orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([orig_inputs['t_spectator'], 
                                                        orig_inputs['t_idx']])
                                                     
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
    x = Concatenate()([pre_selection['coords'],
                       pre_selection['features'],
                       pre_selection['addfeat']])
    energy = pre_selection['energy']
    coords = pre_selection['coords']
    t_idx = pre_selection['t_idx']
    
                   
    scatterids = [pre_selection['group_backgather'], [
        pre_selection['noise_backscatter_N'],pre_selection['noise_backscatter_idx']
        ]] #add them here directly
        
    allfeat = [MultiBackScatterOrGather()([x, scatterids])]
    allcoords= [pre_selection['orig_dim_coords'],MultiBackScatterOrGather()([coords, scatterids])]
    
    n_cluster_space_coordinates = 3
    
    #extend coordinates already here if needed
    if n_cluster_space_coordinates > 3:
        extendcoords = Dense(3-n_cluster_space_coordinates,
                             use_bias=False,
                             kernel_initializer='zeros'
                             )(x)
        coords = Concatenate()([coords, extendcoords])
    
    total_iterations=5

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        ### reduction done
        
        #exchange information, create coordinates
        x = Concatenate()([coords,x])
        x, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=64,
                                                 n_dimensions=7,
                                                 n_filters=128,
                                                 n_propagate=64,
                                                 )([x, rs])

        x = DistanceWeightedMessagePassing([64,64,32,32,16,16])([x,gnnidx,gndist])
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        #gradually improve coordinates
        
        add_to_coords = Dense(n_cluster_space_coordinates,
                             use_bias=False,kernel_initializer='zeros')(x)
                     
        coords = Add()([coords,add_to_coords])
        coords = LLClusterCoordinates(
            scale=0.1,
            active=True,
            print_loss=True
            )([coords, t_idx, rs])
        
                             
        #compress output
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        allfeat.append(MultiBackScatterOrGather()([x, scatterids]))
        
        
    
    ####### back to non-reduced space
    #extend coordinate list
    allcoords = [MultiBackScatterOrGather()([coords, scatterids])]+allcoords
    
    x = Concatenate()(allfeat+allcoords)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    #do one more exchange with all
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    x = Concatenate()(allcoords+[x])
    
    pred_beta, pred_ccoords, pred_dist, pred_energy_corr, \
    pred_pos, pred_time, pred_id = create_outputs(x, orig_inputs['features'], 
                                                  fix_distance_scale=False,
                                                  scale_energy=False,
                                                  energy_factor=True,
                                                  n_ccoords=n_cluster_space_coordinates)
    row_splits = CastRowSplits()(orig_inputs['row_splits'])
    # loss
    pred_beta = LLFullObjectCondensation(print_loss=True, scale=1.,
                                         energy_loss_weight=5.,
                                         position_loss_weight=1e-1,
                                         timing_loss_weight=1e-2,
                                         beta_loss_scale=1.,
                                         too_much_beta_scale=.001,
                                         use_energy_weights=True,
                                         q_min=2.5,
                                         #div_repulsion=True,
                                         # cont_beta_loss=True,
                                         # beta_gradient_damping=0.999,
                                         # phase_transition=1,
                                         huber_energy_scale=0.1,
                                         use_average_cc_pos=0.5,  # smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy_corr, pred_pos, pred_time, pred_id] +
        [orig_inputs['rechit_energy']]+
        # truth information
        [orig_inputs['t_idx'] ,
         orig_inputs['t_energy'] ,
         orig_inputs['t_pos'] ,
         orig_inputs['t_time'] ,
         orig_inputs['t_pid'] ,
         orig_t_spectator_weight ,
         row_splits])

    model_outputs = [('pred_beta', pred_beta), 
                     ('pred_ccoords', pred_ccoords),
                     ('pred_energy_corr_factor', pred_energy_corr),
                     ('pred_pos', pred_pos),
                     ('pred_time', pred_time),
                     ('pred_id', pred_id),
                     ('pred_dist', pred_dist),
                     ('row_splits', row_splits)]

    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    train.setCustomOptimizer(tf.keras.optimizers.Adam(
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


#cb += build_callbacks(train)

#by hand
from plotting_callbacks import plotClusterSummary
cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=800
        )
    ]

#cb=[]
learningrate = 1e-4
nbatch = 120000

train.compileModel(learningrate=learningrate, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=3,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.99
        l.fluidity_decay= 1e-4 #reaches constant 1 after about one epoch
    if 'FullOCLoss' in l.name:
        l.use_average_cc_pos = 0.1
        l.q_min = 2.
        l.cont_beta_loss=False
        l.energy_loss_weight=1e-2 #etc
        l.position_loss_weight=1e-2
    if 'edge_selector' in l.name:
        l.use_truth=False#IMPORTANT

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=10.
nbatch = 120000

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)

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

