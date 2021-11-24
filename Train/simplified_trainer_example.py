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
from GravNetLayersRagged import RecalcDistances

from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CreateGlobalIndices, SelectFromIndices, MultiBackScatter, KNN, MessagePassing, RobustModel
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer

from model_blocks import first_coordinate_adjustment

from lossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from debugLayers import PlotCoordinates

'''

make this about coordinate shifts


'''


td = TrainData_NanoML()

def reduce(x,coords,energy,goodneighbours, nidx, rs, t_idx, t_spectator_weight, 
           threshold = 0.5,
           print_reduction=True,
           return_backscatter=False):
    
    gnidx, gsel, bg, srs = NeighbourGroups(
        threshold = 0.5,
        return_backscatter=return_backscatter,
        print_reduction=print_reduction,
        )([goodneighbours, nidx, rs])
    
    
    
   
    #these are needed in reduced form
    t_idx, t_spectator_weight = SelectFromIndices()([gsel,t_idx, t_spectator_weight])
    
    coords = AccumulateNeighbours('mean')([coords, gnidx])
    coords = SelectFromIndices()([gsel,coords])
    energy = AccumulateNeighbours('sum')([energy, gnidx])
    energy = SelectFromIndices()([gsel,energy])
    x = AccumulateNeighbours('minmeanmax')([x, gnidx])
    x = SelectFromIndices()([gsel,x])


    rs = srs #set new row splits
    
    return x,coords,energy,nidx, rs, bg, t_idx, t_spectator_weight

def gravnet_model(Inputs,
                  viscosity=0.8,
                  print_viscosity=False,
                  fluidity_decay=5e-4,  # reaches after about 7k batches
                  max_viscosity=0.95,
                  debug_outdir=None
                  ):
    # Input preprocessing below. Not much to change here

    feat, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(
        Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)

    t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([t_spectator, t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0, 1, name="energy_selector")(feat)
    time = SelectFeatures(8, 9)(feat_norm)
    orig_coords = SelectFeatures(5, 8)(feat_norm)
    
    x = feat_norm
    sel_gidx = gidx_orig

    allfeat = [x]
    allcoords = [orig_coords]
    backgatheredids = []
    scatterids = []
    backgathered = []
    backgathered_coords = []
    energysums = []
    
    ###### pre-selection
    
    coords,nidx,dist, x = first_coordinate_adjustment(
        orig_coords, x, energy, rs, t_idx, 
        debug_outdir,
        trainable=False,
        name='first_coords',
        debugplots_after=-1
        )
    
    allcoords.append(coords)
    
    #dist = LLLocalClusterCoordinates(
    #        print_loss=True,
    #        scale=1.
    #        )([dist, nidx, t_idx, t_spectator_weight])
            
    #just the segmentation part without beta terms of OC
    #coords = LLClusterCoordinates(
    #    print_loss=False,
    #        scale=1.
    #    )([coords,t_idx,rs])
    coords = PlotCoordinates(500,outdir=debug_outdir,name='a_firstcoord')([coords,energy,t_idx,rs])
        
    dist = RecalcDistances()([coords,nidx])
    dist, nidx = SortAndSelectNeighbours(K=12)([dist, nidx])
    
    
    #not used here
    x = Concatenate()([x,feat_norm,dist])
    goodneighbours = Dense(1, activation='sigmoid')(x)
    goodneighbours = LLNeighbourhoodClassifier(
        print_loss=True,
        scale=1.,
        print_batch_time=False
        )([goodneighbours,nidx,t_idx])
    
    ### pass info
    
    allfeat.append(x)
    allfeat.append(coords)
    
    x,coords,energy,nidx, rs, bg, t_idx, t_spectator_weight = reduce(x, coords, 
                                          energy, goodneighbours, nidx, rs,
                                          t_idx,t_spectator_weight,
                                          threshold = 0.6,#higher 
                                          return_backscatter=True) #really cut
    scatterids.append(bg)
    
    #filter the noise, using information present
    isnotnoise = Dense(1, activation='sigmoid')(Concatenate()([x,coords]))
    isnotnoise = LLNotNoiseClassifier(
        print_loss=True,
        scale=1.
        )([isnotnoise, t_idx])
        
    nonoisesel,rs,bg = NoiseFilter(
        print_reduction=True
        )([isnotnoise,rs])
    scatterids.append(bg)
    nidx = None #is void after this
    x,coords,energy = SelectFromIndices()([nonoisesel,x,coords,energy])
    t_idx, t_spectator_weight = SelectFromIndices()([nonoisesel,t_idx, t_spectator_weight])
    
    coords = PlotCoordinates(500,outdir=debug_outdir,name='b_afternoise')([coords,energy,t_idx,rs])
    
    #allfeat.append(MultiBackScatterOrGather()([x, scatterids]))
    #energysums.append(MultiBackScatterOrGather()([x, energy]))
    ###### created pre-selection data, now do the selection
    
    # here the actual network starts
    
    total_iterations=4

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        ### reduction done
        
        #exchange information
        x = Concatenate()([coords,x])
        x, gncoords, nidx, gndist = RaggedGravNet(n_neighbours=32,
                                                 n_dimensions=6,
                                                 n_filters=128,
                                                 n_propagate=64,
                                                 )([x, rs])

        x = MessagePassing([64,32,32,16])([x,nidx])
        x = Dense(64,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        
        coords = Add()([coords, Dense(3,kernel_initializer='zeros')(x)])
        
        coords = LLClusterCoordinates(
            print_loss=True,
                scale=1.
            )([coords,t_idx,rs])
            
        dist = RecalcDistances()([coords,nidx])
        dist, nidx = SortAndSelectNeighbours(K=12)([dist, nidx])
        
        goodneighbours = Concatenate()([x,dist])
        goodneighbours = Dense(1, activation='sigmoid')(goodneighbours)
        goodneighbours = LLNeighbourhoodClassifier(
            print_loss=True,
            scale=1.,
            print_batch_time=False
            )([goodneighbours,nidx,t_idx])
            
        coords = PlotCoordinates(500,outdir=debug_outdir,name='c_it'+str(i))([coords,energy,t_idx,rs])
            
        x,coords,energy,nidx, rs, bg, t_idx, t_spectator_weight = reduce(x, coords, energy, goodneighbours, 
                                              nidx, rs, t_idx, t_spectator_weight,
                                              print_reduction=i==total_iterations-1,
                                              return_backscatter=False)#here back-gather
        
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        scatterids.append(bg)
        
        allfeat.append(MultiBackScatterOrGather()([x, scatterids]))
        energysums.append(MultiBackScatterOrGather()([energy, scatterids]))
        allcoords.append(MultiBackScatterOrGather()([coords, scatterids]))
        
        

    rs = row_splits  # important! here we are in non-reduced full graph mode again
    
    #x = MultiBackScatterOrGather()([x, scatterids])
    x = Concatenate()(allfeat+energysums+allcoords)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    #do one more exchange with all
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(64,activation='relu')(x)
    x = Concatenate()(allcoords+[x]+energysums)
    
    pred_beta, pred_ccoords, pred_dist, pred_energy, \
    pred_pos, pred_time, pred_id = create_outputs(x, feat, fix_distance_scale=False,
                                                  n_ccoords=3)

    # loss
    pred_beta = LLFullObjectCondensation(print_loss=True, scale=5.,
                                         energy_loss_weight=1e-3,
                                         position_loss_weight=1e-3,
                                         timing_loss_weight=1e-3,
                                         beta_loss_scale=1.,
                                         too_much_beta_scale=.01,
                                         use_energy_weights=True,
                                         q_min=2.5,
                                         div_repulsion=True,
                                         # cont_beta_loss=True,
                                         # beta_gradient_damping=0.999,
                                         # phase_transition=1,
                                         huber_energy_scale=3,
                                         use_average_cc_pos=0.5,  # smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy, pred_pos, pred_time, pred_id] +
        # truth information
        [orig_t_idx, orig_t_energy,
         orig_t_pos, orig_t_time, orig_t_pid,
         orig_t_spectator_weight,
         row_splits])

    model_outputs = [('pred_beta', pred_beta), 
                     ('pred_ccoords', pred_ccoords),
                     ('pred_energy', pred_energy),
                     ('pred_pos', pred_pos),
                     ('pred_time', pred_time),
                     ('pred_id', pred_id),
                     ('pred_dist', pred_dist),
                     ('row_splits', row_splits)]

    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(gravnet_model,debug_outdir=train.outputDir+'/intplots')
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

#cb=[]
learningrate = 1e-3
nbatch = 90000

train.compileModel(learningrate=1e-3, #gets overwritten by CyclicLR callback anyway
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
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/5.,
                                  max_lr = learningrate,
                                  step_size = 250)]+cb)

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
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/10.,
                                  max_lr = learningrate,
                                  step_size = 100)]+cb)
#

