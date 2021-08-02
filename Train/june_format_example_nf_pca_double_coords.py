'''
This is one of the really good models and configurations.
Keep this in mind
'''
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model
from tensorboard_manager import TensorBoardManager
from running_plots import RunningEfficiencyFakeRateCallback
import tensorflow.keras as keras
from datastructures import TrainData_NanoML

from DeepJetCore.modeltools import fixLayersContaining
# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from plotting_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs

from Layers import LocalClusterReshapeFromNeighbours2,ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA,LocalClusterReshapeFromNeighbours,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackGather, KNN, MessagePassing, RobustModel
from Layers import GooeyBatchNorm #make a new line
from datastructures import TrainData_OC

td=TrainData_NanoML()
'''

'''


def gravnet_model(Inputs,
                  viscosity=0.2,
                  print_viscosity=False,
                  fluidity_decay=1e-3,
                  max_viscosity=0.9 # to start with
                  ):

    feature_dropout=-1.
    addBackGatherInfo=True,

    feat, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0,1)(feat)
    time = SelectFeatures(8,9)(feat_norm)
    orig_coords = SelectFeatures(5,8)(feat)

    x = feat_norm

    allfeat=[x]
    backgatheredids=[]
    gatherids=[]
    backgathered = []
    backgathered_coords = []
    energysums = []

    n_reshape_dimensions=3

    #really simple real coordinates
    coords = ManualCoordTransform()(orig_coords)
    coords = Concatenate()([coords, time])
    coords = Dense(n_reshape_dimensions, use_bias=False, kernel_initializer=tf.keras.initializers.Identity()  )(coords)#just rotation and scaling


    #see whats there
    nidx, dist = KNN(K=24,radius=1.0)([coords,rs])

    x_mp = DistanceWeightedMessagePassing([32,16,16,8])([x,nidx,dist])
    x = Concatenate()([x,x_mp])
    #this is going to be among the most expensive operations:
    x = Dense(64, activation='elu',name='pre_dense_a')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)

    allfeat.append(x)
    backgathered_coords.append(coords)

    sel_gidx = gidx_orig

    cdist = dist
    ccoords = coords

    total_iterations=5

    fwdbgccoords=None

    for i in range(total_iterations):

        cluster_neighbours = 5
        #derive new coordinates for clustering
        if i:
            ccoords = Add()([
                ccoords,Dense(n_reshape_dimensions,name='newcoords'+str(i),
                                         kernel_initializer='zeros'
                                         )(x)
                ])
            nidx, cdist = KNN(K=5*cluster_neighbours,radius=-1.0)([ccoords,rs])
            #here we use more neighbours to improve learning of the cluster space
            #this can be adjusted in the final trained model to be equal to 'cluster_neighbours'
        cdist = LocalDistanceScaling(max_scale=10.)([cdist, Dense(1,kernel_initializer='zeros')(Concatenate()([x,cdist]))])

        x, rs, bidxs, sel_gidx, energy, x, t_idx, coords, ccoords, cdist = LocalClusterReshapeFromNeighbours2(
                 K=cluster_neighbours,
                 radius=0.1,
                 print_reduction=True,
                 loss_enabled=True,
                 loss_scale = 3.,
                 loss_repulsion=0.8, # it's important to get this right here
                 hier_transforms=[64,32,16],
                 print_loss=False,
                 name='clustering_'+str(i)
                 )([x, cdist, nidx, rs, sel_gidx, energy, x, t_idx, coords, ccoords, cdist, t_idx])

        gatherids.append(bidxs)

        #explicit
        energy = ReduceSumEntirely()(energy)#sums up all contained energy per cluster

        x = Dense(128, activation='elu',name='dense_clc_a'+str(i))(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
        x = Dense(128, activation='elu',name='dense_clc_b'+str(i))(x)
        x = Dense(64, activation='elu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)


        n_dimensions = 3+i #make it plottable
        nneigh = 64+32*i #this will be almost fully connected for last clustering step
        nfilt = 64
        nprop = 64

        x = Concatenate()([coords,x])

        x_gn, coords, nidx, dist = RaggedGravNet(n_neighbours=nneigh,
                                              n_dimensions=n_dimensions,
                                              n_filters=nfilt,
                                              n_propagate=nprop)([x, rs])


        x_pca = Dense(2+4*i)(x)
        x_pca = NeighbourApproxPCA()([coords,dist,x_pca,nidx])
        x_pca = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x_pca)

        x_mp = DistanceWeightedMessagePassing([64,64,32,32])([x,nidx,dist])
        x_mp = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x_mp)

        x = Concatenate()([x,x_pca,x_mp,x_gn])
        #check and compress it all
        x = Dense(128, activation='elu',name='dense_a_'+str(i))(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
        x = Dense(32+16*i, activation='elu',name='dense_c_'+str(i))(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)


        energysums.append( MultiBackGather()([energy, gatherids]) )#assign energy sum to all cluster components

        allfeat.append(MultiBackGather()([x, gatherids]))

        backgatheredids.append(MultiBackGather()([sel_gidx, gatherids]))
        bgccoords = MultiBackGather()([ccoords, gatherids])
        if fwdbgccoords is None:
            fwdbgccoords = bgccoords
        backgathered_coords.append(bgccoords)




    x = Concatenate(name='allconcat')(allfeat)
    x = Dense(128, activation='elu', name='alldense')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)

    #this is going to be resource intense, give a good starting point with the last ccoords
    coords = Dense(3,use_bias=False,
                   kernel_initializer=tf.keras.initializers.Identity()
                   )(Concatenate()([ fwdbgccoords ,x]))

    nidx, dist = KNN(K=32,radius=1.0)([coords,row_splits])
    x_mp = DistanceWeightedMessagePassing(8*[8])([x,nidx,dist])#only some but a lot of hops
    x = Concatenate()([x,x_mp])
    #
    backgathered_coords.append(coords)

    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    x = Dense(128, activation='elu')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    x = Concatenate()([x]+energysums)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)

    pred_beta, pred_ccoords, pred_dist, pred_energy, pred_pos, pred_time, pred_id = create_outputs(x,feat,add_distance_scale=True)

    #loss
    pred_beta = LLFullObjectCondensation(print_loss=True,
                                         energy_loss_weight=1e-4,
                                         position_loss_weight=1e-3,
                                         timing_loss_weight=1e-3,
                                         beta_loss_scale=1.,
                                         repulsion_scaling=5.,
                                         repulsion_q_min=1.,
                                         super_repulsion=False,
                                         q_min=0.5,
                                         use_average_cc_pos=0.5,
                                         prob_repulsion=True,
                                         phase_transition=1,
                                         huber_energy_scale = 3,
                                         alt_potential_norm=True,
                                         beta_gradient_damping=0.,
                                         payload_beta_gradient_damping_strength=0.,
                                         kalpha_damping_strength=0.,#1.,
                                         use_local_distances=True,
                                         name="FullOCLoss"
                                         )([pred_beta, pred_ccoords,
                                            pred_dist,
                                            pred_energy,
                                            pred_pos, pred_time, pred_id,
                                            orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid,
                                            row_splits])

    return RobustModel(model_inputs=Inputs, model_outputs=[('pred_beta', pred_beta),
                                                           ('pred_ccoords',pred_ccoords),
                                                           ('pred_energy',pred_energy),
                                                           ('pred_pos',pred_pos),
                                                           ('pred_time',pred_time),
                                                           ('pred_id',pred_id),
                                                           ('pred_dist',pred_dist),
                                                           ('row_splits',rs)])


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(gravnet_model)
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(
        clipnorm=0.001
        ))

    train.compileModel(learningrate=1e-4,
                       loss=None)


verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


cb = []
os.system('mkdir -p %s' % (train.outputDir + "/summary/"))
tensorboard_manager = TensorBoardManager(train.outputDir + "/summary/")

cb += [RunningEfficiencyFakeRateCallback(td, tensorboard_manager, dist_threshold=0.5, beta_threshold=0.5, database_manager=None)]


learningrate = 1e-3
nbatch = 140000 #quick first training with simple examples = low # hits

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None,
                          )


model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=38,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/3,
                                  max_lr = learningrate,
                                  step_size = 50)]+cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 1.
        l.fluidity_decay= 5e-4 #reaches constant 1 after about one epoch
    if 'FullOCLoss' in l.name:
        l.use_average_cc_pos = 0.1
        l.q_min = 0.1

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=3.
nbatch = 150000

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)

model, history = train.trainModel(nepochs=121,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/10.,
                                  max_lr = learningrate,
                                  step_size = 100)]+cb)
#
