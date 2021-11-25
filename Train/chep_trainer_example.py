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
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import ProcessFeatures, SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from initializers import EyeInitializer
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from datastructures import TrainData_NanoML

from plotting_callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining, plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs
from GravNetLayersRagged import EdgeCreator, EdgeSelector, GroupScoreFromEdgeScores,NoiseFilter,ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackGather, KNN, MessagePassing, RobustModel
from Layers import GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer

'''

'''


td = TrainData_NanoML()


def gravnet_model(Inputs,
                  viscosity=0.1,
                  print_viscosity=False,
                  fluidity_decay=1e-3,  # reaches after about 7k batches
                  max_viscosity=0.95
                  ):
    # Input preprocessing below. Not much to change here

    featx, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(
        Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(featx)

    t_spectator_weight = CreateTruthSpectatorWeights(threshold=1.21,
                                                     minimum=1e-2,
                                                     active=True
                                                     )([t_spectator, t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([featx, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(featx)
    energy = SelectFeatures(0, 1, name="energy_selector")(featx)
    time = SelectFeatures(8, 9)(feat_norm)
    orig_coords = SelectFeatures(5, 8)(feat_norm)

    x = feat_norm
    sel_gidx = gidx_orig

    allfeat = [x]
    backgatheredids = []
    gatherids = []
    backgathered = []
    backgathered_coords = []
    energysums = []

    x_basic = BatchNormalization(momentum=0.6)(x)  # mask_and_norm is just batch norm now
    x = x_basic
    x = RaggedGlobalExchange(name="global_exchange")([x, row_splits])
    x = Dense(64, activation='elu', name="dense_start")(x)

    n_filters = 0
    n_gravnet_layers = 4
    feat = []
    for i in range(n_gravnet_layers):
        n_filters = 196
        n_propagate = [128, 64, 32, 16, 8, 4, 2]
        n_neighbours = 64
        n_dim = 8
        # if n_dim < 2:
        #    n_dim = 2

        outfeats, coordinates, neighbour_indices, distancesq = RaggedGravNet(n_neighbours=n_neighbours,
                                  n_dimensions=n_dim,
                                  n_filters=n_filters,
                                  n_propagate=n_propagate[0],
                                  name='gravnet_' + str(i))([x, row_splits])

        x = DistanceWeightedMessagePassing(n_propagate[1:])((outfeats, neighbour_indices, distancesq))

        x = BatchNormalization(momentum=0.6)(x)
        x = Dense(128, activation='elu', name="dense_bottom_" + str(i) + "_a")(x)
        x = BatchNormalization(momentum=0.6, name="bn_a_" + str(i))(x)
        x = Dense(96, activation='elu', name="dense_bottom_" + str(i) + "_b")(x)
        x = RaggedGlobalExchange(name="global_exchange_bot_" + str(i))([x, row_splits])
        x = Dense(96, activation='elu', name="dense_bottom_" + str(i) + "_c")(x)
        x = BatchNormalization(momentum=0.6, name="bn_b_" + str(i))(x)

        feat.append(x)

    x = Concatenate(name="concat_gravout")(feat)
    x = Dense(128, activation='elu', name="dense_last_a")(x)
    x = BatchNormalization(momentum=0.6, name="bn_last_a")(x)
    x = Dense(128, activation='elu', name="dense_last_a1")(x)
    x = BatchNormalization(momentum=0.6, name="bn_last_a1")(x)
    x = Dense(128, activation='elu', name="dense_last_a2")(x)
    x = BatchNormalization(momentum=0.6, name="bn_last_a2")(x)
    x = Dense(64, activation='elu', name="dense_last_b")(x)
    x = Dense(64, activation='elu', name="dense_last_c")(x)

    pred_beta, pred_ccoords, pred_dist, pred_energy, \
    pred_pos, pred_time, pred_id = create_outputs(x, featx, fix_distance_scale=False,
                                                  n_ccoords=3
                                                  )

    # loss
    pred_beta = LLFullObjectCondensation(print_loss=True,
                                         energy_loss_weight=1e-7,
                                         position_loss_weight=1e-7,
                                         timing_loss_weight=1e-7,
                                         beta_loss_scale=1.,
                                         too_much_beta_scale=.1,
                                         use_energy_weights=False,
                                         q_min=2.5,
                                         div_repulsion=True,
                                         # phase_transition=1,
                                         huber_energy_scale=3,
                                         use_average_cc_pos=0.2,  # smoothen it out a bit
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

    for i, (x, y) in enumerate(zip(backgatheredids, backgathered_coords)):
        model_outputs.append(('backgatheredids_' + str(i), x))
        model_outputs.append(('backgathered_coords_' + str(i), y))
    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


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
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

#
# cb = []
#
#
# cb += [plotClusteringDuringTraining(
#     use_backgather_idx=8 + i,
#     outputfile=train.outputDir + "/plts/sn" + str(i) + '_',
#     samplefile=samplepath,
#     after_n_batches=200,
#     on_epoch_end=False,
#     publish=None,
#     use_event=0)
#     for i in [0, 2]]
#
# cb += [
#     plotEventDuringTraining(
#         outputfile=train.outputDir + "/plts2/sn0"+str(i),
#         samplefile=samplepath,
#         after_n_batches=200,
#         batchsize=200000,
#         on_epoch_end=False,
#         publish=None,
#         use_event=i)
# for i in range(5)
# ]
#
# cb += [
#     plotGravNetCoordsDuringTraining(
#         outputfile=train.outputDir + "/coords_" + str(i) + "/coord_" + str(i),
#         samplefile=samplepath,
#         after_n_batches=200,
#         batchsize=200000,
#         on_epoch_end=False,
#         publish=None,
#         use_event=0,
#         use_prediction_idx=i,
#     )
#     for i in range(12, 18)  # between 16 and 21
# ]


# cb += build_callbacks(train)

learningrate = 3e-4
nbatch = 120000

train.compileModel(learningrate=1e-3, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=38,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/3,
                                  max_lr = learningrate,
                                  step_size = 50)])

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.99
        l.fluidity_decay= 5e-4 #reaches constant 1 after about one epoch
    if 'FullOCLoss' in l.name:
        l.use_average_cc_pos = 0.1
        #l.q_min = 0.1

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=3.
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
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/10.,
                                  max_lr = learningrate,
                                  step_size = 100)])
#

