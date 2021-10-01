'''
This is one of the really good models and configurations.
Keep this in mind
'''
import matching_and_analysis
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import ElementScaling,EdgeCreator, EdgeSelector, GroupScoreFromEdgeScores,NoiseFilter,LNC,ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model

from experiment_database_reading_manager import ExperimentDatabaseReadingManager
from hgcal_predictor import HGCalPredictor
from hyperparam_optimizer import OCHyperParamOptimizer
from running_full_validation import RunningFullValidation
from tensorboard_manager import TensorBoardManager
from running_plots import RunningMetricsDatabaseAdditionCallback, RunningMetricsPlotterCallback
import tensorflow.keras as keras
from datastructures import TrainData_NanoML
import uuid

from DeepJetCore.modeltools import fixLayersContaining

# from tensorflow.keras.optimizer_v2 import Adam

from plotting_callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs, noise_pre_filter

from Layers import CreateTruthSpectatorWeights,ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackGather, KNN, MessagePassing, RobustModel
from Layers import GooeyBatchNorm #make a new line
import sql_credentials
from datetime import datetime

from Regularizers import OffDiagonalRegularizer

td=TrainData_NanoML()
'''

'''


def gravnet_model(Inputs,
                  viscosity=0.1,
                  print_viscosity=False,
                  fluidity_decay=1e-3, #reaches after about 7k batches
                  max_viscosity=0.95 
                  ):

    
    #Input preprocessing below. Not much to change here

    feat,  t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)
    
    t_spectator_weight = CreateTruthSpectatorWeights(threshold = 1.21, 
                                                minimum=1e-2,
                                                active = True
                                                )([t_spectator,t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0,1,name="energy_selector")(feat)
    time = SelectFeatures(8,9)(feat_norm)
    orig_coords = SelectFeatures(5,8)(feat_norm)

    x = feat_norm
    sel_gidx = gidx_orig

    allfeat=[x]
    backgatheredids=[]
    gatherids=[]
    backgathered = []
    backgathered_coords = []
    energysums = []

    #here the actual network starts


    ############## Keep this part to reload the noise filter with pre-trained weights for other trainings
    
    coords = orig_coords
    
    other = [x, coords, energy, sel_gidx, t_spectator_weight, t_idx]
    
    coords, nidx, dist, noise_score, rs, bg, other = noise_pre_filter(
        x, coords, rs, 
        other,  t_idx, threshold=0.025)
    
    x_nn, coords, energy, sel_gidx, t_spectator_weight, t_idx = other
    
    
    ###>>> Noise filter part done
    
    #this is going to be among the most expensive operations:
    x = Dense(64, activation='elu',name='noise_filter_nf_nf_nf_dummy')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)

    
    pred_beta, pred_ccoords, pred_dist, pred_energy,\
       pred_pos, pred_time, pred_id = create_outputs(x,feat,fix_distance_scale=False,
                                                     name_prefix='noise_filter')

    #loss
    pred_beta = LLFullObjectCondensation(print_loss=True,
                                         scale = 1e-5,
                                         energy_loss_weight=1e-2,
                                         position_loss_weight=1e-2,
                                         timing_loss_weight=1e-2,
                                         beta_loss_scale=1.,
                                         q_min=2.0,
                                         div_repulsion=True,
                                         # phase_transition=1,
                                         huber_energy_scale = 3,
                                         use_average_cc_pos=0.2,#smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  #oc output and payload
                                            [pred_beta, pred_ccoords,pred_dist,
                                            pred_energy,pred_pos, pred_time, pred_id]+
                                             #truth information
                                            [orig_t_idx, orig_t_energy, 
                                             orig_t_pos, orig_t_time, orig_t_pid,
                                             orig_t_spectator_weight,
                                             row_splits])

    model_outputs = [('pred_beta', pred_beta), ('pred_ccoords',pred_ccoords),
       ('pred_energy',pred_energy),
       ('pred_pos',pred_pos),
       ('pred_time',pred_time),
       ('pred_id',pred_id),
       ('pred_dist',pred_dist),
       ('row_splits',rs)]

    for i, (x, y) in enumerate(zip(backgatheredids, backgathered_coords)):
        model_outputs.append(('backgatheredids_'+str(i), x))
        model_outputs.append(('backgathered_coords_'+str(i), y))
    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)

import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)
train.val_data.writeToFile(train.outputDir + 'valsamples.djcdc')

if not train.modelSet():
    train.setModel(gravnet_model)
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(
        clipnorm=0.001
        ))

    train.compileModel(learningrate=1e-4,
                       loss=None)

    #print(train.keras_model.summary())

verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


cb = []
os.system('mkdir -p %s' % (train.outputDir + "/summary/"))
tensorboard_manager = TensorBoardManager(train.outputDir + "/summary/")

unique_id_path = os.path.join(train.outputDir,'unique_id.txt')
if os.path.exists(unique_id_path):
        with open(unique_id_path, 'r') as f:
            unique_id = f.readlines()[0].strip()

else:
    unique_id = str(uuid.uuid4())[:8]
    with open(unique_id_path, 'w') as f:
        f.write(unique_id+'\n')

nbatch = 80000


# This will both to server and a local file
database_manager = ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials, file=os.path.join(train.outputDir,"training_metrics.db"), cache_size=100)
database_reading_manager = ExperimentDatabaseReadingManager(file=os.path.join(train.outputDir,"training_metrics.db"))
database_manager.set_experiment(unique_id)

metadata = matching_and_analysis.build_metadeta_dict(beta_threshold=0.5, distance_threshold=0.5, iou_threshold=0.0001, matching_type=matching_and_analysis.MATCHING_TYPE_MAX_FOUND)
analyzer = matching_and_analysis.OCAnlayzerWrapper(metadata)
cb += [RunningMetricsDatabaseAdditionCallback(td, tensorboard_manager, database_manager=database_manager, analyzer=analyzer)]
cb += [RunningMetricsPlotterCallback(after_n_batches=200, database_reading_manager=database_reading_manager,output_html_location=os.path.join(train.outputDir,"training_metrics.html"), publish=None)]
predictor = HGCalPredictor(os.path.join(train.outputDir, 'valsamples.djcdc'), os.path.join(train.outputDir, 'valsamples.djcdc'),
                           os.path.join(train.outputDir, 'temp_val_outputs'), batch_size=nbatch, unbuffered=False,
                           model_path=os.path.join(train.outputDir, 'KERAS_check_model_last_save'),
                           inputdir=os.path.split(train.inputData)[0], max_files=1)

analyzer2 = matching_and_analysis.OCAnlayzerWrapper(metadata) # Use another analyzer here to be safe since it will run scan on
                                                              # on beta and distance threshold which might mess up settings
optimizer = OCHyperParamOptimizer(analyzer=analyzer2, limit_n_endcaps=10)
os.system('mkdir %s/full_validation_plots' % (train.outputDir))
cb += [RunningFullValidation(trial_batch=10, run_optimization_loop_for=100, optimization_loop_num_init_points=5,
                             after_n_batches=5000,min_batch=8, predictor=predictor, optimizer=optimizer,
                             database_manager=database_manager, pdfs_path=os.path.join(train.outputDir,
                                                                                       'full_validation_plots'))]


cb=[]
cb += [plotClusteringDuringTraining(
    use_backgather_idx=8 + 2*i,
    outputfile=train.outputDir + "/neighbour_clusters/p" + str(i) + '_',
    samplefile=samplepath,
    after_n_batches=500,
    on_epoch_end=False,
    publish=None,
    use_event=0)
    for i in range(5)]

cb += [
    plotGravNetCoordsDuringTraining(
        outputfile=train.outputDir + "/coords/coord_" + str(i),
        samplefile=samplepath,
        after_n_batches=500,
        batchsize=30000,
        on_epoch_end=False,
        publish=None,
        use_event=0,
        use_prediction_idx=9 + 2*i,
    )
    for i in range(5)  # between 16 and 21
]

cb += [
    plotEventDuringTraining(
        outputfile=train.outputDir + "/cluster_space/e" + str(i) + '_',
        samplefile=samplepath,
        after_n_batches=500,
        batchsize=30000,
        on_epoch_end=False,
        publish=None,
        use_event=i) for i in range(4) #first 4 events
]



learningrate = 1e-4
nbatch = 80000 #this is rather low, and can be set to a higher values e.g. when training on V100s

train.compileModel(learningrate=1e-3, #gets overwritten by CyclicLR callback anyway
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
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/5,
                                  max_lr = learningrate,
                                  step_size = 150)]+cb)

print("freeze BN 2")
# Note the submodel here its not just train.keras_model
#for l in train.keras_model.model.layers:
#    if 'gooey_batch_norm' in l.name:
#        l.max_viscosity = 0.95
#        l.fluidity_decay= 5e-5 #
#    if 'FullOCLoss' in l.name:
#        l.use_average_cc_pos = 0.
#        l.q_min = 1.0

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=10.
nbatch = 90000

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


