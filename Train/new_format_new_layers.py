'''
First training file using new format (check the prediction).
Can be trained using the *latest* deepjetcore (there was a minor change to allow for an arbitrary number of predictions for keras models).
A dataset can be found here: /eos/home-j/jkiesele/DeepNtuples/HGCal/Sept2020_19_production_1x1
'''
import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from tensorflow.keras.layers import Dense, Concatenate, GaussianDropout
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model


from DeepJetCore.modeltools import fixLayersContaining
# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from plotting_callbacks import plotEventDuringTraining
from ragged_callbacks import plotRunningPerformanceMetrics
from DeepJetCore.DJCLayers import ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs

from Layers import LocalClusterReshapeFromNeighbours,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackGather, KNN, MessagePassing
from datastructures import TrainData_OC 
td=TrainData_OC()

def gravnet_model(Inputs, feature_dropout=-1., addBackGatherInfo=True):
    
    feat,  t_idx, t_energy, t_pos, t_time, t_pid, row_splits = td.interpretAllModelInputs(Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)
    energy = SelectFeatures(0,1)(feat)
    
    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits
    
    #n_cluster_coords=6
    
    x = feat #Dense(64,activation='elu')(feat)
    
    backgatheredids=[]
    gatherids=[]
    backgathered = []
    backgathered_en = []
    
    allfeat=[x]
    
    sel_gidx = gidx_orig
    for i in range(6):
        

        pixelcompress=16+8*i
        nneigh = 8 + 32*i
        n_dimensions = int(4+i)
        
        x = BatchNormalization(momentum=0.6)(x)
        #do some processing
        x = Dense(64, activation='elu',name='dense_a_'+str(i))(x)
        x = BatchNormalization(momentum=0.6)(x)
        x, coords, nidx, dist = RaggedGravNet(n_neighbours=nneigh,
                                              n_dimensions=n_dimensions,
                                              n_filters=64,
                                              n_propagate=64)([x, rs])
        x = Concatenate()([x,MessagePassing([64,32,16,8])([x,nidx])])
        
        #allfeat.append(x)
        
        x = Dense(128,activation='elu')(x)
        x = Dense(pixelcompress,activation='elu')(x)
        
        x_sp = SoftPixelCNN(length_scale=1.)([coords,x,nidx])
        x = Concatenate()([x,x_sp])
        x = BatchNormalization(momentum=0.6)(x)
        x = Dense(128,activation='elu')(x)
        x = Dense(pixelcompress,activation='elu')(x)
        
        hier = Dense(1, activation='sigmoid',name='hier_'+str(i))(x)#clustering hierarchy
        
        dist = ScalarMultiply(1/(3.*float(i+0.5)))(dist)
        
        x, rs, bidxs, t_idx = LocalClusterReshapeFromNeighbours(
                 K=5, 
                 radius=0.1, 
                 print_reduction=True, 
                 loss_enabled=True, 
                 loss_scale = 2., 
                 loss_repulsion=0.5,
                 print_loss=True)([x, dist, hier, nidx, rs, t_idx, t_idx])
        
        print('>>>>x0',x.shape)
        gatherids.append(bidxs)
        if addBackGatherInfo:
            backgatheredids.append(MultiBackGather()([sel_gidx, gatherids]))
        
        print('>>>>x1',x.shape)    
        x = BatchNormalization(momentum=0.6)(x)
        x_record = Dense(2*pixelcompress,activation='elu')(x)
        x_record = BatchNormalization(momentum=0.6)(x_record)
        allfeat.append(MultiBackGather()([x_record, gatherids]))
        
        
    
    
    x = Concatenate(name='allconcat')(allfeat)
    x = Dense(128, activation='elu', name='alldense')(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(64, activation='elu')(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Concatenate()([feat,x])
    x = BatchNormalization(momentum=0.6)(x)

    pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id = create_outputs(x,feat)
    
    #loss
    pred_beta = LLFullObjectCondensation(print_loss=True,
                                         energy_loss_weight=1e-3,
                                         position_loss_weight=1e-3,
                                         timing_loss_weight=1e-3,
                                         )([pred_beta, pred_ccoords, pred_energy, 
                                            pred_pos, pred_time, pred_id,
                                            orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid,
                                            row_splits])

    return Model(inputs=Inputs, outputs=[pred_beta, 
                                         pred_ccoords,
                                         pred_energy, 
                                         pred_pos, 
                                         pred_time, 
                                         pred_id,
                                         rs]+backgatheredids)





train = training_base(testrun=False, resumeSilently=True, renewtokens=False)


if not train.modelSet():

    train.setModel(gravnet_model)
    train.setCustomOptimizer(tf.keras.optimizers.Nadam())

    train.compileModel(learningrate=1e-4,
                       loss=None)
    
    print(train.keras_model.summary())
    #exit()

verbosity = 2
import os

from plotting_callbacks import plotClusteringDuringTraining

samplepath='/afs/cern.ch/work/j/jkiesele/HGCal/HGCalML/testdata/conv/1949_windowntup.djctd'

cb = [plotClusteringDuringTraining(
           use_backgather_idx=7+i,
           outputfile=train.outputDir + "/plts/sn"+str(i)+'_',
           samplefile=  samplepath,
           after_n_batches=400,
           on_epoch_end=False,
           use_event=0) 
    for i in range(4,6)]

cb += [   
    plotEventDuringTraining(
            outputfile=train.outputDir + "/plts2/sn0",
            samplefile=samplepath,
            after_n_batches=400,
            batchsize=200000,
            on_epoch_end=False,
            #publish = publishpath+"_event_"+ str(ev),
            use_event=0)
    
    ]

learningrate = 5e-5
nbatch = 110000 #quick first training with simple examples = low # hits

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)


model, history = train.trainModel(nepochs=10,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=10,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate,
                                  max_lr = learningrate*2.,
                                  step_size = 100)]+cb)

print("freeze BN")
for l in train.keras_model.layers:
    if isinstance(l, BatchNormalization):
        l.trainable=False
    if 'GravNetLLLocalClusterLoss' in l.name:
        l.active=False
        
#also stop GravNetLLLocalClusterLoss* from being evaluated

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)

model, history = train.trainModel(nepochs=121,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=10,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate,
                                  max_lr = learningrate*2.,
                                  step_size = 100)]+cb)

