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
from GravNetLayersRagged import ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
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
    
    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits
    
    feat_norm = ProcessFeatures()(feat)
    allfeat=[feat_norm]
    x = feat_norm
    
    backgatheredids=[]
    gatherids=[]
    backgathered = []
    backgathered_coords = []
    
    
    energy = SelectFeatures(0,1)(feat)
    coords = SelectFeatures(5,8)(feat_norm)
    coords = Dense(3)(coords)#just rotation and scaling
    
    #see whats there
    nidx, dist = KNN(K=96,radius=1.0)([coords,rs])
    x = Dense(4, activation='elu',name='pre_pre_dense')(x) #just a few features are enough here
    #this can be full blown because of the small number of input features
    x_c = SoftPixelCNN(length_scale=1.0, mode='full', subdivisions=4, name='prePCNN')([coords,x,nidx])
    x = Concatenate()([x,x_c])
    #this is going to be among the most expensive operations:
    x = Dense(128, activation='elu',name='pre_dense_a')(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(64, activation='elu',name='pre_dense_b')(x)
    x = BatchNormalization(momentum=0.6)(x)
    
    allfeat.append(Dense(16, activation='elu',name='feat_compress_pre')(x))
    backgathered_coords.append(coords)
    
    sel_gidx = gidx_orig
    
    
    total_iterations=5
    
    for i in range(total_iterations):
        
        #cluster first
        hier = Dense(1,activation='sigmoid')(x)
        x_cl, rs, bidxs, sel_gidx, x, t_idx = LocalClusterReshapeFromNeighbours(
                 K=8, 
                 radius=0.1, 
                 print_reduction=True, 
                 loss_enabled=True, 
                 loss_scale = 2., 
                 loss_repulsion=0.3,
                 print_loss=True,
                 name='clustering_'+str(i)
                 )([x, dist, hier, nidx, rs, sel_gidx, x, t_idx, t_idx])
                 
        gatherids.append(bidxs)
        x_cl = Dense(128, activation='elu',name='dense_clc_'+str(i))(x_cl)
        x = Concatenate()([x,x_cl])
        
        pixelcompress=4
        nneigh = 32+4*i
        nfilt = 32+4*i
        nprop = 32
        n_dimensions = 3 #make it plottable
        
        x_gn, coords, nidx, dist = RaggedGravNet(n_neighbours=nneigh,
                                              n_dimensions=n_dimensions,
                                              n_filters=nfilt,
                                              n_propagate=nprop)([x, rs])
                                              
                                              
        subdivisions = 4
        
                                              
        #add more shape information
        x_sp = Dense(pixelcompress,activation='elu')(x)
        x_sp = BatchNormalization(momentum=0.6)(x_sp)
        x_sp = SoftPixelCNN(length_scale=1.,mode='full', subdivisions=4)([coords,x_sp,nidx])
        x_sp = Dense(128, activation='elu',name='dense_spc_'+str(i))(x_sp)
        
        x_mp = MessagePassing([32,32,16,16,8,8])([x,nidx])
        x_mp = Dense(nfilt, activation='elu',name='dense_mpc_'+str(i))(x_sp)
        
        x = Concatenate()([x,x_sp,x_gn,x_mp])
        #check and compress it all                                      
        x = Dense(128, activation='elu',name='dense_a_'+str(i))(x)
        x = Dense(64, activation='elu',name='dense_b_'+str(i))(x)
        x = BatchNormalization(momentum=0.6)(x)
        
        x_r=x
        #record more and more the deeper we go
        if i < total_iterations-1:
            x_r = Dense(12*(i+1), activation='elu',name='dense_rec_'+str(i))(x)
        
        allfeat.append(MultiBackGather()([x_r, gatherids]))
        
        backgatheredids.append(MultiBackGather()([sel_gidx, gatherids]))
        backgathered_coords.append(MultiBackGather()([coords, gatherids]))      
        
        
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
                                         energy_loss_weight=1e-2,
                                         position_loss_weight=1e-2,
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
                                         rs]+backgatheredids+backgathered_coords)





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

samplepath='/eos/home-j/jkiesele/DeepNtuples/HGCal/Sept2020_19_production_1x1/999_windowntup.djctd'
publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

cb = [plotClusteringDuringTraining(
           use_backgather_idx=7+i,
           outputfile=train.outputDir + "/plts/sn"+str(i)+'_',
           samplefile=  samplepath,
           after_n_batches=100,
           on_epoch_end=False,
           publish=publishpath+"_cl_"+str(i),
           use_event=0) 
    for i in [0,4]]

cb += [   
    plotEventDuringTraining(
            outputfile=train.outputDir + "/plts2/sn0",
            samplefile=samplepath,
            after_n_batches=100,
            batchsize=200000,
            on_epoch_end=False,
            publish = publishpath+"_event_"+ str(0),
            use_event=0)
    
    ]

learningrate = 1e-4
nbatch = 110000 #quick first training with simple examples = low # hits

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)


model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
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
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate,
                                  max_lr = learningrate*2.,
                                  step_size = 100)]+cb)

