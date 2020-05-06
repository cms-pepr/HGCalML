from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from LayersRagged import RaggedConstructTensor, GraphFunctionFilters, RaggedGravNet, RaggedNeighborIndices, RaggedGlobalExchange, RaggedGravNet_simple, RaggedGravNet, GraphShapeFilters, RaggedNeighborBuilder
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model
from DeepJetCore.DJCLayers import SelectFeatures

# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


import pretrained_models as ptm
# tf.compat.v1.disable_eager_execution()

def where_did_hits_go(feat, coords, nbs, name, filters=0, ndense=[]):
    
    if not isinstance(ndense, list):
        ndense=[ndense]
    shapes = []
    for n in nbs:
        xc = RaggedNeighborBuilder()([coords,n])
        xf = RaggedNeighborBuilder()([feat,n])
        xs = GraphShapeFilters(n_filters=filters,
                                n_moments=2,
                                direct_output = True
                              )( [xc, xf] )
        shapes.append(xs)
        if len(nbs) == 1:
            shapes=xs
    if len(ndense)<1:
        if len(nbs) == 1:
            return Flatten()(shapes)
        return Flatten()(Concatenate()(shapes))
    
    if len(ndense) > 1:
        x = Concatenate()(shapes)
    else:
        x = shapes
    for i in range(len(ndense)):
        d = ndense[i]
        x = Dense(d, activation='elu',name="wdhg_"+name+str(i))(x)
    return Flatten()(x)
    
def where_did_hits_go_func(feat, coords, nbs, name, filters=4, ndense=[]):
    
    if not isinstance(ndense, list):
        ndense=[ndense]
    shapes = []
    for n in nbs:
        xc = RaggedNeighborBuilder()([coords,n])
        xf = RaggedNeighborBuilder()([feat,n])
        xs = GraphFunctionFilters(n_filters=filters,
                              )( [xc, xf] )
        shapes.append(xs)
        if len(nbs) == 1:
            shapes=xs
    if len(ndense)<1:
        if len(nbs) == 1:
            return Flatten()(shapes)
        return Flatten()(Concatenate()(shapes))
    
    if len(ndense) > 1:
        x = Concatenate()(shapes)
    else:
        x = shapes
    for i in range(len(ndense)):
        d = ndense[i]
        x = Dense(d, activation='elu',name="wdhggf_"+name+str(i))(x)
    return Flatten()(x)

def gravnet_model(Inputs, feature_dropout=-1.):
    
    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)

    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])

    input_features = x_data  # these are going to be passed through for the loss

    x_basic = BatchNormalization(momentum=0.6)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic
    
    in_coords_eta_phi_r = Concatenate()([SelectFeatures(1,3)(x),SelectFeatures(4,5)(x)])
    in_en = SelectFeatures(0,1)(x)
    
    #nb16  = RaggedNeighborIndices(16) ([in_coords_eta_phi_r,x_row_splits])
    nb32  = RaggedNeighborIndices(32) ([in_coords_eta_phi_r,x_row_splits])
    #nb64  = RaggedNeighborIndices(64) ([in_coords_eta_phi_r,x_row_splits])
    nb128 = RaggedNeighborIndices(128)([in_coords_eta_phi_r,x_row_splits])
    
    
    #x = where_did_hits_go_func(x_basic, in_coords_eta_phi_r, [nb32,nb128], "polar", filters=8, ndense=[32,16])
    #x = Dense(64, activation='elu')(x)
     
    n_filters = 0
    n_gravnet_layers = 4
    feat = [x]
    all_coords=[]
    coords = None
    idcs = None
    
    for i in range(n_gravnet_layers):
        
        n_filters = [128]
        n_propagate = 32
        n_neighbours = 255
        n_dimensions = 4+i

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x_o = BatchNormalization(momentum=0.6)(x)
        
        x,idcs,coords = RaggedGravNet(n_neighbours=n_neighbours,
                                 n_dimensions=n_dimensions,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 return_idx_and_space=True,
                                 calculate_moments=True,
                                 name='gravnet_' + str(i))([x_o, x_row_splits])
          
        all_coords.append(coords)
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(64, activation='elu')(x) )

    x = Concatenate()(feat)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)

    beta = Dense(1, activation='sigmoid')(x)
    
    energy_a = ScalarMultiply(1.)(Dense(1, activation='relu')(x))
    energy_b = ScalarMultiply(10.)(Dense(1, activation='relu')(x))
    energy_c = ScalarMultiply(100.)(Dense(1, activation='relu')(x))
    
    energy = Dense(1, activation=None)(Concatenate()([energy_a,energy_b,energy_c,]))
    
    eta = Dense(1, activation=None,kernel_initializer='zeros')(x)
    phi = Dense(1, activation=None,kernel_initializer='zeros')(x)
    ccoords = Dense(2, activation=None)(x)

    print('input_features', input_features.shape)

    x = Concatenate()([input_features, beta, energy, eta, phi, ccoords])

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])


train = training_base(testrun=False, resumeSilently=True, renewtokens=False)


from betaLosses import config as loss_config

loss_config.energy_loss_weight = 0.
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = True
loss_config.potential_scaling = 3.
loss_config.s_b = 1.

from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet_model)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=3e-5,
                   loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
 
    

train.change_learning_rate(1e-5)


print(train.keras_model.summary())

#exit()

nbatch = 8000  # **2 #this will be an upper limit on vertices per batch

verbosity = 2
import os

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
callbacks = []
for i in range(5):
    plotoutdir = train.outputDir + "/event_" + str(i)
    os.system('mkdir -p ' + plotoutdir)
    callbacks.append(
        plotEventDuringTraining(
            outputfile=plotoutdir + "/sn",
            samplefile=samplepath,
            cycle_colors=False,
            after_n_batches=100,
            batchsize=100000,
            on_epoch_end=False,
            use_event= i)
    )


from configSaver import copyModules
copyModules(train.outputDir)


print("It should save now")
model, history = train.trainModel(nepochs=10,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=callbacks)

exit()


model, history = train.trainModel(nepochs=5 + 1,
                                  batchsize=nbatch,
                                  run_eagerly=True,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  backup_after_batches=100,
                                  verbose=verbosity, )

train.change_learning_rate(1e-4)
model, history = train.trainModel(nepochs=99 + 5 + 1,
                                  batchsize=nbatch,
                                  run_eagerly=True,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  backup_after_batches=100,
                                  verbose=verbosity, )


