from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedConstructTensor, RaggedGravNet, RaggedGlobalExchange, RaggedGravNet_simple, RaggedGravNet, GraphShapeFilters, RaggedNeighborBuilder
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model

# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


import pretrained_models as ptm
# tf.compat.v1.disable_eager_execution()


def gravnet_model(Inputs, feature_dropout=-1.):
    
    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)

    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])

    input_features = x_data  # these are going to be passed through for the loss

    x_basic = BatchNormalization(momentum=0.6)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    n_gravnet_layers = 4
    feat = [x_basic]
    for i in range(n_gravnet_layers):
        
        n_filters = [32,64,64]
        n_propagate = 32
        n_neighbours = 128
        n_dimensions = 4

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x_o = BatchNormalization(momentum=0.6)(x)
        
        x,idcs,coords = RaggedGravNet(n_neighbours=n_neighbours,
                                 n_dimensions=n_dimensions,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 return_idx_and_space=True,
                                 name='gravnet_' + str(i))([x_o, x_row_splits])
          
          
        neigh_coords = RaggedNeighborBuilder()([coords,idcs])
        shape_x_o = Dense(4, activation='elu')(x_o)  
        neigh_feat =   RaggedNeighborBuilder()([shape_x_o,idcs])
        x = Concatenate()([x, GraphShapeFilters(
                                n_filters=32,
                                n_moments=3,
                              )( [neigh_coords, neigh_feat] )
                           ])  
        x = Dense(64, activation='elu')(x)  
        x = Dense(32, activation='elu')(x)                   
        feat.append(x)

    x = Concatenate()(feat)
    x = Dense(128, activation='elu')(x)
    x = BatchNormalization(momentum=0.6)(x)
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

from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet_model)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=3e-5,
                   loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
 
    

train.change_learning_rate(5e-4)


print(train.keras_model.summary())

#exit()

nbatch = 25000  # **2 #this will be an upper limit on vertices per batch

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


