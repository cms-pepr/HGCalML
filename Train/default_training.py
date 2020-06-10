from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedConstructTensor, RaggedGlobalExchange, FusedRaggedGravNet_simple, FusedRaggedGravNet
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model

# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining, plotGravNetCoordinatesDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


# tf.compat.v1.disable_eager_execution()



n_gravnet_layers = 5
def gravnet_model(Inputs, feature_dropout=-1.):
    nregressions = 5

    # I_data = tf.keras.Input(shape=(num_features,), dtype="float32")
    # I_splits = tf.keras.Input(shape=(1,), dtype="int32")

    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)

    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])

    input_features = x_data  # these are going to be passed through for the loss

    x_basic = BatchNormalization(momentum=0.6)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    feat = []
    allcoords=[]
    for i in range(n_gravnet_layers):
        n_filters = 128 #no additional message passing in standard configuration
        n_propagate = 96
        n_neighbours = 200
        if i % 2:
            n_neighbours = 40

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = BatchNormalization(momentum=0.6)(x)
        x,coords = FusedRaggedGravNet(n_neighbours=n_neighbours,
                                 n_dimensions=4,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 n_modifiers=0, #no distance modifiers in standard configuration
                                 name='gravnet_' + str(i))([x, x_row_splits])
                                 
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(48, activation='elu')(x))
        allcoords.append(coords)

    x = Concatenate()(feat)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)

    beta = Dense(1, activation='sigmoid')(x)
    energy = ScalarMultiply(100.)(Dense(1, activation=None)(x))
    eta = Dense(1, activation=None)(x)
    phi = Dense(1, activation=None)(x)
    ccoords = Dense(2, activation=None)(x)

    print('input_features', input_features.shape)

    x = Concatenate()([input_features, beta, energy, eta, phi, ccoords]+allcoords)


    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])


train = training_base(testrun=False, resumeSilently=True, renewtokens=False)

from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss

# optimizer = Adam(lr=1e-4)
# train.setCustomOptimizer(optimizer)

# train.setDJCKerasModel(simple_model)

if not train.modelSet():
    train.setModel(gravnet_model)  # ser_simple_model)
    # train.keras_model.dynamic=True
    train.compileModel(learningrate=1e-4,
                       loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
    ####### do not use metrics here - keras problem in TF 2.2rc0
    




from betaLosses import config as loss_config

loss_config.energy_loss_weight = 0.0001
loss_config.use_energy_weights = False
loss_config.q_min = 1.0
loss_config.no_beta_norm = False
loss_config.potential_scaling = 1.
loss_config.s_b = 1.
loss_config.position_loss_weight=0.001




print(train.keras_model.summary())

nbatch = 20000  # **2 #this will be an upper limit on vertices per batch

verbosity = 2
import os

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
callbacks = []
for i in range(5):
    plotoutdir = train.outputDir + "/event_" + str(i + 2)
    os.system('mkdir -p ' + plotoutdir)
    callbacks.append(
        plotEventDuringTraining(
            outputfile=plotoutdir + "/sn",
            samplefile=samplepath,
            after_n_batches=600,
            batchsize=100000,
            on_epoch_end=False,
            use_event=2 + i)
    )

#plot the coordinate space for the last layer
for e in range(5):
    
    i=n_gravnet_layers-1
    
    plotoutdir = train.outputDir + "/coords_" +str(e)+"_"+ str(i)
    os.system('mkdir -p ' + plotoutdir)
    callbacks.append(
        plotGravNetCoordinatesDuringTraining(
            outputfile=plotoutdir + "/c",
            start_pred_index=4*i,
            end_pred_index=4*(i+1),
            samplefile=samplepath,
            after_n_batches=600,
            batchsize=100000,
            on_epoch_end=False,
            use_event= 2 + e
            )
        )

from configSaver import copyModules
copyModules(train.outputDir)

train.change_learning_rate(1e-4)

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


