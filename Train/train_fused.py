from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedConstructTensor, RaggedGravNet, RaggedGlobalExchange, RaggedGravNet_simple, FusedRaggedGravNet
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

    x_basic = BatchNormalization(momentum=0.9)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    n_gravnet_layers = 1
    feat = [x_basic]
    for i in range(n_gravnet_layers):
        
        n_filters = [32,32,32,32]#,32,32]
        n_propagate = 32
        n_neighbours = 96
        n_dimensions = 4

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = FusedRaggedGravNet(n_neighbours=n_neighbours,
                                 n_dimensions=n_dimensions,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 name='gravnet_' + str(i))([x, x_row_splits])
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(x)

    #if n_gravnet_layers>1:
    x = Concatenate()(feat)
    #elif n_gravnet_layers==1:
    #    x = feat[0]
    #else:
    #    x = x_basic
    x = Dense(128, activation='elu')(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    #x = Concatenate()([x,x_basic])

    beta = Dense(1, activation='sigmoid')(x)
    
    energy_a = ScalarMultiply(1.)(Dense(1, activation='relu')(x))
    energy_b = ScalarMultiply(10.)(Dense(1, activation='relu')(x))
    energy_c = ScalarMultiply(100.)(Dense(1, activation='relu')(x))
    energy = Dense(1, activation=None)(Concatenate()([energy_a,energy_b,energy_c,]))
    
    eta = Dense(1, activation=None)(x)
    phi = Dense(1, activation=None)(x)
    ccoords = Dense(2, activation=None)(x)

    print('input_features', input_features.shape)

    x = Concatenate()([input_features, beta, energy, eta, phi, ccoords])

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])


train = training_base(testrun=False, resumeSilently=True, renewtokens=False)

from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss, pretrain_obj_cond_loss_rowsplits, pretrain_obj_cond_loss_truth

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet_model)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=3e-5,
                   loss=[pretrain_obj_cond_loss_truth, pretrain_obj_cond_loss_rowsplits])
 
    

print(train.keras_model.summary())

#exit()

from betaLosses import config as loss_config

loss_config.energy_loss_weight = 0.
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = False
loss_config.potential_scaling = 3.
loss_config.s_b = 1.
loss_config.position_scaling=0.5


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
            after_n_batches=300,
            batchsize=100000,
            on_epoch_end=False,
            use_event= i)
    )


from configSaver import copyModules
copyModules(train.outputDir)


nbatch = 120000  # start with small events only to get a first starting point

train.change_learning_rate(3e-3)

from DeepJetCore.training.DeepJet_callbacks import batch_callback_begin

callbacks.append(batch_callback_begin(outputDir=train.outputDir ,plot_frequency=10))

print("It should save now")
model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=callbacks)


train.compileModel(learningrate=3e-4,
                   loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
nbatch = 150000 
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


