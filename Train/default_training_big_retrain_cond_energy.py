from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedConstructTensor, RaggedGlobalExchange, FusedRaggedGravNet_simple, CondensateAndSum
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model

# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply, SelectFeatures


# tf.compat.v1.disable_eager_execution()



def gravnet_model(Inputs, feature_dropout=-1.):
    nregressions = 5

    # I_data = tf.keras.Input(shape=(num_features,), dtype="float32")
    # I_splits = tf.keras.Input(shape=(1,), dtype="int32")

    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)

    x_data, x_row_splits = RaggedConstructTensor(name="construct_ragged")([I_data, I_splits])

    input_features = x_data  # these are going to be passed through for the loss

    x_basic = BatchNormalization(momentum=0.6)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    n_gravnet_layers = 7
    feat = []
    for i in range(n_gravnet_layers):
        n_filters = 128
        n_propagate = 96
        n_neighbours = 200

        x = RaggedGlobalExchange(name="global_exchange_"+str(i))([x, x_row_splits])
        x = Dense(64, activation='elu',name="dense_"+str(i)+"_a")(x)
        x = Dense(64, activation='elu',name="dense_"+str(i)+"_b")(x)
        x = Dense(64, activation='elu',name="dense_"+str(i)+"_c")(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = FusedRaggedGravNet_simple(n_neighbours=n_neighbours,
                                 n_dimensions=4,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 name='gravnet_' + str(i))([x, x_row_splits])
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(48, activation='elu',name="dense_compress_"+str(i))(x))

    x = Concatenate(name="concat_gravout")(feat)
    x = Dense(128, activation='elu',name="dense_last_a")(x)
    x = Dense(64, activation='elu',name="dense_last_b")(x)
    x = Dense(64, activation='elu',name="dense_last_c")(x)

    beta = Dense(1, activation='sigmoid', name="dense_beta")(x)
    
    eta = Dense(1, activation=None, name="dense_eta")(x)
    phi = Dense(1, activation=None, name="dense_phi")(x)
    ccoords = Dense(2, activation=None, name="dense_ccoords")(x)
    
    x_en = Dense(64, activation='elu', name="dense_en_a")(x)#herer so the other names remain the same
    input_energy = SelectFeatures(0,1, name="select_en")(input_features)
    energy_condensates,idxs = CondensateAndSum(radius=0.8, min_beta=0.1, name="condensate_en")([ccoords, beta, input_energy, x_row_splits])
    x_en = Concatenate(name="concat_en_cond")([ScalarMultiply(10, name="multi_en")(x_en),energy_condensates])
    x_en = Dense(64, activation='elu',name="dense_en_b")(x_en)
    energy = Dense(1, activation=None,name="dense_en_final")(x_en)

    print('input_features', input_features.shape)

    x = Concatenate(name="concat_final")([input_features, beta, energy, eta, phi, ccoords])

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
    import pretrained_models as ptm
    from DeepJetCore.modeltools import load_model, apply_weights_where_possible
    
    pretrained_model = load_model(ptm.get_model_path("default_training_big_model.h5"))
    train.setModel(gravnet_model)
    
    apply_weights_where_possible(train.keras_model,pretrained_model)
    
    # train.keras_model.dynamic=True
    train.compileModel(learningrate=1e-4,
                       loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
    ####### do not use metrics here - keras problem in TF 2.2rc0
    




from betaLosses import config as loss_config



print(train.keras_model.summary())

 # **2 #this will be an upper limit on vertices per batch

verbosity = 2
import os

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
callbacks = []
for i in range(10):
    plotoutdir = train.outputDir + "/event_" + str(i + 2)
    os.system('mkdir -p ' + plotoutdir)
    callbacks.append(
        plotEventDuringTraining(
            outputfile=plotoutdir + "/sn",
            samplefile=samplepath,
            after_n_batches=4000,
            batchsize=100000,
            on_epoch_end=False,
            use_event=2 + i)
    )



from configSaver import copyModules
copyModules(train.outputDir)

train.change_learning_rate(1e-4)

nbatch = 20000 

loss_config.energy_loss_weight = 0.001
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = False
loss_config.potential_scaling = 1.
loss_config.s_b = 1.
loss_config.position_loss_weight=0.001
loss_config.use_spectators=True
loss_config.log_energy=False

model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=100,
                                  additional_callbacks=callbacks)


nbatch = 50000

loss_config.energy_loss_weight = 1.
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = False
loss_config.potential_scaling = 1.
loss_config.s_b = 1.
loss_config.position_loss_weight=0.01
loss_config.use_spectators=False

train.change_learning_rate(3e-5)

model, history = train.trainModel(nepochs=10 + 1,
                                  batchsize=nbatch,
                                  run_eagerly=True,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  backup_after_batches=100,
                                  verbose=verbosity, )


loss_config.energy_loss_weight = 2.
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = False
loss_config.potential_scaling = 1.
loss_config.s_b = 1.
loss_config.position_loss_weight=0.1
loss_config.use_spectators=False

train.change_learning_rate(1e-5)
model, history = train.trainModel(nepochs=10 + 10 + 1,
                                  batchsize=nbatch,
                                  run_eagerly=True,
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  backup_after_batches=100,
                                  verbose=verbosity, )


