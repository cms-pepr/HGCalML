from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedConstructTensor, RaggedGravNet, RaggedGlobalExchange, RaggedGravNet_simple
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply

#tf.compat.v1.disable_eager_execution()



def ser_simple_model(Inputs):
    
    
    x = Dense(128)(Inputs[0])
    
    beta    = Dense(1, activation='sigmoid') (x) 
    energy  = Dense(1, activation=None)      (x) 
    eta     = Dense(1, activation=None)      (x) 
    phi     = Dense(1, activation=None)      (x) 
    ccoords = Dense(2, activation=None)      (x) 

    x = Concatenate()([
        beta    ,
        energy  ,
        eta     ,
        phi     ,
        ccoords ])
        
    return Model(inputs=Inputs, outputs=[x,x])


def gravnet_model(Inputs, feature_dropout=-1.):
    nregressions=5

    
    # I_data = tf.keras.Input(shape=(num_features,), dtype="float32")
    # I_splits = tf.keras.Input(shape=(1,), dtype="int32")

    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)

    

    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])
  
    input_features = x_data #these are going to be passed through for the loss
    
    x_basic = BatchNormalization(momentum=0.6)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    n_gravnet_layers = 6
    feat=[x_basic]
    for i in range(n_gravnet_layers):
        n_filters = 96
        n_propagate = 32

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = RaggedGravNet_simple(n_neighbours=80,
                           n_dimensions=4,
                           n_filters=n_filters,
                           n_propagate=n_propagate,
                           name='gravnet_' + str(i))([x, x_row_splits])
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(32, activation='elu')(x))

    x = Concatenate()(feat)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    
    beta = Dense(1, activation='sigmoid')(x)
    energy = ScalarMultiply(100.) (Dense(1, activation=None)(x))
    eta = Dense(1, activation=None)(x)
    phi = Dense(1, activation=None)(x)
    ccoords = Dense(2, activation=None)(x)
    
    print('input_features',input_features.shape)

    x = Concatenate()([input_features, beta,energy,eta,phi,ccoords])

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])

    

train=training_base(testrun=False,resumeSilently=True,renewtokens=False)


from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss

#train.setDJCKerasModel(simple_model)
train.setModel(gravnet_model)#ser_simple_model)
#train.keras_model.dynamic=True
train.compileModel(learningrate=1e-4,
                   loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits],#fraction_loss)
                   ) #clipnorm=1.) 


print(train.keras_model.summary())


nbatch=30000#**2 #this will be an upper limit on vertices per batch

verbosity=2
import os

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
callbacks=[]
for i in range(4):
    plotoutdir=train.outputDir+"/event_"+str(i+2)
    os.system('mkdir -p '+plotoutdir)
    callbacks.append(
        plotEventDuringTraining(
            outputfile=plotoutdir+"/sn",
            samplefile=samplepath,
            after_n_batches=100,
            batchsize=100000,
            on_epoch_end=False,
            use_event=2+i)
        )
#train.saveModel("testmodel.h5")

#train.loadModel("TESTTF115_4/testmodel.h5")

#exit()

model,history = train.trainModel(nepochs=10, 
                                 run_eagerly=True,
                                 batchsize=nbatch,
                                 batchsize_use_sum_of_squares=False,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=callbacks)




exit()

train.change_learning_rate(2e-4)

model,history = train.trainModel(nepochs=5+1, 
                                 batchsize=nbatch,
                                 run_eagerly=True,
                                 batchsize_use_sum_of_squares=False,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,)

train.change_learning_rate(1e-4)
model,history = train.trainModel(nepochs=99+5+1, 
                                 batchsize=nbatch,
                                 run_eagerly=True,
                                 batchsize_use_sum_of_squares=False,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,)

