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

#tf.compat.v1.disable_eager_execution()




num_features = 3
num_vertices_times_batch = 10000
num_vertices = 1000
n_neighbours = 40
passing_operations = 2

class simple_model(DJCKerasModel):
    def __init__(self,stuff=1):
        
        super(simple_model, self).__init__()
        
        self.dense = Dense(128)
        
        self.beta    = Dense(1, activation='sigmoid')
        self.energy  = Dense(1, activation=None)
        self.eta     = Dense(1, activation=None)
        self.phi     = Dense(1, activation=None)
        self.ccoords = Dense(2, activation=None)
        
        self.ncalls=0
        
    def call(self, Inputs):
        #print(Inputs)
        
        print('call ',self.ncalls)
        self.ncalls+=1
        
        x = self.dense(Inputs[0])
        
        x = Concatenate()([
        self.beta    (x),
        self.energy  (x),
        self.eta     (x),
        self.phi     (x),
        self.ccoords (x)])
        
        return [x,x]
        

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

    n_gravnet_layers = 4
    # I_data = tf.keras.Input(shape=(num_features,), dtype="float32")
    # I_splits = tf.keras.Input(shape=(1,), dtype="int32")

    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)


    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])

    #x_row_splits = tf.Print(x_row_splits, [x_row_splits], "Row splits from tensorflow",summarize=200)

    
    feats = []

    #TODO: Jan center phi and select features you may have to implement yourself. I am not sure about its format etc.

    #x_data = tf.Print(x_data,[tf.shape(x_data)],'x_data.shape ')

    x_basic = BatchNormalization(momentum=0.9)(x_data)  # mask_and_norm is just batch norm now
    x = x_basic

    n_filters = 0
    for i in range(n_gravnet_layers):
        n_filters = 32 + 24 * i
        n_propagate = 8 + 8 * i

        x = RaggedGlobalExchange()([x, x_row_splits])

        x = Dense(n_filters, activation='elu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Dense(n_filters, activation='elu')(x)
        x = Dropout(0.05)(x) #just some noise
        x = Dense(n_filters, activation='elu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        #x = Concatenate()([x_basic, x])

        x = RaggedGravNet_simple(n_neighbours=n_neighbours,
                           n_dimensions=4,
                           n_filters=n_filters,
                           n_propagate=n_propagate,
                           name='gravnet_' + str(i))([x, x_row_splits])
        x = BatchNormalization(momentum=0.9)(x)

    #x = Concatenate()([x_basic, x])
    x = Dense(n_filters, activation='elu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dense(n_filters, activation='elu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dense(n_filters, activation='elu')(x)
    # x = Concatenate()([x, x_all]) # TODO: Jan check this
    x = Dense(64, activation='elu', name='last_correction')(x)
    
    beta = Dense(1, activation='sigmoid')(x)
    energy = Dense(1, activation=None)(x)
    eta = Dense(1, activation=None)(x)
    phi = Dense(1, activation=None)(x)
    ccoords = Dense(2, activation=None)(x)

    x = Concatenate()([beta,energy,eta,phi,ccoords])

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])

    

train=training_base(testrun=True,resumeSilently=True,renewtokens=False)


from Losses import min_beta_loss_rowsplits, min_beta_loss_truth, pre_training_loss, null_loss

#train.setDJCKerasModel(simple_model)
train.setModel(ser_simple_model)
#train.keras_model.dynamic=True
train.compileModel(learningrate=1e-3,
                   loss=[min_beta_loss_truth,min_beta_loss_rowsplits],#fraction_loss)
                   ) #clipnorm=1.) 


print(train.keras_model.summary())


nbatch=300000#**2 #this will be an upper limit on vertices per batch

verbosity=2
import os

# callbacks=[]
# for i in range(1):
#     plotoutdir=train.outputDir+"/event_"+str(i+2)
#     os.system('mkdir -p '+plotoutdir)
#     callbacks.append(
#         plotEventDuringTraining(
#             outputfile=plotoutdir+"/sn",
#             samplefile="/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/ml.TestDataSet/Xmas19/windowntup_99.djctd",
#             after_n_batches=200,
#             batchsize=100000,
#             on_epoch_end=False,
#             use_event=2+i)
#         )
#train.saveModel("testmodel.h5")

#train.loadModel("TESTTF115_4/testmodel.h5")

print(train.keras_model.summary())
#exit()

model,history = train.trainModel(nepochs=1, 
                                 run_eagerly=True,
                                 batchsize=nbatch,
                                 batchsize_use_sum_of_squares=False,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,)




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

