import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from LayersRagged import RaggedConstructTensor, RaggedGravNet, RaggedGlobalExchange
from tensorflow.keras.layers import Dense, Concatenate
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model



tf.compat.v1.disable_eager_execution()




num_features = 3
num_vertices_times_batch = 10000
num_vertices = 1000
n_neighbours = 40
passing_operations = 2

def gravnet_model(Inputs, feature_dropout=-1.):
    nregressions=5

    n_gravnet_layers = 4
    # I_data = tf.keras.Input(shape=(num_features,), dtype="float32")
    # I_splits = tf.keras.Input(shape=(1,), dtype="int32")

    I_data = Inputs[0]
    I_splits = tf.cast(Inputs[1], tf.int32)


    x_data, x_row_splits = RaggedConstructTensor()([I_data, I_splits])

    x_row_splits = tf.Print(x_row_splits, [x_row_splits], "Row splits from tensorflow")

    coords = []
    feats = []

    #TODO: Jan center phi and select features you may have to implement yourself. I am not sure about its format etc.

    x_basic = BatchNormalization(momentum=0.3)(x_data)  # mask_and_norm is just batch norm now


    n_filters = 0
    for i in range(n_gravnet_layers):
        n_filters = 32 + 24 * i
        n_propagate = 8 + 8 * i

        x = RaggedGlobalExchange()([x_data, x_row_splits])

        x = Dense(n_filters, activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dense(n_filters, activation='elu')(x)
        x = Dense(n_filters, activation='elu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x_basic, x])

        x, coord = RaggedGravNet(n_neighbours=n_neighbours,
                           n_dimensions=4,
                           n_filters=n_filters,
                           n_propagate=n_propagate,
                           additional_message_passing=passing_operations,
                           name='gravnet_' + str(i),
                           also_coordinates=True,
                           feature_dropout=feature_dropout)([x, x_row_splits])
        print(coord.shape)
        coords.append(coord)
        x = BatchNormalization()(x)

    x = Concatenate()([x_basic, x])
    x = Dense(n_filters, activation='elu')(x)
    x = Dense(n_filters, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(n_filters, activation='elu')(x)
    x = BatchNormalization()(x)

    x = Dense(nregressions, activation=None)(x)
    x = BatchNormalization()(x)
    # x = Concatenate()([x, x_all]) # TODO: Jan check this
    x = Dense(64, activation='elu', name='last_correction')(x)
    x = Dense(nregressions, activation=None, kernel_initializer='zeros')(x)

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])

    

train=training_base(testrun=False,resumeSilently=True,renewtokens=False)


from Losses import min_beta_loss_rowsplits, min_beta_loss_truth, pre_training_loss, null_loss

train.setModel(gravnet_model,feature_dropout=-1)
    
train.compileModel(learningrate=1e-10,
                   loss=[min_beta_loss_truth,min_beta_loss_rowsplits],#fraction_loss)
                   clipnorm=0.001) 

print(train.keras_model.summary())

nbatch=1500 #this will be an upper limit on vertices per batch
verbosity=2

model,history = train.trainModel(nepochs=5, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity)



