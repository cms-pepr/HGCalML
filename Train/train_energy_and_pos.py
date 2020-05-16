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

# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply

import pretrained_models as ptm
# tf.compat.v1.disable_eager_execution()


train = training_base(testrun=False, resumeSilently=True, renewtokens=False)


from betaLosses import config as loss_config

loss_config.energy_loss_weight = 1.
loss_config.use_energy_weights = False
loss_config.q_min = 0.5
loss_config.no_beta_norm = False
loss_config.potential_scaling = 3.
loss_config.s_b = 1.


from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss


train.loadModel(ptm.get_model_path("shah_rukh_apr30_2020.h5"))

train.compileModel(learningrate=1e-4,
                   loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])


print(train.keras_model.summary())

#exit()

nbatch = 10000  # **2 #this will be an upper limit on vertices per batch

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
            after_n_batches=500,
            batchsize=100000,
            on_epoch_end=False,
            use_event= i)
    )





train.change_learning_rate(1e-5)
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

train.change_learning_rate(2e-4)

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


