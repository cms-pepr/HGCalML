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
from tensorflow.keras.callbacks import Callback
import DeepJetCore.DataCollection as dc
from tensorflow import keras
from tensorflow.keras.models import load_model
import Layers
from numba import jit
import math

from DeepJetCore.training.training_base import custom_objects_list
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


# tf.compat.v1.disable_eager_execution()


class WeightsSaver(Callback):
    def __init__(self, N, out):
        self.N = N
        self.batch = 0
        self.out = out

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = self.out + '/latest.h5'
            self.model.save_weights(name)
        self.batch += 1


def ser_simple_model(Inputs):
    x = Dense(128)(Inputs[0])

    beta = Dense(1, activation='sigmoid')(x)
    energy = Dense(1, activation=None)(x)
    eta = Dense(1, activation=None)(x)
    phi = Dense(1, activation=None)(x)
    ccoords = Dense(2, activation=None)(x)

    x = Concatenate()([
        beta,
        energy,
        eta,
        phi,
        ccoords])

    return Model(inputs=Inputs, outputs=[x, x])


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
    n_gravnet_layers = 5
    feat = [x_basic]
    for i in range(n_gravnet_layers):
        n_filters = 128
        n_propagate = 64
        n_neighbours = 200
        if i % 2:
            n_neighbours = 40

        x = RaggedGlobalExchange()([x, x_row_splits])
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = RaggedGravNet_simple(n_neighbours=n_neighbours,
                                 n_dimensions=4,
                                 n_filters=n_filters,
                                 n_propagate=n_propagate,
                                 name='gravnet_' + str(i))([x, x_row_splits])
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(48, activation='elu')(x))

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

    x = Concatenate()([input_features, beta, energy, eta, phi, ccoords])

    # x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = x

    # outputs = tf.tuple([predictions, x_row_splits])

    return Model(inputs=Inputs, outputs=[predictions, predictions])


# train = training_base(testrun=False, resumeSilently=True, renewtokens=False)

from Losses import obj_cond_loss_truth, obj_cond_loss_rowsplits, null_loss

# optimizer = Adam(lr=1e-4)
# train.setCustomOptimizer(optimizer)

# train.setDJCKerasModel(simple_model)
# train.setModel(gravnet_model)  # ser_simple_model)
# # train.keras_model.dynamic=True
# train.compileModel(learningrate=1e-4,
#                    loss=[obj_cond_loss_truth, obj_cond_loss_rowsplits])
# ####### do not use metrics here - keras problem in TF 2.2rc0


# print(train.keras_model.summary())

nbatch = 15000  # **2 #this will be an upper limit on vertices per batch

verbosity = 2
import os


data = dc.DataCollection('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_sample_spectators/test/snapshot.djcdc')

data.setBatchSize(20000)
data.invokeGenerator()
nbatches = data.generator.getNBatches()
print("The data has",nbatches,"batches.")
gen = data.generatorFunction()


inputs = []

input_shapes = [[9],[1]]

for s in input_shapes:
    inputs.append(keras.layers.Input(shape=s))

i = 0

model2 = gravnet_model(inputs)

# print(model2)



model2.build(input_shapes)
s = '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/HGCalML-tbr-remote/Train/train-data/grav-train-etax/latest5_nnn.h5'
model2 = load_model(s, custom_objects=custom_objects_list)


# model2.
# model2.load_weights(s)

sum = 0

for layer in model2.layers:
    weights = layer.get_weights()
    sum_this = 0
    for w in weights:
        sum_this += tf.reduce_sum(w)
    sum += sum_this



model2.run_eagerly = True


# model2.summary()

print("Wait is it already loaded?")

# 0/0


ragged_constructor = RaggedConstructTensor()


num_real_showers = []
num_predicted_showers = []

num_found_g = []
num_missed_g = []
num_fakes_g = []

iii = 0


def find_uniques_from_betas(betas, coords, dist_threshold):

    n2_distances = np.sqrt(np.sum(np.abs(np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)), axis=-1))
    betas_checked = np.zeros_like(betas) - 1

    index = 0

    arange_vector = np.arange(len(betas))

    while True:
        betas_remaining = betas[betas_checked==-1]
        arange_remaining = arange_vector[betas_checked==-1]

        if len(betas_remaining)==0:
            break

        max_beta = arange_remaining[np.argmax(betas_remaining)]


        n2 = n2_distances[max_beta]

        distances_less = np.logical_and(n2<dist_threshold, betas_checked==-1)
        betas_checked[distances_less] = index

        index += 1


    return betas_checked


def x_function(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment):
    global  iii, num_predicted_showers, num_real_showers, num_fakes_g, num_found_g, num_missed_g

    iii+=1
    uniques = tf.unique(classes_this_segment)[0].numpy()





    beta_all = pred_this_segment[:, -6]
    is_spectator = np.logical_not(y_this_segment[:, 14])
    is_spectator = np.logical_and(is_spectator, beta_all>0.1)
    is_spectator = np.logical_and(is_spectator, classes_this_segment>=0)


    beta_all_filtered = beta_all[is_spectator==1]


    clustering_coords_all = pred_this_segment[:, -2:]
    clustering_coords_all_filtered = clustering_coords_all[is_spectator==1, :]

    labels = find_uniques_from_betas(beta_all_filtered, clustering_coords_all_filtered, dist_threshold=0.8)
    num_showers_this_segment = len(np.unique(classes_this_segment))
    # print(num_showers_this_segment, len(np.unique(labels)))

    classes_all_filtered = classes_this_segment[is_spectator==1]


    predicted_showers_this_segment = len(np.unique(labels))
    num_predicted_showers.append(predicted_showers_this_segment)

    num_real_showers.append(num_showers_this_segment)


    unique_labels = np.unique(labels)
    unique_classes = np.unique(classes_all_filtered)


    found = dict()
    for x in np.unique(classes_this_segment):
        found[x]=False


    found_predicted = dict()
    for x in np.unique(unique_labels):
        found_predicted[x]=False

    num_fakes = 0
    num_found = 0
    num_missed = 0

    for x in unique_classes:
        classes = classes_all_filtered[classes_all_filtered==x]
        betas = beta_all_filtered[classes_all_filtered==x]


        this_class = classes[np.argmax(betas)]
        label_max = labels[classes_all_filtered==x][np.argmax(betas)]

        if found[this_class]==True:
            pass
        else:
            found[this_class]=True
            found_predicted[label_max]=True
            num_found+=1

    for x in unique_labels:
        # labels_x = labels[labels==x]
        # betas = beta_all_filtered[labels==x]
        #
        # this_class = classes[np.argmax(betas)]

        if found_predicted[x]==True:
            pass
        else:
            num_fakes+=1


    for k,v in found.items():
        if v==False:
            num_missed += 1

    # print(num_found, num_missed, num_fakes)

    num_found_g.append(num_found)
    num_missed_g.append(num_missed)
    num_fakes_g.append(num_fakes)

    print(num_showers_this_segment, num_found, num_missed, num_fakes, predicted_showers_this_segment)

    return




for i in range(1000):
    try:
        print("%d"%i)
        batch = next(gen)
    except:
        break


    result,_ = model2.predict_on_batch(batch[0])



    row_splits = batch[0][1][:, 0]
    x_data, _ = ragged_constructor((batch[0][0], row_splits))
    y_data, _ = ragged_constructor((batch[1][0], row_splits))
    classes, row_splits = ragged_constructor((batch[1][0][:, 0][..., tf.newaxis], row_splits))

    classes = tf.cast(classes[:,0], tf.int32)

    num_unique = []
    shower_sizes = []

    for i in range(len(row_splits)-1):
        classes_this_segment = classes[row_splits[i]:row_splits[i+1]].numpy()
        x_this_segment = x_data[row_splits[i]:row_splits[i+1]].numpy()
        y_this_segment = y_data[row_splits[i]:row_splits[i+1]].numpy()
        pred_this_segment = result[row_splits[i]:row_splits[i+1]].numpy()

        x_function(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment)



    print("Hello world")
    i += 1

print(num_real_showers, num_predicted_showers)


plt.hist(num_real_showers, bins=np.arange(0,50), histtype='step')
plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
plt.xlabel('Num showers')
plt.ylabel('Frequency')
plt.legend(['Real showers','Predicted showers'])
plt.savefig('myplot.png')


plt.clf()


print(num_found_g)
print(num_missed_g)
print(num_fakes_g)


num_found_g = np.array(num_found_g)
num_missed_g = np.array(num_missed_g)
num_fakes_g = np.array(num_fakes_g)


# plt.hist(num_found_g, bins=30, histtype='step')
# plt.hist(num_missed_g, bins=30, histtype='step')
# plt.hist(num_fakes_g, bins=30, histtype='step')
# plt.hist(num_real_showers, bins=30, histtype='step')
# plt.xlabel('Num showers')
# plt.ylabel('Frequency')
# plt.legend(['Found','Missed', 'Fakes'])
# plt.savefig('myplot2.png')


plt.clf()


bins = np.linspace(0,1,11)
num_real_showers = np.array(num_real_showers, np.float)
# plt.hist(num_found_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
# plt.hist(num_missed_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
# plt.hist(num_fakes_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
# plt.xlabel('(Num found/missed/fakes) / Total showers')
# plt.ylabel('Frequency')
# plt.legend(['Found','Missed', 'Fakes'])
# plt.savefig('myplot3.png')



x_num_real = []
x_fraction_found = []
x_fraction_missed = []
x_fraction_fakes = []


y_fraction_found = []
y_fraction_missed = []
y_fraction_fakes = []


f_found_g = num_found_g/num_real_showers
f_missed_g = num_missed_g/num_real_showers
f_fakes_g = num_fakes_g/num_real_showers

for i in np.unique(num_real_showers):
    if i<=0:
        continue

    print("XYZ", i, np.mean(f_found_g[num_real_showers==i]))
    print("ABC", i, np.mean(f_missed_g[num_real_showers==i]))
    print("DEF", i, np.mean(f_fakes_g[num_real_showers==i]))

    x_num_real.append(i)
    x_fraction_found.append(np.mean(f_found_g[num_real_showers==i]))
    x_fraction_missed.append(np.mean(f_missed_g[num_real_showers==i]))
    x_fraction_fakes.append(np.mean(f_fakes_g[num_real_showers==i]))


    y_fraction_found.append(np.var(f_found_g[num_real_showers==i]))
    y_fraction_missed.append(np.var(f_missed_g[num_real_showers==i]))
    y_fraction_fakes.append(np.var(f_fakes_g[num_real_showers==i]))

x_num_real = np.array(x_num_real)
x_fraction_found = np.array(x_fraction_found)
x_fraction_missed = np.array(x_fraction_missed)
x_fraction_fakes = np.array(x_fraction_fakes)

y_fraction_found = np.array(y_fraction_found)
y_fraction_missed = np.array(y_fraction_missed)
y_fraction_fakes = np.array(y_fraction_fakes)


order = np.argsort(x_num_real)
x_num_real = x_num_real[order]

x_fraction_found = x_fraction_found[order]
x_fraction_missed = x_fraction_missed[order]
x_fraction_fakes = x_fraction_fakes[order]


y_fraction_found = y_fraction_found[order]
y_fraction_missed = y_fraction_missed[order]
y_fraction_fakes = y_fraction_fakes[order]


print(x_fraction_found, x_fraction_missed, x_fraction_fakes)
print(y_fraction_found, y_fraction_missed, y_fraction_fakes)



plt.clf()
plt.plot(x_num_real, x_fraction_found, linewidth=0.7, marker='o')
plt.plot(x_num_real, x_fraction_missed, linewidth=0.7, marker='o')
plt.plot(x_num_real, x_fraction_fakes, linewidth=0.7, marker='o')
plt.xlabel('Num showers')
plt.ylabel('Fraction (mean)')
plt.legend(['Found','Missed', 'Fakes'])
plt.savefig('myplot4.png')

plt.clf()

plt.plot(x_num_real, y_fraction_found, linewidth=0.7, marker='o')
plt.plot(x_num_real, y_fraction_missed, linewidth=0.7, marker='o')
plt.plot(x_num_real, y_fraction_fakes, linewidth=0.7, marker='o')
plt.xlabel('Num showers')
plt.ylabel('Fraction (variance)')
plt.legend(['Found','Missed', 'Fakes'])
plt.savefig('myplot5.png')