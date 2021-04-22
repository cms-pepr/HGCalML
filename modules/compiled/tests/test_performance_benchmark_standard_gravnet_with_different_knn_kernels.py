from __future__ import print_function
import tensorflow as tf
# from K import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from LayersRagged import RaggedGlobalExchange
from GravNetLayersRagged import RaggedGravNet
from tensorflow.keras.layers import Dense, Concatenate, GaussianDropout
from DeepJetCore.modeltools import DJCKerasModel
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model
import time
from tensorflow.keras import backend as K
# from tensorflow.keras.utils.layer_utils import count_params
from DeepJetCore.modeltools import fixLayersContaining
# from tensorflow.keras.models import load_model
from DeepJetCore.training.training_base import custom_objects_list
# from tensorflow.keras.optimizer_v2 import Adam
from ragged_callbacks import plotEventDuringTraining
from ragged_callbacks import plotRunningPerformanceMetrics
from DeepJetCore.DJCLayers import ScalarMultiply, SelectFeatures, ReduceSumEntirely
from clr_callback import CyclicLR
from Layers import ExpMinusOne, GridMaxPoolReduction
from model_blocks import  create_output_layers
from lossLayers import LLObjectCondensation
from index_dicts import create_ragged_cal_feature_dict, create_ragged_cal_truth_dict, create_ragged_cal_pred_dict
from pre2_knn_op import Pre2Knn

gpus_to_use = "1" # "comma-separated string"
from DeepJetCore.training.gpuTools import DJCSetGPUs
DJCSetGPUs(gpus_to_use)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class GravNetModel(tf.keras.Model):
    def __init__(self):
        super(GravNetModel, self).__init__()
        self.gx = RaggedGlobalExchange(name="global_exchange")
        self.dense1 = Dense(64, activation='elu', name="dense_start")
        self.n_filters = 0
        self.n_gravnet_layers = 4
        self.n_filters = 196
        self.n_propagate = 128
        self.n_propagate_2 = [64, 32, 16, 8, 4, 2]
        self.n_neighbours = 64
        self.n_dim = 8
        blocks = []
        for i in range(self.n_gravnet_layers):
            block = dict()
            block['gravnet'] = RaggedGravNet(n_neighbours=self.n_neighbours,
                                              n_dimensions=self.n_dim,
                                              n_filters=self.n_filters,
                                              n_propagate=self.n_propagate,
                                              name='gravnet_' + str(i))
            # block['gravnet'] = Dense(self.n_filters, activation='elu')
            # block['message_passing'] = DistanceWeightedMessagePassing(self.n_propagate_2)
            # block['bn1'] = BatchNormalization(momentum=0.6)
            block['dn1'] = Dense(128, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn2'] = BatchNormalization(momentum=0.6)
            block['dn2'] = Dense(96, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn3'] = BatchNormalization(momentum=0.6)
            block['dn3'] = Dense(96, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn4'] = BatchNormalization(momentum=0.6)
            blocks.append(block)
        self.blocks = blocks
        self.odn1 = Dense(128, activation='relu')
        # self.obn1 = BatchNormalization(momentum=0.6)
        self.odn2 = Dense(128, activation='elu')
        # self.obn2 = BatchNormalization(momentum=0.6)
        #
        self.odn3 = Dense(128, activation='elu')
        # self.obn3 = BatchNormalization(momentum=0.6)
        self.odn4 = Dense(128, activation='relu')
        self.odn5 = Dense(128, activation='relu')
    def call(self, inputs):
        feat, row_splits = inputs
        x_data, x_row_splits = feat, row_splits #self.constructor([feat, row_splits])
        # x_basic = self.bn1(x_data)  # mask_and_norm is just batch norm now
        x_basic = x_data
        x = x_basic
        #  print(x.shape)
        x = self.gx([x, x_row_splits])
        #  print(x.shape)
        x = self.dense1(x)
        #  print(x.shape)
        feat = [x_basic, x]
        for i in range(self.n_gravnet_layers):
            # if n_dim < 2:
            #    n_dim = 2
            block = self.blocks[i]
            x, coords, neighbor_indices, neighbor_dist = block['gravnet']([x, x_row_splits])
            # x = block['gravnet']([x, x_row_splits])
            # x = block['message_passing']([x, neighbor_indices, neighbor_dist])
            # x = block['bn1'](x)
            x = block['dn1'](x)
            # x = block['bn2'](x)
            x = block['dn2'](x)
            x = self.gx([x, x_row_splits])
            x = block['dn3'](x)
            # x = block['bn4'](x)
            feat.append(x)
        x = tf.concat(feat, axis=-1)
        #  print(x.shape)
        x = self.odn1(x)
        # x = self.obn1(x)
        x = self.odn2(x)
        # x = self.obn2(x)
        x = self.odn3(x)
        # x = self.obn3(x)
        x = self.odn4(x)
        x = self.odn5(x)
        return x


class GravNetModel_mod(tf.keras.Model):
    def __init__(self):
        super(GravNetModel_mod, self).__init__()
        self.gx = RaggedGlobalExchange(name="global_exchange")
        self.dense1 = Dense(64, activation='elu', name="dense_start")
        self.n_filters = 0
        self.n_gravnet_layers = 4
        self.n_filters = 196
        self.n_propagate = 128
        self.n_propagate_2 = [64, 32, 16, 8, 4, 2]
        self.n_neighbours = 64
        self.n_dim = 8
        blocks = []
        for i in range(self.n_gravnet_layers):
            block = dict()
            block['gravnet'] = RaggedGravNet(n_neighbours=self.n_neighbours,
                                              n_dimensions=self.n_dim,
                                              n_filters=self.n_filters,
                                              n_propagate=self.n_propagate,
                                              name='gravnet_' + str(i))
            # block['gravnet'] = Dense(self.n_filters, activation='elu')
            # block['message_passing'] = DistanceWeightedMessagePassing(self.n_propagate_2)
            # block['bn1'] = BatchNormalization(momentum=0.6)
            block['dn1'] = Dense(128, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn2'] = BatchNormalization(momentum=0.6)
            block['dn2'] = Dense(96, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn3'] = BatchNormalization(momentum=0.6)
            block['dn3'] = Dense(96, activation='relu', name="dense_bottom_" + str(i) + "_a")
            # block['bn4'] = BatchNormalization(momentum=0.6)
            blocks.append(block)
        self.blocks = blocks
        self.odn1 = Dense(128, activation='relu')
        # self.obn1 = BatchNormalization(momentum=0.6)
        self.odn2 = Dense(128, activation='elu')
        # self.obn2 = BatchNormalization(momentum=0.6)
        #
        self.odn3 = Dense(128, activation='elu')
        # self.obn3 = BatchNormalization(momentum=0.6)
        self.odn4 = Dense(128, activation='relu')
        self.odn5 = Dense(128, activation='relu')
    def call(self, inputs):
        feat, row_splits, test_arr = inputs
        x_data, x_row_splits = feat, row_splits #self.constructor([feat, row_splits])
        # x_basic = self.bn1(x_data)  # mask_and_norm is just batch norm now
        x_basic = x_data
        x = x_basic
        #  print(x.shape)
        x = self.gx([x, x_row_splits])
        #  print(x.shape)
        x = self.dense1(x)
        #  print(x.shape)
        feat = [x_basic, x]
        for i in range(self.n_gravnet_layers):
            # if n_dim < 2:
            #    n_dim = 2
            block = self.blocks[i]
            x, coords, neighbor_indices, neighbor_dist = block['gravnet']([x, x_row_splits, test_arr])
            # x = block['gravnet']([x, x_row_splits])
            # x = block['message_passing']([x, neighbor_indices, neighbor_dist])
            # x = block['bn1'](x)
            x = block['dn1'](x)
            # x = block['bn2'](x)
            x = block['dn2'](x)
            x = self.gx([x, x_row_splits])
            x = block['dn3'](x)
            # x = block['bn4'](x)
            feat.append(x)
        x = tf.concat(feat, axis=-1)
        #  print(x.shape)
        x = self.odn1(x)
        # x = self.obn1(x)
        x = self.odn2(x)
        # x = self.obn2(x)
        x = self.odn3(x)
        # x = self.obn3(x)
        x = self.odn4(x)
        x = self.odn5(x)
        return x


model = GravNetModel_mod()

n_loops = 10
nv = 60000
n_dims = 10
sorting_time = np.zeros(n_loops)
total_forward_prop_time = np.zeros(n_loops)

for i in range(n_loops):
    X = np.random.random((nv,n_dims))
    R = np.array([0,nv], np.int32)
    N_BINS_X = 10
    N_BINS_Y = 10

    target = np.random.random((nv, 128))

    t1 = time.time()
    t_global = time.time()

    sorted_coords, auxaliry_knn_arrays = Pre2Knn(X, N_BINS_X, N_BINS_Y)
    sorting_time[i]=(time.time()-t1)
    print("\nCUDA Coordinate sorting: ", sorting_time[i],"seconds")

    t1 = time.time()
    with tf.GradientTape() as tape:
        Y = model((sorted_coords,R,auxaliry_knn_arrays))
        loss = tf.nn.l2_loss(Y, target)
    total_forward_prop_time[i] = (time.time()-t_global)
    print("Forward propagation only: ", time.time()-t1,"seconds")

    print("Forward propagation + sort: ", total_forward_prop_time[i],"seconds")


model = GravNetModel()
time_default_knn = np.zeros(n_loops)

for i in range(n_loops):
    X = np.random.random((nv,n_dims))
    R = np.array([0,nv], np.int32)

    target = np.random.random((nv, 128))

    t1 = time.time()
    with tf.GradientTape() as tape:
        Y = model((X,R))
        loss = tf.nn.l2_loss(Y, target)
    time_default_knn[i] = (time.time()-t1)
    print("Forward propagation only: ", time.time()-t1,"seconds")

sorting_time = sorting_time[int(n_loops/5):]
total_forward_prop_time = total_forward_prop_time[int(n_loops/5):]

print("Sorting time average: ",sorting_time.sum()/len(sorting_time))
print("Total forward prop time average (slicing kNN): ",total_forward_prop_time.sum()/len(total_forward_prop_time))

time_default_knn = time_default_knn[int(n_loops/5):]
print("Total forward prop time average (default kNN): ",time_default_knn.sum()/len(time_default_knn))
