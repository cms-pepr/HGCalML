import numpy as np
import tensorflow as tf
from object_condensation import remove_zero_length_elements_from_ragged_tensors
from LayersRagged import RaggedEdgeConvLayer, RaggedConstructTensor



class DgcnnModelBeta(tf.keras.models.Model):
    def __init__(self, clustering_space_dimensions=2, training=True):
        super(DgcnnModelBeta, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)

        self.edge_conv_1 = RaggedEdgeConvLayer(30)
        self.edge_conv_2= RaggedEdgeConvLayer(30)
        self.edge_conv_3= RaggedEdgeConvLayer(30)
        self.edge_conv_4= RaggedEdgeConvLayer(30)


        self.dense3 = tf.keras.layers.Dense(96, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(96, activation=tf.nn.relu)

        self.dense_beta = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dense_clustering_space = tf.keras.layers.Dense(clustering_space_dimensions, activation=None)
        self.ragged_constructor = RaggedConstructTensor()

        self.training = training



    def call(self, inputs, rowsplits):

        inputs, rowsplits = self.ragged_constructor((inputs, rowsplits))
        rowsplits = remove_zero_length_elements_from_ragged_tensors(rowsplits)

        x = inputs
        x = self.bn1(x, training=self.training)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.edge_conv_1((x, rowsplits))
        x = self.edge_conv_2((x, rowsplits))


        x = self.bn2(x, training=self.training)

        x = self.edge_conv_3((x, rowsplits))
        x = self.edge_conv_4((x, rowsplits))
        x = self.bn3(x, training=self.training)


        x = self.dense3(x)
        x = self.dense4(x)

        return self.dense_clustering_space(x), tf.squeeze(self.dense_beta(x), axis=-1)


