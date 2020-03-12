import numpy as np
import tensorflow as tf
from object_condensation import remove_zero_length_elements_from_ragged_tensors
from LayersRagged import RaggedEdgeConvLayer, RaggedConstructTensor


class DgcnnModelAlpha(tf.keras.models.Model):
    def __init__(self, clustering_space_dimensions=2, same_variable_for_beta_and_clustering_space=False):
        super(DgcnnModelAlpha, self).__init__(dynamic=True)
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

        self.same_variable_for_beta_and_clustering_space = same_variable_for_beta_and_clustering_space


    def call(self, inputs, rowsplits):
        inputs, rowsplits = self.ragged_constructor((inputs, rowsplits))
        rowsplits = remove_zero_length_elements_from_ragged_tensors(rowsplits)


        x = inputs
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.edge_conv_1((x, rowsplits))
        x = self.edge_conv_2((x, rowsplits))
        x = self.edge_conv_3((x, rowsplits))
        x = self.edge_conv_4((x, rowsplits))


        x = self.dense3(x)
        x = self.dense4(x)
        c = self.dense_clustering_space(x)
        b = self.dense_beta(x)

        if self.same_variable_for_beta_and_clustering_space:
            ret = tf.concat((b,c), axis=-1)

            return ret

        return c, tf.squeeze(b, axis=-1)


