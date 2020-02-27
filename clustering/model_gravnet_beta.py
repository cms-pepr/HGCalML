import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor, RaggedGlobalExchange
from object_condensation import remove_zero_length_elements_from_ragged_tensors


class GravnetModelBeta(tf.keras.models.Model):
    def __init__(self, clustering_space_dimensions=2):
        super(GravnetModelBeta, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)


        self.glayer_1 = RaggedGravNet(n_filters=64, n_propagate=32)
        self.glayer_2 = RaggedGravNet(n_filters=64, n_propagate=32)
        self.glayer_3 = RaggedGravNet(n_filters=64, n_propagate=32)
        self.glayer_4 = RaggedGravNet(n_filters=64, n_propagate=32)
        self.glayer_5 = RaggedGravNet(n_filters=64, n_propagate=32)


        self.gx1 = RaggedGlobalExchange()
        self.gx2 = RaggedGlobalExchange()

        self.ragged_constructor = RaggedConstructTensor()
        self.clustering_space_dimensions = clustering_space_dimensions

        self.output_dense_clustering = tf.keras.layers.Dense(clustering_space_dimensions, activation=None)
        self.output_dense_beta = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        # self.output_dense_3 = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, rowsplits):


        inputs, rowsplits = self.ragged_constructor((inputs, rowsplits))
        rowsplits = remove_zero_length_elements_from_ragged_tensors(rowsplits)

        rowsplits = tf.cast(rowsplits, tf.int32)

        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.glayer_1((x, rowsplits))

        a = x = self.glayer_2((x, rowsplits))
        x = self.gx1((x, rowsplits))
        x = self.dense3(x)
        x = self.glayer_3((x, rowsplits))
        b = x = self.glayer_4((x, rowsplits))
        x = self.gx2((x, rowsplits))
        x = self.glayer_5((x, rowsplits))

        x = tf.concat((a,b,x), axis=-1)

        x = self.dense4(x)
        x = self.dense5(x)

        return self.output_dense_clustering(x), tf.squeeze(self.output_dense_beta(x), axis=-1)

