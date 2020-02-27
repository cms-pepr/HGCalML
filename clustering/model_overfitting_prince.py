import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor, RaggedGlobalExchange
from object_condensation import remove_zero_length_elements_from_ragged_tensors


class OverfittingPrince(tf.keras.models.Model):
    def __init__(self, ):
        super(OverfittingPrince, self).__init__()

        n_units = 256

        self.gr1 = tf.keras.layers.Conv1D(n_units, 10, padding='same')
        self.gr2 = tf.keras.layers.Conv1D(n_units, 10, padding='same')
        self.gr3 = tf.keras.layers.Conv1D(n_units, 10, padding='same')
        self.gr4 = tf.keras.layers.Conv1D(n_units, 10, padding='same')

        self.d1 = tf.keras.layers.Dense(768, activation='relu')
        self.d2 = tf.keras.layers.Dense(768, activation='relu')


        self.layer_output_clustering_space = tf.keras.layers.Dense(2, activation=None)
        self.layer_output_beta = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data, row_splits):

        x = input_data[tf.newaxis, ...]

        x = self.gr1(x)
        x = self.gr2(x)
        x = self.gr3(x)
        x = self.gr4(x)

        x = x[0,:,:]
        x = self.d1(x)
        x = self.d2(x)

        x1 = self.layer_output_clustering_space(x)
        x2 = tf.nn.sigmoid(self.layer_output_beta(x))

        return x1, tf.squeeze(x2, axis=-1)

