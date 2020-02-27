import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor, RaggedGlobalExchange
from object_condensation import remove_zero_length_elements_from_ragged_tensors


class OverfittingKing(tf.keras.models.Model):
    def __init__(self, clustering_dims=2):
        super(OverfittingKing, self).__init__()

        n_units = 768
        self.layer_1 = tf.keras.layers.Dense(n_units, input_dim=2, activation='relu')
        self.layer_2 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_3 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_4 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_5 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_output_clustering_space = tf.keras.layers.Dense(clustering_dims, input_dim=768, activation=None)
        self.layer_output_beta = tf.keras.layers.Dense(1, input_dim=768, activation=None)

    def call(self, input_data, row_splits):
        x = self.layer_1(input_data)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x1 = self.layer_output_clustering_space(x)
        x2 = tf.nn.sigmoid(self.layer_output_beta(x))

        return x1, tf.squeeze(x2, axis=-1)

