import tensorflow as tf
import tensorflow.keras as keras
from rknn_op import rknn_ragged, rknn_op
from caloGraphNN import gauss_of_lin


class RaggedConstructTensor(keras.layers.Layer):
    """
    This layer is used to construct ragged tensor from data and row_splits. data and row_splits have 2 and 1 axis
    respectively. They have same dimensions along the first dimension (for cater to keras limitations of ragged
    tensors). The second axis of the data tensor contains features. The last value of the row_split vector must be
    (batch size + 1). Since we cannot yet return RaggedTensor(s) from layers, we return data and row_splits in a format
    compatible with tf.RaggedTensor.from_row_splits for ease. We'll probably not need this when keras starts to support
    RaggedTensors.

    """


    def __init__(self, **kwargs):
        super(RaggedConstructTensor, self).__init__(**kwargs)
        self.num_features = -1

    def build(self, input_shape):
        super(RaggedConstructTensor, self).build(input_shape)

    def call(self, x):
        x_data = x[0]
        x_row_splits = x[1]

        data_shape = x_data.shape
        # assert (data_shape[0]== row_splits_shape[0])
        self.num_features = data_shape[1]

        if len(x_row_splits.shape) == 2:
            x_row_splits = x_row_splits[:,0]

        row_splits = tf.reshape(x_row_splits, (-1,))
        batch_size_plus_1 = tf.cast(row_splits[-1], tf.int32)
        row_splits = tf.slice(row_splits, [0], batch_size_plus_1[..., tf.newaxis])

        num_elements = tf.cast(row_splits[-1], tf.int32)
        data = tf.slice(x_data, [0, 0], [num_elements, self.num_features])

        return data, row_splits

    def compute_output_shape(self, input_shape):
        return [(None, self.num_features), (None,)]


class RaggedGravNet(keras.layers.Layer):
    def __init__(self, n_neighbours=40, n_dimensions=4, n_filters=32, n_propagate=8, 
                 also_coordinates=False, feature_dropout=-1,
                 coordinate_kernel_initializer=keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False,
                 coordinate_activation=None,
                 additional_message_passing=0,
                 **kwargs):
        super(RaggedGravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        # self.name = name TODO: Jan this doesnt work for me
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.additional_message_passing = additional_message_passing

        self.input_feature_transform = keras.layers.Dense(n_propagate, name=self.name + '_FLR',
                                                          kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = keras.layers.Dense(n_dimensions, name=self.name + '_S',
                                                          kernel_initializer=coordinate_kernel_initializer,
                                                          activation=coordinate_activation)
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh', name=self.name + '_Fout',
                                                           kernel_initializer=other_kernel_initializer)

        self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [self.input_feature_transform, self.output_feature_transform]

        self.message_passing_layers = []
        for i in range(additional_message_passing):
            self.message_passing_layers.append(
                keras.layers.Dense(n_propagate, activation='elu',
                                   name=self.name + '_mp_' + str(i), kernel_initializer=other_kernel_initializer)
            )
        self._sublayers += self.message_passing_layers

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)

        self.output_feature_transform.build((input_shape[0], input_shape[1] + self.input_feature_transform.units * 2))

        self.message_parsing_distance_weights = []
        for i in range(len(self.message_passing_layers)):
            l = self.message_passing_layers[i]
            l.build((input_shape[0], input_shape[1] + self.n_propagate * 2))
            d_weight = self.add_weight(name=self.name + '_mp_distance_weight_' + str(i),
                                       shape=(1,),
                                       initializer='uniform',
                                       constraint=keras.constraints.NonNeg(),
                                       trainable=True)
            self.message_parsing_distance_weights.append(d_weight)

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(RaggedGravNet, self).build(input_shape)

    def call(self, inputs):

        x = inputs[0]
        row_splits = inputs[1]

        mask = None

        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:, :, 0:self.n_dimensions]

        collected_neighbours = self.collect_neighbours(coordinates, x, row_splits)

        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)


        if self.also_coordinates:
            return [output, coordinates]
        return output

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]

        if self.also_coordinates:
            return [(input_shape[0], self.output_feature_transform.units),
                    (input_shape[0], self.n_dimensions)]

        return (input_shape[0], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, x, row_splits):
        features = self.input_feature_transform(x) # [SV, F]

        ragged_split_added_indices, _ = rknn_op.RaggedKnn(num_neighbors=int(self.n_neighbours), row_splits=row_splits, data=coordinates, add_splits=True) # [SV, N+1]
        ragged_split_added_indices = ragged_split_added_indices[:,1:][..., tf.newaxis]  # [SV, N]

        distance = tf.reduce_sum((coordinates[:, tf.newaxis, :] - tf.gather_nd(coordinates, ragged_split_added_indices))**2, axis=-1)  # [SV, N]

        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)  # [SV, N, 1]

        for i in range(len(self.message_passing_layers) + 1):
            if i:
                features = self.message_passing_layers[i - 1](tf.concat([features, x], axis=-1))  # [SV, G]
                w = self.message_parsing_distance_weights[i - 1]  # [??]
                weights = gauss_of_lin(w * distance) # [SV, N]
                weights = tf.expand_dims(weights, axis=-1) # [SV, N, 1]

            if self.feature_dropout > 0 and self.feature_dropout < 1:
                features = keras.layers.Dropout(self.feature_dropout)(features)

            neighbour_features = tf.gather_nd(features, ragged_split_added_indices) # [SV, N, F]
            # weight the neighbour_features
            neighbour_features *= weights  # [SV, N, F]

            neighbours_max = tf.reduce_max(neighbour_features, axis=1)  # [SV, F]
            neighbours_mean = tf.reduce_mean(neighbour_features, axis=1)  # [SV, F]

            features = tf.concat([neighbours_max, neighbours_mean], axis=-1)  # [SV, 2F]

        return features

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours,
                  'n_dimensions': self.n_dimensions,
                  'n_filters': self.n_filters,
                  'n_propagate': self.n_propagate,
                  'name': self.name,
                  'also_coordinates': self.also_coordinates,
                  'feature_dropout': self.feature_dropout,
                  'additional_message_passing': self.additional_message_passing}
        base_config = super(RaggedGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class RaggedGravNet_simple(tf.keras.layers.Layer):
    def __init__(self, 
                 n_neighbours, 
                 n_dimensions, 
                 n_filters, 
                 n_propagate,**kwargs):
        super(RaggedGravNet_simple, self).__init__(**kwargs)
        
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        
        self.input_feature_transform = tf.keras.layers.Dense(n_propagate)
        self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions)
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh')

    def build(self, input_shapes):
        
        input_shape = input_shapes[0]
        self.input_feature_transform.build(input_shape)
        self.input_spatial_transform.build(input_shape)
        
        self.output_feature_transform.build((input_shape[0],  
                                             input_shape[1] + self.input_feature_transform.units * 2))
 
        super(RaggedGravNet_simple, self).build(input_shape)
        
    def call(self, inputs):
        
        x = inputs[0]
        row_splits = inputs[1]
        
        coordinates = self.input_spatial_transform(x)
        features = self.input_feature_transform(x)
        collected_neighbours = self.collect_neighbours(coordinates, features, row_splits)
        
        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        return self.output_feature_transform(updated_features)
    

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], self.output_feature_transform.units)
    
    def collect_neighbours(self, coordinates, features, row_splits):
        
        ragged_split_added_indices, _ = rknn_op.RaggedKnn(num_neighbors=int(self.n_neighbours), row_splits=row_splits, data=coordinates, add_splits=True) # [SV, N+1]
        
        print(ragged_split_added_indices)
        ragged_split_added_indices = ragged_split_added_indices[:,1:][..., tf.newaxis]  # [SV, N]

        distance = tf.reduce_sum((coordinates[:, tf.newaxis, :] - tf.gather_nd(coordinates, ragged_split_added_indices))**2, axis=-1)  # [SV, N]

        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)  # [SV, N, 1]
        
        neighbour_features = tf.gather_nd(features, ragged_split_added_indices)
        neighbour_features *= weights
        neighbours_max  = tf.reduce_max(neighbour_features, axis=1)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=1)
        
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)
    
    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate}
        base_config = super(RaggedGravNet_simple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class RaggedGlobalExchange(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RaggedGlobalExchange, self).__init__(**kwargs)
        self.num_features = -1

    def build(self, input_shape):
        data_shape = input_shape[0]
        row_splits_shape = input_shape[1]
        # assert (data_shape[0]== row_splits_shape[0])
        self.num_features = data_shape[1]
        super(RaggedGlobalExchange, self).build(input_shape)

    def call(self, x):
        x_data, x_row_splits = x[0], x[1]
        rt = tf.RaggedTensor.from_row_splits(values=x_data, row_splits=x_row_splits)  # [B, {V}, F]
        means = tf.reduce_mean(rt, axis=1)  # [B, F]
        data_means = tf.gather_nd(means, tf.ragged.row_splits_to_segment_ids(rt.row_splits)[..., tf.newaxis])  # [SV, F]

        return tf.concat((data_means, x_data), axis=-1)

    def compute_output_shape(self, input_shape):
        data_input_shape = input_shape[0]
        return None, data_input_shape[1]*2



class RaggedEdgeConvLayer(keras.layers.Layer):

    def __init__(self, num_neighbors=30,
                          mpl_layers=[64, 64, 64],
                          aggregation_function=tf.reduce_max, edge_activation=None, **kwargs):
        super(RaggedEdgeConvLayer, self).__init__(**kwargs)

        self.num_neighbors = num_neighbors
        self.aggregation_function = aggregation_function
        self.edge_activation=edge_activation


        dense_layers = []
        for f in mpl_layers:
            dense_layers+= [tf.keras.layers.Dense(f, activation=tf.nn.relu)]

        self.mpl_layers = dense_layers



    def compute_output_shape(self, input_shape):
        data_input_shape = input_shape[0]
        return (data_input_shape[0], self.mpl_layers[-1])



    def call(self, x):

        vertices_in, rowsplits = x
        rowsplits = tf.cast(rowsplits, tf.int32)

        ragged_split_added_indices, _ = rknn_op.RaggedKnn(num_neighbors=int(self.num_neighbors+1), row_splits=rowsplits,
                                                          data=vertices_in, add_splits=True)  # [SV, N+1]
        ragged_split_added_indices = ragged_split_added_indices[:, 1:][..., tf.newaxis]  # [SV, N]

        neighbor_space = tf.gather_nd(vertices_in, ragged_split_added_indices)


        expanded_trans_space = tf.expand_dims(vertices_in, axis=1)
        expanded_trans_space = tf.tile(expanded_trans_space, [1, self.num_neighbors, 1])
        diff = expanded_trans_space - neighbor_space
        edge = tf.concat([expanded_trans_space, diff], axis=-1)


        for f in self.mpl_layers:
            edge = f(edge)

        if self.edge_activation is not None:
            edge = self.edge_activation(edge)



        vertex_out = self.aggregation_function(edge, axis=1)

        return vertex_out

        0/0

        print(vertices_in.shape)
        0/0
        trans_space = vertices_in
        indexing, _ = indexing_tensor(trans_space, self.num_neighbors)
        # change indexing to be not self-referential
        neighbour_space = tf.gather_nd(vertices_in, indexing)

        expanded_trans_space = tf.expand_dims(trans_space, axis=2)
        expanded_trans_space = tf.tile(expanded_trans_space, [1, 1, self.num_neighbors, 1])

        diff = expanded_trans_space - neighbour_space
        edge = tf.concat([expanded_trans_space, diff], axis=-1)

        for f in self.mpl_layers:
            edge = tf.layers.dense(edge, f, activation=tf.nn.relu)

        if self.edge_activation is not None:
            edge = self.edge_activation(edge)

        vertex_out = self.aggregation_function(edge, axis=2)

        return vertex_out

