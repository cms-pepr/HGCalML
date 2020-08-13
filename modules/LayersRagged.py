import tensorflow as tf
import tensorflow.keras as keras
from rknn_op import rknn_ragged, rknn_op
from caloGraphNN import gauss_of_lin
import uuid
from select_knn_op import SelectKnn
from accknn_op import AccumulateKnn
from condensate_op import BuildCondensates
from pseudo_rs_op import CreatePseudoRS
from select_threshold_op import SelectThreshold

from latent_space_grid_op import LatentSpaceGrid
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D


class RaggedSumAndScatter(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(RaggedSumAndScatter, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shapes): # data, rs, scat
        return input_shapes[0][-1]
    
    def call(self, inputs):
        data, rs, indices = inputs[0], inputs[1], inputs[2]
        ragged = tf.RaggedTensor.from_row_splits(values=data,
              row_splits=rs)
        ragged = tf.reduce_mean(ragged,axis=1)
        gath = tf.gather_nd(ragged, indices) 
        return tf.reshape(gath, tf.shape(data))#so that shape is known to keras
        

class CondensateToPseudoRS(keras.layers.Layer):
    
    def __init__(self, radius, threshold, soft,**kwargs):
        self.radius=radius
        self.threshold=threshold
        self.soft=soft
        super(CondensateToPseudoRS, self).__init__(**kwargs)
        
    def get_config(self):
            config = {'radius': self.radius,
                      'threshold': self.threshold,
                      'soft': self.soft}
            base_config = super(CondensateToPseudoRS, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes): #input is data, coords, beta, rs
        outshapes = [input_shapes[0][-1],None , 1, None]
        print('outshapes',outshapes)
        return outshapes #returns reshuffled data, new rs, indices to go back   

    def call(self, inputs):
        data, ccoords, betas, row_splits = inputs[0],inputs[1],inputs[2],inputs[3],
        asso_idx, is_cpoint,n = BuildCondensates(ccoords, betas, row_splits, 
                                                 radius=self.radius, min_beta=self.threshold, soft=self.soft)
        sids, psrs, sdata, belongs_to_prs = CreatePseudoRS(asso_idx,data)
        sdata = tf.reshape(sdata, tf.shape(data) )#same shape
        return sdata, psrs, sids, asso_idx, belongs_to_prs
    

        

class GridMaxPoolReduction(keras.layers.Layer):
    
    def __init__(self, gridsize = 1., depth = 1, cnn_kernels=None,**kwargs):
        super(GridMaxPoolReduction, self).__init__(**kwargs)
        self.gridsize=gridsize
        self.depth=depth
        self.cnn_kernels=cnn_kernels
        
        assert depth == 1 #for now

    def get_config(self):
            config = {'gridsize': self.gridsize,
                      'depth': self.depth,
                      'cnn_kernels': self.cnn_kernels}
            base_config = super(GridMaxPoolReduction, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes): #input is coords, feat, rs
        return (input_shapes[1][1]*self.depth,) #returns only features x depth

    def call(self, x):
        coords,feat,rs = x[0], x[1], x[2]
        
        assert coords.shape[-1] < 4 and coords.shape[-1] > 1
        
        poolop=None #not yet
        
        idxs,psrs,ncc,backgather = LatentSpaceGrid(size=1., 
                                       min_cells = self.depth**2,
                                       coords = coords,
                                       row_splits=rs )
        
        #tf.print('ncc',ncc)
        resorted_feat = tf.gather_nd(feat, tf.expand_dims(idxs,axis=1))
        pseudo_ragged_feat = tf.RaggedTensor.from_row_splits(
              values=resorted_feat,
              row_splits=psrs)
        
        pseudo_ragged_feat = tf.reduce_max(pseudo_ragged_feat, axis=1)
        pseudo_ragged_feat = tf.where(tf.abs(pseudo_ragged_feat)>1e10,0.,pseudo_ragged_feat)

        back_sorted_max_feat = tf.gather_nd(pseudo_ragged_feat, tf.expand_dims(backgather,axis=1))
        return tf.reshape(back_sorted_max_feat, tf.shape(feat))
        


class RaggedSelectThreshold(keras.layers.Layer):
    def __init__(self,  threshold  = 0.5, **kwargs):
        self.threshold=threshold
        super(RaggedSelectThreshold, self).__init__(**kwargs)
    
    
    def get_config(self):
            config = {'threshold': self.threshold}
            base_config = super(RaggedSelectThreshold, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shapes):
        #in: th, feat, rs
        #out: feat, rs, scatter_idxs
        #print('input_shapes',input_shapes)
        return [input_shapes[1][-1], None, [1]]  
    
    def call(self, x):
        xs,pl,rs = x[0], x[1], x[2]
        newfeat, rs, scatter_idxs = SelectThreshold(xs,pl,rs,threshold=self.threshold)
        return newfeat, rs, scatter_idxs
     
       
    



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




class RaggedGravNet_simple(tf.keras.layers.Layer):
    def __init__(self,
                 n_neighbours: int,
                 n_dimensions: int,
                 n_filters,
                 n_propagate, 
                 **kwargs):
        super(RaggedGravNet_simple, self).__init__(**kwargs)

        n_neighbours += 1 #includes the 'self' vertex
        assert n_neighbours > 1

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        
        
        if isinstance(n_filters, list):
            print('RaggedGravNet_simple.__init__: Warning: n_filters is not a list anymore! taking only the first')
            n_filters=n_filters[0]
        self.n_filters = n_filters
        
        if not isinstance(n_propagate, list):
            n_propagate=[n_propagate]
            
        self.n_propagate = n_propagate
        
        self.n_prop_total = 0
        for p in self.n_propagate:
            self.n_prop_total += 2* p

        self.input_feature_transform=[]
        with tf.name_scope(self.name+"/1/"):
            for i in range(len(n_propagate)):
                self.input_feature_transform.append(tf.keras.layers.Dense(n_propagate[i]))

        with tf.name_scope(self.name+"/2/"):
            self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions)

        with tf.name_scope(self.name+"/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(self.n_filters, activation='tanh')

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform[0].build(input_shape)
            for i in range(1, len(self.n_propagate)):
                with tf.name_scope(self.name + "/1/"+str(i)+"/"):
                    self.input_feature_transform[i].build((input_shape[0],self.n_propagate[i-1]*2))

        with tf.name_scope(self.name + "/2/"):
            self.input_spatial_transform.build(input_shape)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform.build((input_shape[0], self.n_prop_total + input_shape[1]))
            
        

        super(RaggedGravNet_simple, self).build(input_shape)


    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x
        for t in self.input_feature_transform:
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices, distancesq)
            features = tf.reshape(features, [-1, prev_feat.shape[1]*2])
            allfeat.append(features)
            
        features = tf.concat(allfeat +[x], axis=-1)
        return self.output_feature_transform(features)
        
    def priv_call(self, inputs):
        x = inputs[0]
        row_splits = inputs[1]

        coordinates = self.input_spatial_transform(x)
        
        neighbour_indices, distancesq = self.compute_neighbours_and_distancesq(coordinates, row_splits)

        return self.create_output_features(x, neighbour_indices, distancesq), coordinates
    
    def call(self, inputs):
        return self.priv_call(inputs)[0]

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        return (self.output_feature_transform.units[-1],)
    
    def compute_neighbours_and_distancesq(self, coordinates, row_splits):

        ragged_split_added_indices,_ = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=1.0, tf_compatible=True)
        
        ragged_split_added_indices = ragged_split_added_indices[:,1:]
            
        
        ragged_split_added_indices=ragged_split_added_indices[...,tf.newaxis] 

        distancesq = tf.reduce_sum(
            (coordinates[:, tf.newaxis, :] - tf.gather_nd(coordinates,ragged_split_added_indices)) ** 2,
            axis=-1)  # [SV, N]

        return ragged_split_added_indices,distancesq 
        

    def collect_neighbours(self, features, neighbour_indices, distancesq):

        weights = gauss_of_lin(10.*distancesq)
        weights = tf.expand_dims(weights, axis=-1)  # [SV, N, 1]
        neighbour_features = tf.gather_nd(features, neighbour_indices)
        neighbour_features *= weights
        neighbours_max = tf.reduce_max(neighbour_features, axis=1)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=1)

        return tf.concat([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours,
                  'n_dimensions': self.n_dimensions,
                  'n_filters': self.n_filters,
                  'n_propagate': self.n_propagate}
        base_config = super(RaggedGravNet_simple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class FusedRaggedGravNet_simple(RaggedGravNet_simple):
    def __init__(self,
                 **kwargs):
        super(FusedRaggedGravNet_simple, self).__init__(**kwargs)
        
        
    def compute_neighbours_and_distancesq(self, coordinates, row_splits):
        idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=1.0, tf_compatible=False)
        return idx[:,1:], dist[:,1:]
        

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices, n_moments=0)
        return f

class FusedRaggedGravNet(FusedRaggedGravNet_simple):
    '''
    also returns coordinates
    '''
    def __init__(self,
                 **kwargs):
        super(FusedRaggedGravNet, self).__init__(**kwargs)
        
    
    def call(self, inputs):
        x,coords = self.priv_call(inputs) 
        return x,coords   
    
        
    def compute_output_shape(self, input_shapes):
        return (self.output_feature_transform.units[-1],), (self.n_dimensions, )
    
    
    def compute_neighbours_and_distancesq(self, coordinates, row_splits):
        idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=1.0, tf_compatible=False) 
        return idx[:,1:], dist[:,1:]


class FusedRaggedGravNetLinParse(FusedRaggedGravNet):
    '''
    linear parsing
    '''
    def __init__(self,
                 **kwargs):
        super(FusedRaggedGravNetLinParse, self).__init__(**kwargs)
        
    
    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x
        

        for t in self.input_feature_transform:
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices, distancesq)
            features = tf.reshape(features, [-1, prev_feat.shape[1]*2])
            allfeat.append(features)
            distancesq *= 0. #the remaining parsing operations will be at zero distance
            
        features = tf.concat(allfeat +[x], axis=-1)
        return self.output_feature_transform(features)

class FusedRaggedGravNetLinParsePool(FusedRaggedGravNetLinParse):
    def __init__(self,
                 cut_off = 0.5,
                 hardness = 20,
                 **kwargs):
        self.cut_off = cut_off
        self.hardness = hardness
        super(FusedRaggedGravNetLinParsePool, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'cut_off': self.cut_off,
                  'hardness': self.hardness}
        base_config = super(FusedRaggedGravNetLinParsePool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def compute_output_shape(self, input_shapes):
        GNouts = super(FusedRaggedGravNetLinParsePool, self).compute_output_shape(input_shapes)
        #this layer adds a set of scatter indices to the output
        return GNouts + (None,) + (1,)

    def call(self, inputs):
        x = inputs[0]
        row_splits = inputs[1]
        
        coordinates = self.input_spatial_transform(x)
        selcoords = coordinates
        selx = x
        selrs = row_splits
        scatter_idxs = tf.range(tf.shape(row_splits)[0])[...,tf.newaxis]
        if self.cut_off is not None:
            threshold_values = inputs[2]
            
            selx, selrs, scatter_idxs = SelectThreshold(threshold_values, x, row_splits, hardness=self.hardness, threshold=self.cut_off)
            
            selcoords = tf.gather_nd(coordinates,scatter_idxs)
        
        neighbour_indices, distancesq = self.compute_neighbours_and_distancesq(selcoords, selrs)

        return self.create_output_features(selx, neighbour_indices, distancesq), selcoords, selrs, scatter_idxs
    

        
        
class FusedMaskedRaggedGravNet(FusedRaggedGravNet):
    '''
    direction: acc, scat 
      accumulates for vertices above threshold only, or scatters information of vertices
      above threshold to other vertices
    '''
    def __init__(self,
                 direction , 
                 threshold = 0.5,
                 ex_mode = 'xor',
                 **kwargs):
        
        assert direction=='acc' or direction=='scat'
        assert ex_mode=='xor' or ex_mode=='and'
        
        self.direction = direction
        self.threshold = threshold
        self.ex_mode = ex_mode
        
        super(FusedMaskedRaggedGravNet, self).__init__(**kwargs)
    
    def get_config(self):
        config = {'direction': self.direction,
                  'threshold': self.threshold,
                  'ex_mode': self.ex_mode}
        base_config = super(FusedMaskedRaggedGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def call(self, inputs):
        x = inputs[0]
        row_splits = inputs[1]
        masking_values = inputs[2]

        coordinates = self.input_spatial_transform(x)
        
        neighbour_indices, distancesq = self.compute_neighbours_and_distancesq(coordinates, row_splits, masking_values)

        return self.create_output_features(x, neighbour_indices, distancesq), coordinates
    
    def compute_neighbours_and_distancesq(self, coordinates, row_splits, masking_values):
        idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=1.0, tf_compatible=False,
                             masking_values=masking_values, threshold=self.threshold,
                             mask_mode=self.direction, mask_logic=self.ex_mode) 
        return idx[:,1:], dist[:,1:]



class FusedRaggedGravNetGarNetLike(FusedRaggedGravNet):
    '''
    bounces back and forth information between vertices above and below threshold
    '''
    def __init__(self,
                 threshold = 0.5,
                 parse_lin=True,
                 max_radius = 1.0,
                 **kwargs):
        
        self.threshold = threshold
        self.parse_lin=parse_lin
        self.max_radius=max_radius
        super(FusedRaggedGravNetGarNetLike, self).__init__(**kwargs)
        
        if len(self.n_propagate)%2:
            raise ValueError("FusedRaggedGravNetGarNetLike: n_propagate must have an even number of entries (altering up and down propagation)")
        
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'parse_lin': self.parse_lin,
                  'max_radius': self.max_radius}
        base_config = super(FusedRaggedGravNetGarNetLike, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    

    def call(self, inputs):
        x = inputs[0]
        row_splits = inputs[1]
        thresh_values = inputs[2]

        coordinates = self.input_spatial_transform(x)
        
        up_neighbour_indices, up_distancesq  = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=self.max_radius, masking_values =thresh_values, tf_compatible=False, 
                             threshold=self.threshold,
                             mask_mode='acc', mask_logic='xor') 
        
        up_neighbour_indices, up_distancesq = up_neighbour_indices[:,1:], up_distancesq[:,1:]
        
        down_neighbour_indices, down_distancesq  = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius=self.max_radius, masking_values =thresh_values, tf_compatible=False, 
                             threshold=self.threshold,
                             mask_mode='scat', mask_logic='xor') 
        
        down_neighbour_indices, down_distancesq = down_neighbour_indices[:,1:], down_distancesq[:,1:]
        
        allfeat = []
        tfet=[]
        features = x
        
        #tf.print(down_neighbour_indices)

        for i in range(len(self.input_feature_transform)):
            t = self.input_feature_transform[i]
            
            neighbour_indices = up_neighbour_indices
            distancesq = up_distancesq
            if i%2:
                neighbour_indices = down_neighbour_indices
                distancesq = down_distancesq
            
            features = t(features)
            orig_tf = features
            prev_shape = features.shape
            features = self.collect_neighbours(features, neighbour_indices, distancesq)
            features = tf.reshape(features, [-1, prev_shape[1]*2])
            
            #tf.print(i, tf.concat([thresh_values,features,orig_tf],axis=-1), summarize=10)
            
            allfeat.append(features)
            if self.parse_lin and i>0:
                up_distancesq *= 0. #the remaining parsing operations will be at zero distance
                down_distancesq *= 0.
        #tf.print(thresh_values.shape, allfeat[0].shape)
        #tf.print(tf.concat([thresh_values]+allfeat+tfet,axis=-1), summarize=120)
        features = tf.concat(allfeat +[x], axis=-1)
        
        return self.output_feature_transform(features), coordinates
    
        

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
        min = tf.reduce_min(rt, axis=1)  # [B, F]
        max = tf.reduce_max(rt, axis=1)  # [B, F]
        data_means = tf.gather_nd(means, tf.ragged.row_splits_to_segment_ids(rt.row_splits)[..., tf.newaxis])  # [SV, F]
        data_min = tf.gather_nd(min, tf.ragged.row_splits_to_segment_ids(rt.row_splits)[..., tf.newaxis])  # [SV, F]
        data_max = tf.gather_nd(max, tf.ragged.row_splits_to_segment_ids(rt.row_splits)[..., tf.newaxis])  # [SV, F]

        return tf.concat((data_means, data_min, data_max, x_data), axis=-1)

    def compute_output_shape(self, input_shape):
        data_input_shape = input_shape[0]
        return (data_input_shape[0], data_input_shape[1]*4)



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
    
    
def unique_with_inverse(x):
    y, idx = tf.unique(x)
    num_segments = tf.shape(y)[0]
    num_elems = tf.shape(x)[0]
    return y, tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments)
      
class RaggedVertexEater(keras.layers.Layer):
    '''
    reduces a set of neighbours (sum(V) x N x F) to a unique set of maximum activation vertices
    inputs : data (sum(V) x N x F), row splits (TF format)
    outputs: data (sum(V') x F , new row splits, remaining indices (as in sum(V)) to reconstruct original vertices (to be used with scatted_nd)
    '''
    def __init__(self, **kwargs):
        super(RaggedVertexEater, self).__init__(**kwargs)

    def call(self, x):
        
        x_data, row_splits = x[0], x[1]
        maxed = tf.reduce_max(x_data, axis=1) #sum(V) x F
        
        #for keras compile
        if row_splits.shape[0] is None:
            return maxed, row_splits, tf.range(0)
        
        batch_size = row_splits.shape[0] - 1
        
        new_rs = [int(0)]
        all_idcs = []
        rs_idcs =[]
        for b in tf.range(batch_size):
            n_u, idx = unique_with_inverse(tf.reduce_sum(maxed[row_splits[b]:row_splits[b + 1]],axis=1)) #unique only works on 1D
            idx = tf.cast(idx,dtype='int64')
            all_idcs.append(idx + tf.cast(row_splits[b],dtype='int64'))
            new_rs.append(new_rs[-1] + n_u.shape[0] )
            
        #print('new_rs',new_rs, all_idcs)
        all_idcs = tf.concat(all_idcs,axis=0)
        all_idcs,_ = tf.unique(all_idcs)
        all_idcs = tf.expand_dims(all_idcs,axis=1)
        all_idcs = tf.cast(all_idcs,dtype='int64')
        
        new_rs = tf.convert_to_tensor(new_rs)
        
        new_d = tf.gather_nd(maxed, all_idcs)
        
        shapes = self.compute_output_shape(x)
        #return new_d, new_rs, all_idcs
        return new_d,new_rs,all_idcs # tf.reshape(new_d, (-1,)+shapes[0]), tf.reshape(new_rs, (-1,)), tf.reshape(all_idcs, (-1,)+shapes[2])

    def compute_output_shape(self, input_shape):
        data_input_shape = input_shape[0]
        rs_input_shape = input_shape[1]
        return ( data_input_shape[1],), (None,), (1,) 


class RaggedNeighborIndices(keras.layers.Layer):
    def __init__(self, k,  **kwargs):
        super(RaggedNeighborIndices, self).__init__(**kwargs)
        self.k = k
        
        assert self.k > 2
        
    def call(self, x):
        assert isinstance(x, list)
        x_space, row_splits = x[0], x[1]
        row_splits = tf.cast(row_splits, tf.int32)
        ragged_split_added_indices, _ = rknn_op.RaggedKnn(num_neighbors=int(self.k), row_splits=row_splits,
                                                          data=x_space, add_splits=True)  # [SV, N+1]
        ragged_split_added_indices = ragged_split_added_indices[:, :][..., tf.newaxis]  # [SV, N]
        return ragged_split_added_indices

    def compute_output_shape(self, input_shape): 
        return (self.k,1)
    
    def get_config(self):
        config = {'k': self.k}
        base_config = super(RaggedNeighborIndices, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RaggedNeighborBuilder(keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(RaggedNeighborBuilder, self).__init__(**kwargs)
    
    def call(self, x):
        data, indices = x
        self_data = tf.expand_dims(data,axis=1) # sum(V) c 1 x F
        return tf.concat([self_data,tf.gather_nd(data, indices)],axis=1)
        
    def compute_output_shape(self, input_shape):
        dshape = input_shape[0]
        idcsshape = input_shape[1]
        return ( idcsshape[1]+1, dshape[1])
        
   
    
    
class VertexScatterer(keras.layers.Layer):
    '''
    '''
    def __init__(self, **kwargs):
        super(VertexScatterer, self).__init__(**kwargs)

    def call(self, x):
        
        x_data, scatter_idcs, protoshape = x[0], x[1], x[2]
        if protoshape.shape[0] is None:
            return x_data 
        
        scatter_idcs_n = tf.cast(scatter_idcs,dtype='int64')
        shape = tf.cast(tf.shape(protoshape),dtype='int64')[:-1]
        shape = tf.concat([shape,tf.cast(tf.shape(x_data)[1:2],dtype='int64')],axis=-1)
        
        tf.print(self.name, 'is scattering', x_data.shape, 'to', shape)
        
        return tf.scatter_nd(indices=scatter_idcs_n, updates=x_data, shape=shape)

    def compute_output_shape(self, input_shape):
        data_input_shape = input_shape[0]
        return data_input_shape


class GraphFunctionFilters(keras.layers.Layer):
    '''
    similar to shape filters, but instead of moments assume basic functions
    works because of space transformations
    
    input: sum(V) x N x F 
    
    features will be normalised w.r.t. first, central input vertex
    applies Nf functions independently in + and -: exp(c x^2) (p: 2*c)
    and evaluates compatibility as chi2-like measure
    something like 1 / (chi^2 + epsilon)
    
    output: sum(V) x F x S x 2*Nf
    
    as opposed to GraphShapeFilters: sum(V) x F x S x 3*Nf
    
    make this more generic by outputting sum(V) x F x S * n*Nf
    '''
    def __init__(self, 
                 n_filters: int,
                 **kwargs):
        
        super(GraphFunctionFilters, self).__init__(**kwargs)
        assert n_filters>0
        self.n_filters = n_filters
        
    def call(self, x): #takes space and feature dims in sum(V) x N x F_all format, interprets parts differently
        
        assert isinstance(x, list)
        
        space = x[0]# [:,:,:self.n_space]
        feat = x[1]

        #distances
        distances = space - space[:,0:1,:] #sum(V) x N x S
        distances = tf.expand_dims(tf.expand_dims(distances, axis=2),axis=4) #sum(V) x N x 1 x S x 1
        
        feat = tf.expand_dims(tf.expand_dims(feat, axis=3),axis=4)  #sum(V) x N x F x 1 x 1
        norm_feat = feat[:,0:1,...] #sum(V) x 1 x F x 1 x 1
        
        function_form_plus  = norm_feat * tf.math.exp(self.scalers[:,:,:,0,...] *  distances**2) - feat
        function_form_minus = norm_feat * tf.math.exp(self.scalers[:,:,:,1,...] *  distances**2) - feat
        
        function_diff = tf.where(distances>=0,function_form_plus,function_form_minus)
        #sum(V) x N x F x S x Nf
        function_diff = tf.reduce_sum(function_diff, axis=1)
        #sum(V) x F x S x Nf
        function_diff = tf.reshape(function_diff, (-1,function_diff.shape[1],function_diff.shape[2]*function_diff.shape[3]))
        #sum(V) x F x S * Nf
        
        return function_diff
        
        
        
    
    def build(self, input_shapes):

        self.n_space = input_shapes[0][-1] 
        self.n_shape_features = input_shapes[1][-1] 
        
        with tf.name_scope(self.name+"/1/"):
            self.scalers = self.add_weight(shape=(2*self.n_filters*self.n_space,),
                                            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.),
                                            name='scaler')
            self.scalers = tf.reshape(self.scalers, [1,1,1,2, self.n_space, self.n_filters])
        
        super(GraphFunctionFilters, self).build(input_shapes)
        
        
        
    def compute_output_shape(self, input_shapes):
        n_space = input_shapes[0][-1] 
        n_shape_features = input_shapes[1][-1] 
        return (n_shape_features, self.n_filters * self.n_space, )
    
    
    def get_config(self):
        config = {'n_filters': self.n_filters}
        base_config = super(GraphFunctionFilters, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    
class GraphShapeFilters(keras.layers.Layer):
    '''
    '''
    def __init__(self, 
                 n_filters: int,
                 n_moments: int,
                 direct_output : bool = True,
                 **kwargs):
        super(GraphShapeFilters, self).__init__(**kwargs)
        
        if direct_output:
            n_filters = 1
            
        self.n_filters = n_filters
        self.n_moments = n_moments
        self.direct_output = direct_output
        
        
        assert n_moments > 0 and n_moments < 5
        assert n_filters > 0
        

    
    def call(self, x): #takes space and feature dims in sum(V) x N x F_all format, interprets parts differently
        
        space = x[0]# [:,:,:self.n_space]
        feat = x[1] #[:,:,self.n_space:self.n_shape_features+self.n_space] #sum(V) x N x F
        
        #distances
        distances = space - space[:,0:1,:] #sum(V) x N x S
        distances = tf.expand_dims(distances, axis=2) #sum(V) x N x 1 x S
        
        feat = tf.expand_dims(feat, axis=3)  #sum(V) x N x F x 1
        feat_sum = tf.reduce_sum(feat, axis=1) #sum(V) x F x 1 , for norm
        
        #clip
        feat_sum = tf.where(tf.math.logical_and(feat_sum <  1e-3, feat_sum >=0),  1e-3, feat_sum)
        feat_sum = tf.where(tf.math.logical_and(feat_sum > -1e-3, feat_sum <=0), -1e-3, feat_sum)
        
        mean = 1. / (feat_sum) * tf.reduce_sum(distances*feat, axis=1) # sum(V) x F x S
        exp_mean = tf.expand_dims(mean, axis=1) # sum(V) x 1 x F x S
        
        all_moments = mean
        
        if self.n_moments > 1:
            var = 1. / (feat_sum) * tf.reduce_sum((distances - exp_mean)**2 *feat, axis=1) # sum(V) x F x S
            var = tf.math.sqrt(var + 1e-4) #get to same unit
            all_moments = tf.concat([all_moments, var],axis=-1)
        if self.n_moments > 2:
            skew = 1. / (feat_sum) * tf.reduce_sum((distances - exp_mean)**3 *feat, axis=1) # sum(V) x F x S
            skew = tf.math.pow(skew, 1./3.) #this might be resource intense
            all_moments = tf.concat([all_moments, skew],axis=-1)
        if self.n_moments > 3:
            mom4  = 1. / (feat_sum) * tf.reduce_sum((distances - exp_mean)**4 *feat, axis=1)
            mom4 = tf.math.sqrt(tf.math.sqrt(mom4 +1e-6) +1e-6)
            all_moments = tf.concat([all_moments, mom4],axis=-1)
        
        #maybe leave feature here and then process space*moments first. "this has an x shape in feature bla)
        shaped = tf.reshape(all_moments, [-1, self.n_shape_features, self.n_moments*self.n_space])
        
        if self.direct_output:
            return shaped
        
        shaped = tf.tile(shaped, [1,self.n_filters])   
        shaped = tf.nn.bias_add(shaped, self.expected_moments, data_format='N...C')     
        active =  1. / (100.* tf.abs(shaped) + 0.2)  #tf.math.exp(-10.*tf.abs(shaped)) #better gradient for large delta than gaus
        #tf.print(tf.reduce_mean(active), tf.reduce_max(active), tf.reduce_min(active))
        return  active

    
    def build(self, input_shapes):
        
        self.n_space = input_shapes[0][-1] 
        self.n_shape_features = input_shapes[1][-1] 
        if not self.direct_output:
            with tf.name_scope(self.name+"/1/"):
                self.expected_moments = self.add_weight(shape=(self.n_filters*self.n_moments*self.n_space,),
                                            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.),
                                            name='expected_moments')
            
    
        super(GraphShapeFilters, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        n_space = input_shapes[0][-1] 
        n_shape_features = input_shapes[1][-1] 
        return (n_shape_features, self.n_moments * self.n_space, )
    
    
    def get_config(self):
        config = {'n_filters': self.n_filters,
                  'n_moments': self.n_moments,
                  'direct_output': self.direct_output}
        base_config = super(GraphShapeFilters, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

