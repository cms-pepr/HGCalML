
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL", "move layers still in usage or restructure all the layer files")

import tensorflow as tf
import tensorflow.keras as keras
from select_knn_op import SelectKnn
from accknn_op import AccumulateKnn
from condensate_op import BuildCondensates
from pseudo_rs_op import CreatePseudoRS
from select_threshold_op import SelectThreshold

from latent_space_grid_op import LatentSpaceGrid


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



class Condensate(tf.keras.layers.Layer):
    def __init__(self, t_d, t_b, soft, feature_length, **kwargs):
        print("Condensate Layer: warning, do not use this in the model, a bunch of hardcoded stuff!")
        self.t_d = t_d
        self.t_b = t_b
        self.soft = soft
        self.feature_length=feature_length
        super(Condensate, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'t_d': self.t_d,
                  't_b': self.t_b,
                  'soft': self.soft,
                  'feature_length': self.feature_length}
        base_config = super(Condensate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shapes): # data, rs
        return input_shapes[0], (None,) #condensates and row splits
    
    def call(self, inputs):
        x, row_splits = inputs[0], inputs[1]
        
        
        n_ccoords = x.shape[-1] - 14 #FIXME: make this less static
        x_pred = x[:,self.feature_length:]
        n_pred = x_pred.shape[-1]
        
        betas = x_pred[:,0:1]
        print('betas min, mean, max', tf.reduce_min(betas),tf.reduce_mean(betas),tf.reduce_max(betas))
        ccoords = x_pred[:,n_pred-n_ccoords:n_pred]
        
        print('n cluster coordinates', n_ccoords)
        
        _, row_splits = RaggedConstructTensor()([x, row_splits])
        row_splits = tf.cast(row_splits,dtype='int32')
        
        asso, iscond, ncond = BuildCondensates(ccoords, betas, row_splits, 
                                               radius=self.t_d, min_beta=self.t_b, 
                                               soft=self.soft)
        iscond = tf.reshape(iscond, (-1,))
        ncond = tf.reshape(ncond, (-1,1))

        zero = tf.constant([[0]],dtype='int32')
        ncond = tf.concat([zero,ncond],axis=0,name='output_row_splits')
        dout = x[iscond>0]
        dout = tf.reshape(dout,[-1,dout.shape[-1]], name="predicted_final_condensates")
        
        return dout, ncond 
        

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
        




class RaggedGlobalExchange(keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        Inputs:
        - data
        - row splits
        
        Outputs:
        - (means,min,max,data)
        
        '''
        super(RaggedGlobalExchange, self).__init__(**kwargs)
        self.num_features = -1

    def build(self, input_shape):
        data_shape = input_shape[0]
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



