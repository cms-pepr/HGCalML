
import tensorflow as tf

from binned_select_knn_op import BinnedSelectKnn as select_knn
from accknn_op import AccumulateLinKnn as acc_knn
from push_knn_op import PushKnn as push_sum
from oc_helper_ops import SelectWithDefault as select
from oc_helper_ops import CreateMidx as oc_indices


from LossLayers import LossLayerBase, smooth_max

graph_condensation_layers = {}

#make a uniform interface to acc and push
def acc_sum(w, f, nidx,  mean_and_max=False):
    fs,_  = acc_knn(w, f, nidx, mean_and_max=mean_and_max)
    fs *= tf.cast(tf.shape(nidx)[1], 'float32') #get rid of mean in acc_knn
    return fs

def acc_mean(w, f, nidx,  mean_and_max=False):
    fs,_  = acc_knn(w, f, nidx, mean_and_max=mean_and_max)
    return fs


class RestrictedDict(dict):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for k in self.allowed_keys:
            self[k] = None
            
    def __setitem__(self, key, value):
        if not key  in self.allowed_keys:
            raise ValueError("only the following keys are allowed: "+str(self.allowed_keys))
        super().__setitem__(key, value)
        
    def check_filled(self):
        for k in self.allowed_keys:
            if self[k] is None:
                raise ValueError("check failed, at least one item not filled")


class GraphCondensation(RestrictedDict):
    
    def __init__(self,*args,**kwargs):
        '''
        This is a simple dict wrapper as dicts can be passed between keras layers
        '''
        self.allowed_keys = [
            'rs_down',
            'rs_up',
            'nidx_down',
            'distsq_down', #in case it's needed
            'sel_idx_up', # -> can also be used to scatter
            'weights_down',
            #'is_up'
            ]
        
        super().__init__(*args,**kwargs)
        
    ## just for convenience ##
    def check(self):
        assert self['weights_down'].shape[1] == self['nidx_down'].shape[1] 

    def K(self):
        return self['nidx_down'].shape[1]
    


from GravNetLayersRagged import SortAndSelectNeighbours  

class CreateGraphCondensation(tf.keras.layers.Layer):
    
    def __init__(self,
                 K=5,
                 score_threshold=0.5,
                 reduction_target=None,
                 n_knn_bins = 21,
                 safeguard = True, #makes sure there are never no points selected per row split
                 print_reduction = False,
                 **kwargs):
        
        super(CreateGraphCondensation, self).__init__(**kwargs)
        
        self.K = K
        self.score_threshold = score_threshold
        if reduction_target is not None:
            assert reduction_target > 0 and reduction_target < 1.
        self.reduction_target = reduction_target
        self.n_knn_bins = n_knn_bins
        self.safeguard = safeguard
        self.print_reduction = print_reduction
        
    def get_config(self):
        config = {'K': self.K, 'score_threshold': self.score_threshold, 
                  'reduction_target': self.reduction_target,
                  'n_knn_bins': self.n_knn_bins, 'safeguard': self.safeguard, 'print_reduction': self.print_reduction}
        base_config = super(CreateGraphCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    def build(self,input_shape):
        
        def _init(shape, dtype=None):
            return tf.constant(self.score_threshold)[...,tf.newaxis]

        self.dyn_score_threshold = self.add_weight(name = 'dyn_th', shape=(1, ),
                                       initializer = _init, 
                                        constraint = 'non_neg',
                                        trainable = False) 
            
        super(CreateGraphCondensation, self).build(input_shape)
        
    def update_thresh(self, trans, training):
        if self.reduction_target is None:
            return 
        if not self.trainable: #establish expected behaviour
            return
        smoothness = 10. #hard coded, but should be fine
        
        red = tf.cast(trans['rs_up'][-1], 
                      dtype='float32') / tf.cast(trans['rs_down'][-1],
                                                dtype='float32') 
        red_diff = red - self.reduction_target  # < 1, > -1
        step_up = (1. - self.dyn_score_threshold) * red_diff
        step_down = self.dyn_score_threshold * red_diff
        #if reduction is larger than target (diff < 0), score needs to step up
        step = tf.where(red_diff > 0., step_up, step_down)
        score_update = self.dyn_score_threshold + step / smoothness #slight reduction for safety
        #update only in training phase
        score_update = tf.keras.backend.in_train_phase(score_update,
                                                       self.dyn_score_threshold,
                                                       training=training)
        tf.keras.backend.update(self.dyn_score_threshold,score_update)
        
        tf.print(self.name, 'dyn th',self.dyn_score_threshold, 'red', red, 'target', self.reduction_target)
        
    def call(self, score, coords, rs, nidx = None, dist = None, always_promote=None, training = None):
        
        trans = GraphCondensation()
        trans['rs_down'] = rs
        
        #make threshold
        direction = tf.zeros_like(score, dtype='int32') + 1
        '''
        direction, if provided, has the following options:
         - 0: can only be neighbour
         - 1: can only have neighbours
         - 2: cannot be neighbour or have neighbours
         - any other number: can be neighbour and have neighbours
        '''
        
        direction = tf.where(score > self.dyn_score_threshold, 0, direction)
        
        if always_promote is not None:
            direction = tf.where(always_promote>0, 2, direction) #this should be a 2!!
            score = tf.where(always_promote>0, 1., score)
            
            
        #make this indices for gather and scatter   
        sel = tf.range(tf.shape(score)[0])[...,tf.newaxis]
        

        rsel = tf.RaggedTensor.from_row_splits(sel,rs)
        rscore = tf.RaggedTensor.from_row_splits(score,rs)
        
        threshold = self.dyn_score_threshold
        
        #make sure there is something left, bad with very inhomogenous batches, but good for training
        if self.safeguard: 
            mrss = tf.reduce_max(rscore,axis=1, keepdims=True)
            threshold = tf.reduce_min( tf.concat( [ tf.reduce_min(mrss)[tf.newaxis]*0.98, 
                                                self.dyn_score_threshold ], axis=0))
        
        #trans['is_up'] = tf.cast(score >= threshold, 'float32')
        rsel = tf.ragged.boolean_mask(rsel, rscore[...,0]  >= threshold)
        #use ragged to select
        trans['rs_up'] = tf.cast(rsel.row_splits,'int32')#for whatever reason
        
        #print(self.name, 'rs down',trans['rs_down'])
        #print(self.name, 'rs up',trans['rs_up'])
        #undo ragged
        trans['sel_idx_up'] = rsel.values
        
        with tf.control_dependencies([tf.assert_greater(rs + 1, trans['rs_up']),
                                      tf.assert_greater(tf.shape(score)[0]+1,
                                                        tf.shape(trans['sel_idx_up'])[0])]):
        
            if (nidx is not None) and (dist is not None):
                dist, nidx = SortAndSelectNeighbours.raw_call(dist,nidx, K=self.K+1) 
                raise ValueError("not implemented yet. Needs cleaning w.r.t. directions.")
            else: #yes this is swapped ordering
                nidx, dist = select_knn(self.K+1, coords, rs, direction = direction, 
                                    n_bins = self.n_knn_bins, name=self.name)
                
            nidx = tf.reshape(nidx, [-1, self.K+1]) #to define shape for later
            dist = tf.reshape(dist, [-1, self.K+1])
            dist = tf.stop_gradient(dist)
            dist = tf.where(nidx<0 , 0., dist)
        
        nidx = nidx[:,1:] #remove the self reference
        dist = dist[:,1:]
        
        #always sort by distance
        ssc = tf.argsort(tf.where(nidx<0 , 1e9, dist), axis=1)[:,:,tf.newaxis]
        trans['nidx_down'] = tf.gather_nd(nidx,ssc,batch_dims=1)
        trans['distsq_down'] = tf.gather_nd(dist,ssc,batch_dims=1)
        
        #set defined shape
        trans['nidx_down'] = tf.reshape(trans['nidx_down'], [-1, self.K])
        trans['distsq_down'] = tf.reshape(trans['distsq_down'], [-1, self.K])
        
        #print('''trans['nidx_down']''',trans['nidx_down'].shape)
        
        #a starting point, can be refined
        trans['weights_down'] = tf.nn.softmax(tf.exp(-trans['distsq_down']), axis=-1)
        
        #now make all shapes easy to recognise for keras
        trans['nidx_down'] = tf.reshape(trans['nidx_down'], [-1, self.K])
        trans['distsq_down'] = tf.reshape(trans['distsq_down'], [-1, self.K])
        trans['weights_down'] = tf.reshape(trans['weights_down'], [-1, self.K])
        
        if self.print_reduction:
            print(self.name, 'reduction', trans['rs_up'][-1], 'from', trans['rs_down'][-1],
                  tf.cast(trans['rs_up'][-1],dtype='float32')/tf.cast(trans['rs_down'][-1],dtype='float32') * 100, '%')
        #curiosity:
        #print(self.name, 'max number of assigned:', tf.reduce_max( tf.unique_with_counts( tf.reshape(trans['nidx_down'], [-1]) )[2] ))
        
        #trans.check_filled() # just during debugging
        self.update_thresh(trans,training)
        
        return trans
    
graph_condensation_layers['CreateGraphCondensation'] = CreateGraphCondensation

class CreateGatherIndices(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        '''
        This layer creates indices that can be used to relate back the condensed 
        points to the full dimensionality. This uses a maximum-fraction relation.
        If previous indices are provided, this operation can be chained
        
        Inputs:
        - graph transition
        - previous indices (opt.)
        
        Outputs:
        - indices that describe the relation to the 'upper' level. such that gather(feat, idxs) repeats features
        
        '''
        super(CreateGatherIndices, self).__init__(**kwargs)
        
    def call(self, transition : GraphCondensation, idxs = None):
        
        '''
        we have:
        - nidx, weight: 
            low -> high (use argmax weight ('down') from transition and relate back to reduced set ('up'))
            low -> None. (=-1); just keep empty
        '''
        
        if idxs is None:
            idxs = tf.range(transition['rs_up'][-1])
        else:
            tf.assert_equal(idxs.shape[0], transition['rs_up'][-1])#sanity check
        
        i_maxw = tf.argmax( transition['weights_down'], axis=1)
        print('i_maxw',i_maxw,i_maxw.shape)
        i_maxidx = tf.gather_nd(transition['nidx_down'], i_maxw, batch_dims=1)
        print('i_maxidx',i_maxidx,i_maxidx.shape)
        #now this needs to be somehow translated to the 'up' part...
        up_idx = tf.scatter_nd(transition['sel_idx_up'], idxs,
                               shape = transition['sel_idx_up'].shape)
        print('up_idx',up_idx,up_idx.shape)
        
        g_idx = tf.gather_nd(up_idx, i_maxidx)# now we're done?
        print('g_idx',g_idx,g_idx.shape)
        return g_idx
        
graph_condensation_layers['CreateGatherIndices'] = CreateGatherIndices
 
    
class PushUp(tf.keras.layers.Layer):
    
    def __init__(self,
                 mode = 'mean',
                 add_self = False,
                 **kwargs):
        '''
        If weights are provided to the call function, they will be forced to be >=0 .
        '''
        
        assert mode == 'sum' or mode == 'mean'
        
        self.mode = mode
        self.add_self = add_self
        super(PushUp, self).__init__(**kwargs)
    
    
    def get_config(self):
        config = {'mode': self.mode, 'add_self': self.add_self}
        base_config = super(PushUp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return (None, input_shapes[0][-1])
    
    def call(self,features, transition : GraphCondensation, weight = None):
        
        assert len(features.shape) == 2 
        
        up_f = features
        if weight is not None:
            weight = tf.nn.relu(weight) # + 1e-4 #safe guard, assume weights are O(1)
            up_f *= weight
                
        if self.mode == 'mean': 
            if weight is None:
                weight = tf.ones_like(up_f[:,0:1])
            up_f = tf.concat([weight, up_f], axis=-1)
            
        nidx = transition['nidx_down']
        nweights = transition['weights_down']
        
        if self.add_self:
            snidx = tf.concat([tf.range(tf.shape(nidx)[0])[:,tf.newaxis], nidx[:,1:]*0 -1 ],axis=1)
            is_up = nidx[:,0:1] < 0 #could also have no neighbours but then select would kill it anyway
            nidx = tf.where(is_up, snidx, nidx)
            sweight = tf.ones_like(nweights)
            nweights = tf.where(is_up, sweight, nweights)
            
        up_f = push_sum(nweights, up_f, nidx)
        up_f = tf.gather_nd(up_f, transition['sel_idx_up'])
        if self.mode == 'mean':
            wsum = tf.nn.relu(up_f[:,0:1]) #just to catch numerics
            wsum = tf.where(wsum>0., wsum, 1e-3)
            up_f = tf.math.divide_no_nan(up_f[:,1:] , wsum)
        up_f = tf.reshape(up_f, [-1, features.shape[1]])#just so the shapes are defined upon placeholder call
        
        return up_f
        
graph_condensation_layers['PushUp'] = PushUp

class SelectUp(tf.keras.layers.Layer):
    
    def __init__(self,
                 **kwargs):
        
        super(SelectUp, self).__init__(**kwargs)
    
    
    def compute_output_shape(self, input_shapes):
        return (None, input_shapes[0][-1])
    
    def call(self,features, transition : GraphCondensation):
        assert len(features.shape) == 2 
            
        up_f = tf.gather_nd(features, transition['sel_idx_up'])
        up_f = tf.reshape(up_f, [-1, features.shape[1]])#just so the shapes are defined upon placeholder call
        return up_f
    
graph_condensation_layers['SelectUp'] = SelectUp
    
class PullDown(tf.keras.layers.Layer):
    
    def __init__(self,
                 mode = 'mean',
                 **kwargs):
        
        assert mode == 'sum' or mode == 'mean' or mode == 'max' or mode == 'mean,max'
        
        self.mode = mode
        super(PullDown, self).__init__(**kwargs)
    
    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(PullDown, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        if self.mode == 'mean,max':
            return (None, 2*input_shapes[0][-1])
        return (None, input_shapes[0][-1])
    
    def call(self, features, transition : GraphCondensation, weights=None):
        
        if weights is not None:
            features *= weights
            if self.mode == 'mean':
                features = tf.concat([weights, features], axis=-1)
        
        down_f = tf.scatter_nd(transition['sel_idx_up'], 
                               features, 
                               shape = [tf.shape(transition['weights_down'])[0],
                                        tf.shape(features)[1]])
        
        if self.mode == 'mean':
            down_f = acc_mean(transition['weights_down'], down_f, transition['nidx_down'])
            if weights is not None:
                down_f = tf.math.divide_no_nan(down_f[:,1:] , down_f[:,0:1])
            down_f = tf.reshape(down_f, [-1, features.shape[1]])
        elif self.mode == 'sum':
            down_f = acc_sum(transition['weights_down'], down_f, transition['nidx_down'])
            down_f = tf.reshape(down_f, [-1, features.shape[1]])
        else: #mean,max
            if weights is not None:
                raise ValueError("mean,max with weights not implemented yet")
            
            down_f = acc_mean(transition['weights_down'], down_f, transition['nidx_down'], mean_and_max=True)
            if not self.mode == 'mean,max':
                down_f = down_f[:,features.shape[1]:] # select only max
                down_f = tf.reshape(down_f, [-1, features.shape[1]])
            else:
                down_f = tf.reshape(down_f, [-1, 2*features.shape[1]])
         
        return down_f
    
graph_condensation_layers['PullDown'] =   PullDown  


class SelectDown(tf.keras.layers.Layer): 
    
    def call(self, features, transition : GraphCondensation):
        
        #simply copied down
        down_f = tf.scatter_nd(transition['sel_idx_up'], 
                               features, 
                               shape = [tf.shape(transition['weights_down'])[0],
                                        tf.shape(features)[1]])
        
        nidx = transition['nidx_down']
        out = tf.reshape(select(nidx, down_f, 0.), [tf.shape(nidx)[0], features.shape[1] * nidx.shape[1]])
        print(self.name,'out shape',out.shape)
        return out
        
graph_condensation_layers['SelectDown'] =   SelectDown          
        
        

class Mix(tf.keras.layers.Layer): 
    '''
    Simply mixes the connected 'up' features with the 'down' features.
    Beware, the oprtation includes flattening, so can get big.
    
    Inputs:
    - features (V x F)
    - graph transition
    
    Output:
    - mixed features (V x (K+1)*F)
    '''
    
    def call(self, x, transition : GraphCondensation):
        nidx = transition['nidx_down']
        return tf.concat([x, tf.reshape(select(nidx, x, 0.), [-1, x.shape[-1] * nidx.shape[1]])],axis=1)  # V x (K+1)*F

graph_condensation_layers['Mix'] =   Mix

class UpDenseDown(tf.keras.layers.Layer):
    def __init__(self, dense, align_up = True, **kwargs):
        '''
        Pushes up, applies dense nodes (elu activated), and pulls down using 'mean,max' mode
        
        Inputs:
        - features
        - graph transition
        
        Outputs:
        - features
        
        Options:
        - dense: a list describing how many dense nodes should be applied to the accumulated 'up' features
        - align_up: if True (default)
                         the output of the last dense layer will be concatenated with the 
                         original 'up' features and fed through a dense layer (elu) that matches the 'down' units.
                    if False
                         the up vertices will be zeroed out.
        
        '''
        assert isinstance(dense, list)
        super(UpDenseDown, self).__init__(**kwargs)
        
        self.dense = dense
        self.align_up = align_up
        
        self.dense_layers = []
        for i,n in enumerate(dense):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.dense_layers.append(tf.keras.layers.Dense(n, activation='elu'))
                
        self.push_up = PushUp()
        self.pull_down = PullDown(mode='mean,max')
        
        if self.align_up:
            with tf.name_scope(self.name + "/2/"):
                self.up_dense = tf.keras.layers.Dense(2 * self.dense_layers[-1].units, activation='elu')
    
    def get_config(self):
        
        config = {'dense': self.dense, 'align_up': self.align_up}
        base_config = super(UpDenseDown, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        
        for i,d in enumerate(self.dense_layers):
            with tf.name_scope(self.name + "/1/" + str(i)):
                if not i:
                    d.build(input_shape)
                else:
                    d.build((None, self.dense_layers[i-1].units))
                    
        if self.align_up:
            with tf.name_scope(self.name + "/2/"):
                self.up_dense.build((None, input_shape[-1] + self.dense_layers[-1].units))
                    
        super(UpDenseDown, self).build(input_shape)
    
    def call(self, x, transition : GraphCondensation):
        
        up_x_orig = self.push_up(x, transition)
        up_x = up_x_orig
        for d in self.dense_layers:
            up_x = d(up_x)
        
        down_x = self.pull_down(up_x, transition)
        if self.align_up:
            up_x = self.up_dense(tf.concat([up_x_orig, up_x],axis=1))
            #zero for 'down'
            down_x += tf.scatter_nd(transition['sel_idx_up'], 
                               up_x, 
                               shape = tf.shape(down_x))
        return down_x


graph_condensation_layers['UpDenseDown'] =   UpDenseDown   


class DenseOnUp(tf.keras.layers.Layer):
    def __init__(self, dense, **kwargs):
        '''
        Pushes up, applies dense nodes (elu activated), and re-integrates down 
        (down nodes are kept as they are)
        
        Inputs:
        - features
        - graph transition
        
        Outputs:
        - features
        '''
        assert isinstance(dense, list)
        super(DenseOnUp, self).__init__(**kwargs)
        
        self.dense = dense
        
        self.dense_layers = []
        for i,n in enumerate(dense):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.dense_layers.append(tf.keras.layers.Dense(n, activation='elu'))
                
        self.select_up = SelectUp()
    
    def get_config(self):
        
        config = {'dense': self.dense}
        base_config = super(DenseOnUp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        
        if self.dense_layers[-1].units != input_shape[-1]:
            raise ValueError("DenseOnUp: the nodes of the last dense layer must match the input shape again.")
        
        for i,d in enumerate(self.dense_layers):
            with tf.name_scope(self.name + "/1/" + str(i)):
                if not i:
                    d.build(input_shape)
                else:
                    d.build((None, self.dense_layers[i-1].units))
                    
        super(DenseOnUp, self).build(input_shape)
    
    def call(self, x, transition : GraphCondensation):
        
        up_x = self.select_up(x, transition)
        
        for d in self.dense_layers:
            up_x = d(up_x)
            
        #this will add non zeros only to the 'up' ones
        x += tf.scatter_nd(transition['sel_idx_up'], 
                               up_x, shape = tf.shape(x))
        return x
        


graph_condensation_layers['DenseOnUp'] =   DenseOnUp                    
       
class CreateGraphCondensationEdges(tf.keras.layers.Layer): 
    
    def __init__(self,
                 edge_dense,
                 pre_nodes,
                 K,
                 self_dense = None,
                 no_self = False,
                 **kwargs):
        
        '''

        
        Inputs:
        - features
        - graph transition
        
        Outputs:
        - processed edges with K+1 entries (first is self-edge), V x K+1
        
        '''
        
        super(CreateGraphCondensationEdges, self).__init__(**kwargs)
        
        assert len(edge_dense) > 0 and isinstance(edge_dense, list) 
        assert self_dense is None or len(self_dense) > 0
        
        self.no_self = no_self
        self.pre_nodes = pre_nodes
        self.K = K
        
        
        with tf.name_scope(self.name + "/0/"):
            self.pre_dense = tf.keras.layers.Dense(self.pre_nodes, activation='elu', 
                                                   trainable = self.trainable)
        
        self.edge_dense = []
        self.self_dense = []
        for i,n in enumerate(edge_dense):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.edge_dense.append(tf.keras.layers.Dense(n, activation='elu', 
                                                   trainable = self.trainable))
                
        with tf.name_scope(self.name + "/1/" + str(i+1)):
            self.edge_dense.append(tf.keras.layers.Dense(self.K, 
                                                   trainable = self.trainable))
        
        if self_dense is None:
            self_dense = edge_dense
        if self.no_self:
            self_dense = []
            
        for i,n in enumerate(self_dense):
            with tf.name_scope(self.name + "/2/" + str(i)):
                self.self_dense.append(tf.keras.layers.Dense(n, activation='elu', 
                                                   trainable = self.trainable))
            
        with tf.name_scope(self.name + "/2/" + str(i+1)):
            self.self_dense.append(tf.keras.layers.Dense(1, 
                                                   trainable = self.trainable))
        
    def get_config(self):
        config = {'edge_dense': [self.edge_dense[i].units for i in range(len(self.edge_dense)-1)],
                  'self_dense': [self.self_dense[i].units for i in range(len(self.self_dense)-1)],
                  'no_self': self.no_self,
                  'K': self.K, 
                  'pre_nodes': self.pre_nodes}
        base_config = super(CreateGraphCondensationEdges, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        #print('input_shapes',input_shapes)
        #input_shape = input_shapes[0]
        
        self.pre_dense.build(input_shape)
        
        for i,d in enumerate(self.edge_dense):
            with tf.name_scope(self.name + "/1/" + str(i)):
                if not i:
                    d.build((None, self.K * (self.pre_dense.units+1)))
                else:
                    d.build((None, self.edge_dense[i-1].units))
                
        
        for i,d in enumerate(self.self_dense):
            with tf.name_scope(self.name + "/2/" + str(i)):
                if not i:
                    d.build(input_shape)
                else:
                    d.build((None, self.self_dense[i-1].units))
                    
        super(CreateGraphCondensationEdges, self).build(input_shape)
    
    def call(self, x, transition : GraphCondensation):
        nidx = transition['nidx_down']
        
        x_n = self.pre_dense(x)
        
        x_n = x_n[:, tf.newaxis, :] - select(nidx, x_n, 0.) # V x K x F
        x_n = tf.concat([x_n, transition['distsq_down'][:,:,tf.newaxis]],axis=2) # V x K x F+1
        x_n = tf.reshape(x_n, [-1, x_n.shape[1]*x_n.shape[2]]) # V x K * (F+1)
        
        for d in self.edge_dense:
            x_n = d(x_n)
        
        
            
        if self.no_self:
            x = x_n
        else:
            x_s = x
            for d in self.self_dense:
                x_s = d(x_s)
            x = tf.concat([x_s, x_n],axis=1)
            
        x = tf.nn.softmax(x,axis=1)
        
        return x
        
    
graph_condensation_layers['CreateGraphCondensationEdges'] =   CreateGraphCondensationEdges          

class AddNeighbourDiff(tf.keras.layers.Layer): 
        def __init__(self, *args, **kwargs):
            '''
            Concatenates the difference between the vertex and its neighbours to the vertex features.
            It also adds the distances to the neighbouts
            '''
            super(AddNeighbourDiff, self).__init__(*args, **kwargs)
            
        
        def call(self, x, transition : GraphCondensation):
            nidx = transition['nidx_down']
            x_nn = select(nidx, x, 0.)
            x_n = x[:, tf.newaxis, :] - x_nn # V x K x F, make it equivariant
            # now flatten and concat self in the beginning
            x_n = tf.reshape(x_n, [-1, x_n.shape[1]*x_n.shape[2]] ) # V x K * F
            x = tf.concat([x, x_n, transition['distsq_down']], axis=1)
            return x_n
        
graph_condensation_layers['AddNeighbourDiff'] =   AddNeighbourDiff

       
class CreateGraphCondensationEdges2(tf.keras.layers.Layer): 
    
    
    def call(self, x, transition : GraphCondensation):
        nidx = transition['nidx_down']
        x_nn = select(nidx, x, 0.)
        x_n = x[:, tf.newaxis, :] - x_nn # V x K x F
        
        #add one for noise
        x_n = tf.concat([x[:, tf.newaxis, :] - tf.reduce_mean(x_nn, axis=1, keepdims=True) , 
                         x_n], axis=1)
        
        return x_n
        
    
graph_condensation_layers['CreateGraphCondensationEdges2'] =   CreateGraphCondensationEdges2          
        
        
class CreateMultiAttentionGraphTransitions(tf.keras.layers.Layer): 
    
    def __init__(self,
                 n_heads,
                 pre_nodes,
                 K,
                 dense_nodes,
                 **kwargs):
        
        '''
        Inputs:
        - features
        - graph transition
        
        Outputs:
        - a list of graph transform descriptors, one per head
        
        '''
        
        super(CreateMultiAttentionGraphTransitions, self).__init__(**kwargs)
        self.dense_nodes = dense_nodes
        self.n_heads = n_heads
        self.pre_nodes = pre_nodes
        self.K = K
        
        with tf.name_scope(self.name + "/0/"):
            self.pre_dense = tf.keras.layers.Dense(self.pre_nodes, activation='elu')
        
        self.dense = []
        for i,n in enumerate(self.dense_nodes):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.dense.append(tf.keras.layers.Dense(n, activation='elu'))
        
        with tf.name_scope(self.name + "/1/" + str(i+1)):
            self.dense.append(tf.keras.layers.Dense(self.K * self.n_heads))
    
    def get_config(self):
        config = {'dense_nodes': self.dense_nodes, 
                  'n_heads': self.n_heads, 
                  'K': self.K, 
                  'pre_nodes': self.pre_nodes}
        base_config = super(CreateMultiAttentionGraphTransitions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def build(self, input_shape):
        
        self.pre_dense.build(input_shape)
        
        for i,d in enumerate(self.dense):
            with tf.name_scope(self.name + "/1/" + str(i)):
                if not i:
                    d.build((None, self.K * (self.pre_dense.units+1)))
                else:
                    d.build((None, self.dense[i-1].units))
    
    def call(self, x, transition : GraphCondensation):
        
        nidx = transition['nidx_down']
        
        x_n = self.pre_dense(x)
        
        x_n = x_n[:, tf.newaxis, :] - select(nidx, x_n, 0.) # V x K x F
        x_n = tf.concat([x_n, transition['distsq_down'][:,:,tf.newaxis]],axis=2) # V x K x F+1
        x_n = tf.reshape(x_n, [-1, x_n.shape[1]*x_n.shape[2]]) # V x K * (F+1)
        
        for d in self.dense:
            x_n = d(x_n)
            
        #now output is V x K x Nheads, transpose to V x Nheads x K
        x_n = tf.reshape(x_n, [-1, self.n_heads, self.K])
        x_n = tf.nn.softmax(x_n, axis=-1) # softmax over K
        out = []
        for i in range(self.n_heads):
            t = transition.copy() # shallow so not a big deal
            t['weights_down'] = x_n[:,i] # V x K
            out.append(t)
        return out
        
    
    
graph_condensation_layers['CreateMultiAttentionGraphTransitions'] =   CreateMultiAttentionGraphTransitions       
        

class InsertEdgesIntoTransition(tf.keras.layers.Layer):
    
    def __init__(self, 
                 exponent = 1, 
                 noise_assign_norm_threshold = 0.1, 
                 **kwargs):
        '''
        This also takes care of normalisation! Should not be done beforehand
        '''
        self.exponent = exponent
        self.noise_assign_norm_threshold = noise_assign_norm_threshold
        super(InsertEdgesIntoTransition, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'exponent': self.exponent,
                  'noise_assign_norm_threshold': self.noise_assign_norm_threshold}
        base_config = super(InsertEdgesIntoTransition, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
    
    def call(self, x_e, transition : GraphCondensation):
        trans = transition.copy()
        
        if len(x_e.shape) > 2:
            x_e = tf.squeeze(x_e, axis=2)

        if self.exponent != 1:
            x_e = x_e**self.exponent

        norm = tf.reduce_sum(x_e, axis=1, keepdims=True)
            
        if x_e.shape[1] == trans['weights_down'].shape[1]:
            trans['weights_down'] = x_e
        elif x_e.shape[1] == trans['weights_down'].shape[1] + 1:#these have noise edges
            trans['weights_down'] = x_e[:,1:]
        else:
            raise ValueError("Edges don't match graph transition")
        
        #apply norm threshold
        trans['weights_down'] = tf.where(norm < self.noise_assign_norm_threshold, 
                       tf.zeros_like(trans['weights_down']), trans['weights_down'])
        
        #normalise always
        trans['weights_down'] /= norm + 1e-6
        
        return trans
        
graph_condensation_layers['InsertEdgesIntoTransition'] =   InsertEdgesIntoTransition  

class GetNeighbourMask(tf.keras.layers.Layer): 
    
    def __init__(self, add_left = True, add_dim = True, **kwargs):
        self.add_left = add_left
        self.add_dim = add_dim
        super(GetNeighbourMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'add_left': self.add_left, 'add_dim': self.add_dim}
        base_config = super(GetNeighbourMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, trans : GraphCondensation):
        nidx = trans['nidx_down']
        ones = tf.ones_like(nidx, dtype='float32')
        out = tf.where(nidx<0, 0., ones)
        if self.add_left:
            out = tf.concat([ones[:,0:1],out], axis=1)#noise always possible
        if self.add_dim:
            out = out[...,tf.newaxis]
        return out
        
graph_condensation_layers['GetNeighbourMask'] =   GetNeighbourMask  
   
                 
###################### Only losses for building the hypergraph #######################

class LLGraphCondensationScore(LossLayerBase):
    
    def __init__(self,
                 K=64,
                 penalty_fraction=0.5,
                 extra_oc_scale = 0.25,
                 noise_scale=0.5,
                 low_energy_cut = 0.,#in GeV, below which the efficiency is allowed to drop
                 **kwargs):
        
        assert 0. <= penalty_fraction <= 1.
        self.K = K
        self.penalty_fraction = penalty_fraction
        self.extra_oc_scale = extra_oc_scale
        self.noise_scale = noise_scale
        self.low_energy_cut = low_energy_cut
        self.loss = self.weighted_bce_loss

        super(LLGraphCondensationScore, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'K': self.K, 'penalty_fraction': self.penalty_fraction,
                   'extra_oc_scale': self.extra_oc_scale, 'noise_scale': self.noise_scale,
                   'low_energy_cut': self.low_energy_cut}
        base_config = super(LLGraphCondensationScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _create_labels(self, nidx, t_idx, n_score, t_energy, smooth_labels = False):
       '''
       returns truth labels for the K neighbours of each hit V x K x 1
       and a loss mask (V x 1)
       '''
    
       n_t_idx = select(nidx, t_idx, -2) # V x K x 1
    
       # create the truth label -> V x K here, not V x 1!
       # needs to be w.r.t. each hit, not the max score one to not miss an object; mask noise at the end
       is_same = t_idx[...,tf.newaxis] == n_t_idx 
       is_same = tf.where(n_t_idx<0, False, is_same)#noise is never same
       none_same = tf.reduce_sum(tf.cast(is_same, tf.int32), axis=1) == 0
       
       # final loss mask applied to V x 1
       loss_mask = 1. - tf.cast( tf.logical_or(none_same, t_idx < 0), 'float32' )
       
       is_same_n_score = tf.where(is_same, n_score, 0.) # V x K x 1
       arg_max_neighbour = tf.argmax(is_same_n_score, axis=1) # V x 1
       
       ###### define the label
       if smooth_labels:
           y = 1. - smooth_max(is_same_n_score, axis=1) # V x 1; this is the label all but the max one should have
           y = tf.tile(y, [1, self.K])[:,:,tf.newaxis] # V x K x 1
           y_inv = 1. - y # V x 1; this is the label the max one should have
       else:
           y = tf.zeros_like(is_same_n_score)
           y_inv = tf.ones_like(is_same_n_score) #remove the extra last dim
       
       #set y to 1 for the max neighbour
       max_mask = tf.one_hot(arg_max_neighbour, self.K, on_value=True, off_value=False, axis=-1) # V x 1 x K
       max_mask = max_mask[:,0,:, tf.newaxis] # V x K x 1
       
       y = tf.where(max_mask, y_inv, y)
       # add low energy cut
       n_t_energy = select(nidx, t_energy, 0.) # V x K x 1
       y = tf.where(n_t_energy > self.low_energy_cut, y, 0.) # keep target score low also for low energy objects

       #set y for noise to zero
       y = tf.where(n_t_idx < 0, 0., y)

       return y, loss_mask, arg_max_neighbour, is_same
    
    
    def _get_d_weight(self, x, nidx, max_idx, mask):
       # x: V x K x C, 
       # nidx: V x K , 
       # max_idx -> select V x 1, 
       # mask: V x K x 1
       
       x_n = select(nidx, x, 0.) # V x K x C
       x_max = select(max_idx, x, 0.) #tf.gather_nd(x, max_idx, batch_dims=2 )
    
       distsq = tf.reduce_sum((x_n - x_max)**2, axis=-1, keepdims=True) # V x K x 1
       d_weight = tf.exp(-distsq)
       d_weight = tf.where(mask, d_weight, 0.)
       d_weight = tf.math.divide_no_nan(d_weight, tf.reduce_sum(d_weight, axis = 1, keepdims=True))
       return d_weight 

    def weighted_bce_loss(self, inputs):
        if len(inputs) == 4:
            score, coords, t_idx, rs = inputs
            t_energy = tf.ones_like(score) + self.low_energy_cut + 1000.
        elif len(inputs) == 5:
            score, coords, t_idx, t_energy, rs = inputs
        else:
            raise ValueError("Wrong number of inputs")

        assert len(score.shape) == 2 and len(coords.shape) == 2 and len(t_idx.shape) == 2
        
        nidx, _ = select_knn(self.K+1, coords, rs) # omit self
        nidx = nidx[...,1:] # V x K, remove self reference 
        
        n_score = select(nidx, score, 0.) # V x K x 1
        
        y, loss_mask, max_ids, n_mask = self._create_labels(nidx, t_idx, n_score, t_energy, smooth_labels = False)
        
        coords = tf.stop_gradient(coords) #only as static weights
        w = self._get_d_weight(coords, nidx, max_ids, n_mask)

        n_score = tf.clip_by_value(n_score, 1e-6, 1.-1e-6) #clip to avoid log(0)

        bce = - ( (1 - w) * y * tf.math.log(n_score) + w * (1. - y) * tf.math.log(1. - n_score) )
        bce = tf.reduce_sum(bce, axis=1) # V x 1
        bce = tf.math.divide_no_nan(tf.reduce_sum(bce * loss_mask) , tf.reduce_sum(loss_mask) + 0.1 ) # (this is typically > 10k if not > 100k)

        return bce


    def loss_v1(self, inputs):
        assert len(inputs) == 4 or (self.low_energy_cut > 0. and len(inputs) == 5)
        if len(inputs) == 4:
            score, coords, t_idx, rs = inputs
            t_energy = 1.
        elif len(inputs) == 5:
            score, coords, t_idx, t_energy, rs = inputs
        
        nidx, _ = select_knn(self.K, coords, rs, name=self.name)
        
        n_score = select(nidx, score, 0.)
        n_t_idx = select(nidx, t_idx, -2)
        
        is_not_noise = tf.where(t_idx>=0, 1., tf.zeros_like(score))
        n_not_noise = tf.reduce_sum( is_not_noise )
        #n_noise = tf.reduce_sum(tf.where(t_idx<0, 1., tf.zeros_like(score)) )
        
        is_same = t_idx[...,tf.newaxis] == n_t_idx
        #among all that are the same as 0 and not -1 require at least one score ~ 1
        same_n_score = tf.where(is_same, n_score, 0.) #mask other truth and noise
        n_same_n = tf.reduce_sum(tf.where(is_same, 1., tf.zeros_like(n_score)), axis=1)
        
        required_score = 1.
        if self.low_energy_cut > 0.:
            required_score = tf.where(t_energy > self.low_energy_cut, 1., tf.zeros_like(t_energy))
        
        one_max_loss = is_not_noise * tf.abs(required_score - smooth_max(same_n_score, axis=1)) #only for not noise, V x 1
        
        # for faster convergence in the beginning mostly. 
        # This becomes obsolete once max ishigh enough
        one_max_loss += is_not_noise * (required_score - tf.clip_by_value(tf.reduce_sum(same_n_score, axis=1), 
                                                                          0., required_score))
        
        one_max_loss = tf.reduce_sum(one_max_loss) / (n_not_noise + 1e-3)
        
        rest_low_loss = tf.reduce_sum(same_n_score, axis=1) 
        rest_low_loss -= is_not_noise * tf.reduce_max(same_n_score, axis=1) #remove max only if not noise
        rest_low_loss /= n_same_n + 1e-3
        
        rest_low_loss = tf.reduce_mean(rest_low_loss)
        
        self.wandb.log({self.name+'_global_loss': rest_low_loss,
                        self.name+'_local_max_loss': one_max_loss})

        return  (1. - self.penalty_fraction) * one_max_loss + self.penalty_fraction * rest_low_loss

    def loss_alternative(self, inputs):
        
        #this one works
        
        assert len(inputs) == 4
        score, coords, t_idx, rs = inputs
        
        nidx, _ = select_knn(self.K, coords, rs, name=self.name)
        
        n_score = select(nidx, score, 0.)
        n_t_idx = select(nidx, t_idx, -2)
        
        n_not_noise = tf.reduce_sum(tf.where(t_idx>=0, 1., tf.zeros_like(score)) )
        n_noise = tf.reduce_sum(tf.where(t_idx<0, 1., tf.zeros_like(score)) )
        
        t_idx = tf.where(t_idx<0, -1000, t_idx)
        
        #among all that are the same as 0 and not -1 require at least one score ~ 1
        same_n_score = tf.where(t_idx[...,tf.newaxis] == n_t_idx, n_score, 0.) #mask other truth and noise
        
        ssc  = tf.abs(1. - smooth_max(same_n_score, axis=1))
        local_max_loss = tf.reduce_sum(ssc) # + tf.reduce_sum(ssc2) + tf.reduce_sum(ssc4)
        local_max_loss /= n_not_noise + 1e-3
        
        global_loss = tf.reduce_mean(score)
        
        # now the noise
        noise_loss = tf.reduce_sum(tf.where(t_idx<0, score, 0.) )
        noise_loss /= n_noise + 1e-3
        
        self.add_prompt_metric(0.*local_max_loss, self.name+'_extra_score_loss')
        self.add_prompt_metric(global_loss, self.name+'_global_loss')
        self.add_prompt_metric(local_max_loss, self.name+'_local_max_loss')
        self.add_prompt_metric(noise_loss, self.name+'_noise_loss')

        return self.noise_scale * noise_loss +  (1. - self.penalty_fraction) * local_max_loss + self.penalty_fraction * global_loss
    
        
    def loss_alt2(self, inputs):
        assert len(inputs) == 4
        score, coords, t_idx, rs = inputs
        
        # simply push down score globally, ut ask for >= 1 per KNN truth?
        # see if that works
        
        #global_loss = tf.reduce_mean(score) #push down globally --> change this to push down all but the max per truth shower (OC like)
        #if score has no entries, reduce_mean returns nan
        #global_loss = tf.where(tf.math.is_finite(global_loss), global_loss ,0.)
        
        # now the more complicaed one
        nidx, _ = select_knn(self.K, coords, rs, name=self.name)
        
        n_score = select(nidx, score, 0.)
        n_t_idx = select(nidx, t_idx, -2)
        
        t_idx = tf.where(t_idx<0, -1000, t_idx)
        
        #among all that are the same as 0 and not -1 require at least one score ~ 1
        same_n_score = tf.where(t_idx[...,tf.newaxis] == n_t_idx, n_score, 0.) #mask other truth and noise
        
        #take the smooth max
        max_score = smooth_max(same_n_score,  axis=1)
        local_max_loss = tf.reduce_mean(1. - max_score)
        
        max_score_idx = tf.argmax(same_n_score, axis=1)
        max_score_global_idx = tf.gather_nd(nidx, max_score_idx, batch_dims=1)
        max_score_global_idx,_ = tf.unique(max_score_global_idx)
        
        not_noise = tf.gather_nd(t_idx[:,0], max_score_global_idx[:,tf.newaxis]) >= 0
        max_score_global_idx = tf.boolean_mask(max_score_global_idx, not_noise)
        
        n_max = tf.cast(max_score_global_idx.shape[0], 'float32') 
        n_nonmax = tf.cast(n_score.shape[0], 'float32') - n_max + 1e-3
        
        global_max_scores = tf.gather_nd(score, max_score_global_idx[:,tf.newaxis]) 
        #print('global_max_scores',global_max_scores.shape)
        global_loss = tf.reduce_sum( score ) - tf.reduce_sum(global_max_scores )
        global_loss /= n_nonmax
        
        ones = tf.ones_like(score)
        
        noise_loss = tf.reduce_sum( tf.where(t_idx < 0, score, 0.) ) / (tf.reduce_sum(tf.where(t_idx < 0, ones, 0.)) + 1e-3)
        
        self.add_prompt_metric(0.*global_loss, self.name+'_extra_score_loss')
        self.add_prompt_metric(global_loss, self.name+'_global_loss')
        self.add_prompt_metric(local_max_loss, self.name+'_local_max_loss')
        self.add_prompt_metric(noise_loss, self.name+'_noise_loss')

        return self.noise_scale * noise_loss +   self.penalty_fraction * global_loss + (1. - self.penalty_fraction) * local_max_loss

        
    def alt_loss(self, inputs):
        assert len(inputs) == 4
        score, coords, t_idx, rs = inputs
        
        # simply push down score globally, ut ask for >= 1 per KNN truth?
        # see if that works
        
        #global_loss = tf.reduce_mean(score) #push down globally --> change this to push down all but the max per truth shower (OC like)
        #if score has no entries, reduce_mean returns nan
        #global_loss = tf.where(tf.math.is_finite(global_loss), global_loss ,0.)
        
        # now the more complicaed one
        nidx, _ = select_knn(self.K, coords, rs, name=self.name)
        
        n_score = select(nidx, score, 0.)
        n_t_idx = select(nidx, t_idx, -2)
        
        t_idx = tf.where(t_idx<0, -1000, t_idx)
        
        #among all that are the same as 0 and not -1 require at least one score ~ 1
        n_score = tf.where(t_idx[...,tf.newaxis] == n_t_idx, n_score, 0.) #mask other truth and noise
        
        #take the smooth max
        max_score = smooth_max(n_score,  axis=1)
        
        ## test: push only the rest down, not globally everything?
        
        local_max_loss = tf.reduce_mean(1. - max_score)
        #if score has no entries, reduce_mean returns nan
        local_max_loss = tf.where(tf.math.is_finite(local_max_loss), local_max_loss ,0.)
        
        extra_score_loss = tf.zeros_like(local_max_loss)
        global_loss = extra_score_loss
        batch_counter = 0.*extra_score_loss
        ## add the OC beta penalty term to make sure there is at least one with highest beta
        #
        # this part is not so nice. maybe something more smooth such as sum of 1 - score**N or something?
        #
        ones = tf.ones_like(score)
        for i in tf.range(tf.shape(rs)-1):
            rs_score = score[rs[i]:rs[i+1]]
            rs_t_idx = t_idx[rs[i]:rs[i+1]]
            rs_ones = ones[rs[i]:rs[i+1]]
            Msel, *_ = oc_indices(rs_t_idx, calc_m_not=False)
            if Msel is None:
                continue
            rs_score_k_m = select(Msel, rs_score, 0.)
            
            K = tf.cast(tf.shape(rs_score_k_m)[0], 'float32')
            
            mask_k_m = select(Msel, rs_ones, 0.)
            #this can occasionally be slightly above 1.
            max_rs_score_k = smooth_max(rs_score_k_m, axis=1)
            
            #mean over all the other non-max hits
            pen_score_k = tf.reduce_sum(rs_score_k_m, axis=1) - tf.reduce_max(rs_score_k_m, axis=1) # K x 1 
            
            n_hits_k = tf.reduce_sum(mask_k_m, axis=1)
            
            pen_score_norm_k = pen_score_k / (n_hits_k - 1. + 1e-3) # K x 1 
            
            
            #print(tf.concat([ n_hits_k, max_rs_score_k, pen_score_k, pen_score_norm_k],axis=1))
            
            pen_score_norm_k = tf.reduce_sum(pen_score_norm_k) / (K + 1e-3)
            
            global_loss += pen_score_norm_k
            
            penalty = 1. - max_rs_score_k
        
            extra_score_loss += tf.reduce_sum(penalty) / (tf.reduce_sum( K ) + 1e-3 )
            batch_counter += 1.
        
        #this will be basically zero unless there are only few points left and objects might get lost
        extra_score_loss /= batch_counter + 1e-3
        
        global_loss /= batch_counter + 1e-3
        
        noise_loss = tf.reduce_sum( tf.where(t_idx < 0, score, 0.) ) / (tf.reduce_sum(tf.where(t_idx < 0, ones, 0.)) + 1e-3)
        
        self.add_prompt_metric(extra_score_loss, self.name+'_extra_score_loss')
        self.add_prompt_metric(global_loss, self.name+'_global_loss')
        self.add_prompt_metric(local_max_loss, self.name+'_local_max_loss')
        self.add_prompt_metric(noise_loss, self.name+'_noise_loss')

        return self.noise_scale * noise_loss +  self.extra_oc_scale * extra_score_loss + self.penalty_fraction * global_loss + (1. - self.penalty_fraction) * local_max_loss

    
graph_condensation_layers['LLGraphCondensationScore'] =   LLGraphCondensationScore  
        



class LLGraphCondensationEdges(LossLayerBase):
    def __init__(self,
                 **kwargs):
        '''
        Inputs:
        - edge score (V x (K + 1) x 1 ), the noise fraction is represented by the *first* entry in K+1
        - distances (V x K) (all undefined in case of an 'up' vertex)
        - neighbour index (V x K) (all -1 in case of an 'up' vertex)
        - truth index (V x 1)
        '''
        
        super(LLGraphCondensationEdges, self).__init__(**kwargs)
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
        
    def call(self, x_e, trans, t_idx):  
        
        nidx = trans['nidx_down']
            
        return super(LLGraphCondensationEdges, self).call([x_e, nidx, t_idx])  
        
    def loss(self, inputs):
        assert len(inputs) == 3
        e_score, nidx, t_idx = inputs 

        if len(e_score.shape) == 2:
            e_score = e_score[:,:, tf.newaxis]
        
        n_t_idx = select(nidx, t_idx, -2)[:,:,0]
        
        def create_encoding():
            
            ones_nidx = tf.ones_like(nidx, dtype='float32')
            
            noise_encoding = tf.concat([tf.ones_like(ones_nidx[:,0:1]),
                                        tf.zeros_like(ones_nidx)], axis=-1)
            
            #now the non-noise encoding
            # nidx does not have self !
            n_same = t_idx == n_t_idx  #V x K 
            
            no_noise_encoding = tf.concat([tf.zeros_like(ones_nidx[:,0:1]),
                                        tf.cast(n_same, 'float32')], axis=-1)
            
            none_same = tf.reduce_all( n_same == False ,axis=1, keepdims=True ) #also treat as noise
            
            like_noise = tf.logical_or(t_idx < 0, none_same)
            return tf.where(like_noise,  noise_encoding, no_noise_encoding)[:,:,tf.newaxis]#match score

        is_up = nidx[:,0] < 0
        prob = create_encoding()
        is_down_f = 1. - tf.cast(is_up,'float32')
 
        #just a sanity check
        assert prob.shape == e_score.shape and len(prob.shape) == 3
        bce = self.bce(prob, e_score) # -> V x K
        
        bce = tf.where( is_up[..., tf.newaxis], 0., bce )
            
        bce = tf.reduce_mean(bce, axis=1)
        
        lossval = tf.reduce_sum(is_down_f * bce) / (tf.reduce_sum(is_down_f) + 1.)
        
        #just for metrics
        def calc_acc(accscore):
            acc_score = 1. - tf.abs(accscore - prob)[:,:,0] #V x K
            acc_score = tf.reduce_mean(acc_score, axis=1)# V
            acc_score = tf.reduce_sum(is_down_f * acc_score) / tf.reduce_sum(is_down_f)
            return acc_score
        
        self.wandb_log({self.name+'_accuracy':  calc_acc(e_score),
                        self.name+'_bin_accuracy': calc_acc(tf.where(e_score>0.5, 1., tf.zeros_like(e_score))),
                        })
        
        return lossval  
        
        
        
graph_condensation_layers['LLGraphCondensationEdges'] =   LLGraphCondensationEdges  

        
from MetricsLayers import MLReductionMetrics

class MLGraphCondensationMetrics(MLReductionMetrics):
    
    def __init__(self, **kwargs):
        '''
        Inputs:
        - GraphCondensation
        - t_idx
        - t_energy
        - is_track (opt)
        
        '''
        super(MLGraphCondensationMetrics, self).__init__(**kwargs)
    def call(self, graph_transition : GraphCondensation, t_idx, t_energy, is_track = None):
        gt = graph_transition
        if is_track is None:
            self.metrics_call([gt['sel_idx_up'], t_idx, t_energy, gt['rs_down'], gt['rs_up']])
        else:
            self.metrics_call([gt['sel_idx_up'], t_idx, t_energy, is_track, gt['rs_down'], gt['rs_up']])
        return graph_transition


graph_condensation_layers['MLGraphCondensationMetrics'] =   MLGraphCondensationMetrics 



# convenience function
from LossLayers import LLClusterCoordinates

def add_attention(graph_transition, x, name, trainable = True):
    a = graph_transition.copy()
    att = tf.keras.layers.Dense(a['weights_down'].shape[1], activation='softmax', name=name, trainable = trainable)(x)
    a['weights_down'] = att
    return a
    

def point_pool(indict : dict, rs, name="p_pool", n_heads = 3, K_loss=64, trainable=True):
    
    #dict needs to contain: x, is_track, t_idx, t_spectator_weight
    
    x, is_track, t_idx, t_spectator_weight = indict['x'], indict['is_track'], indict['t_idx'], indict['t_spectator_weight']
    
    score = tf.keras.layers.Dense(1, activation='sigmoid',name=name+'_gc_score', trainable = trainable)(x)
    coords = tf.keras.layers.Dense(3, name=name+'_xyz_cond', use_bias = False, trainable = trainable)(x) 

    coords = LLClusterCoordinates(
        active = trainable,
                downsample=5000, #no need to use all
                scale = 1.,
                name = name+'_ll_cc',
                ignore_noise = True, #this is filtered by the graph condensation anyway
                hinge_mode = True
                )([coords, t_idx, t_spectator_weight, 
                                            score, rs ])
                
    score = LLGraphCondensationScore(
        active = trainable,
                name = name+'_ll_score',
        K=K_loss,
            )([score, coords, t_idx, rs])

    trans_a = CreateGraphCondensation(
        print_reduction = False,
                name = name+'_gc_create',
            score_threshold = 0.5,
            K=5
            )(score,coords,rs,
              always_promote = is_track)
    
    out = []        
    for i in range(n_heads):
        att = add_attention(trans_a, x, name+'_up_att_'+str(i), trainable = trainable)
        out.append( PushUp()(x,att) )
        
    odict = {}
    for k in indict.keys():
        odict[k] = SelectUp()(indict[k],trans_a)
        
    out.append(odict['x'])
    out = tf.keras.layers.Concatenate()(out)
    odict['x'] = out
        
    return trans_a, odict, trans_a['rs_up'] #for backscatter


def point_scatter(x, trans : list, dense_nodes = 64, name = ""):
    '''
    watch out -> this can become big
    '''
    
    trans = trans.copy()
    trans.reverse()
    for t in range(len(trans)):
        ta = trans[t]
        x = SelectDown()(x,ta)
        x = tf.keras.layers.Dense(dense_nodes, activation='elu', name = name+'_d_'+str(t))(x)
        
    return x
    














    
    
    
    