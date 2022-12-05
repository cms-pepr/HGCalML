
import tensorflow as tf

from binned_select_knn_op import BinnedSelectKnn as select_knn
from accknn_op import AccumulateLinKnn as acc_knn
from push_knn_op import PushKnn as push_sum
from oc_helper_ops import SelectWithDefault as select
from oc_helper_ops import CreateMidx as oc_indices


from LossLayers import LossLayerBase, smooth_max

graph_condensation_layers = {}

#make a uniform interface to acc and push
def acc_sum(w, f, nidx):
    fs,_  = acc_knn(w, f, nidx, mean_and_max=False)
    fs *= tf.cast(tf.shape(nidx)[1], 'float32') #get rid of mean in acc_knn
    return fs

def acc_mean(w, f, nidx):
    fs,_  = acc_knn(w, f, nidx, mean_and_max=False)
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
            'weights_down'
            ]
        
        super().__init__(*args,**kwargs)
        
    ## just for convenience ##
    


class CreateGraphCondensation(tf.keras.layers.Layer):
    
    def __init__(self,
                 K,
                 score_threshold=0.5,
                 n_knn_bins = 21,
                 **kwargs):
        
        super(CreateGraphCondensation, self).__init__(**kwargs)
        
        self.K = K
        self.score_threshold = score_threshold
        self.n_knn_bins = n_knn_bins
        
    def get_config(self):
        config = {'K': self.K, 'score_threshold': self.score_threshold, 'n_knn_bins': self.n_knn_bins}
        base_config = super(CreateGraphCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, score, coords, rs, always_promote=None):
        
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
        
        direction = tf.where(score > self.score_threshold, 0, direction)
        
        if always_promote is not None:
            direction = tf.where(always_promote>0, 2, direction)
            score = tf.where(always_promote>0, 1., score)
        
        #make this indices for gather and scatter   
        sel = tf.range(tf.shape(score)[0])[...,tf.newaxis]
        

        rsel = tf.RaggedTensor.from_row_splits(sel,rs)
        rscore = tf.RaggedTensor.from_row_splits((score  > self.score_threshold),rs)
        rsel = tf.ragged.boolean_mask(rsel, rscore[...,0])
        #use ragged to select
        trans['rs_up'] = tf.cast(rsel.row_splits,'int32')#for whatever reason
        #undo ragged
        trans['sel_idx_up'] = rsel.values
        
        nidx, dist = select_knn(self.K+1, coords, rs, direction = direction, n_bins = self.n_knn_bins)
        
        trans['nidx_down'] = nidx[:,1:] #remove the self reference?
        trans['distsq_down'] = dist[:,1:]
        
        trans['weights_down'] = tf.nn.softmax(tf.exp(-dist)[:,1:], axis=-1)
        
        #now make all shapes easy to recognise for keras
        trans['nidx_down'] = tf.reshape(trans['nidx_down'], [-1, self.K])
        trans['distsq_down'] = tf.reshape(trans['distsq_down'], [-1, self.K])
        trans['weights_down'] = tf.reshape(trans['weights_down'], [-1, self.K])
        
        
        trans.check_filled() # just debugging
        
        return trans
    
graph_condensation_layers['CreateGraphCondensation'] = CreateGraphCondensation

'''
these also need implementations as weighted means, and possibly a weird fused version of it
(but that is >LO performance optimisation, keep it for later if needed

maybe put the 'include-self' here instead of removign it up there?
or add the self ref back?
'''
    
    
class PushUp(tf.keras.layers.Layer):
    
    def __init__(self,
                 mode = 'mean',
                 **kwargs):
        
        assert mode == 'sum' or mode == 'mean'
        
        self.mode = mode
        super(PushUp, self).__init__(**kwargs)
    
    
    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(PushUp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return (None, input_shapes[0][-1])
    
    def call(self,features, transition : GraphCondensation, weight = None):
        
        assert len(features.shape) == 2 
        
        up_f = features
        if weight is not None:
            up_f *= weight
                
        if self.mode == 'mean': 
            if weight is None:
                weight = 1. + tf.zeros_like(up_f[:,0:1])
            up_f = tf.concat([weight, up_f], axis=-1)
            
        up_f = push_sum(transition['weights_down'], up_f, transition['nidx_down'])
        up_f = tf.gather_nd(up_f, transition['sel_idx_up'])
        if self.mode == 'mean': 
            up_f = tf.math.divide_no_nan(up_f[:,1:] , up_f[:,0:1])
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
        
        assert mode == 'sum' or mode == 'mean'
        
        self.mode = mode
        super(PullDown, self).__init__(**kwargs)
    
    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(PullDown, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return (None, input_shapes[0][-1])
    
    def call(self, features, transition : GraphCondensation):
        
        down_f = tf.scatter_nd(transition['sel_idx_up'], 
                               features, 
                               shape = [tf.shape(transition['weights_down'])[0],
                                        tf.shape(features)[1]])
        
        if self.mode == 'mean':
            down_f = acc_mean(transition['weights_down'], down_f, transition['nidx_down'])
        else:
            down_f = acc_sum(transition['weights_down'], down_f, transition['nidx_down'])
         
        down_f = tf.reshape(down_f, [-1, features.shape[1]])
        return down_f
    
graph_condensation_layers['PullDown'] =   PullDown  
       
    
###################### Only losses for building the hypergraph #######################

class LLGraphCondensationScore(LossLayerBase):
    
    def __init__(self,
                 K=64,
                 penalty_fraction=0.5,
                 extra_oc_scale = 1.,
                 **kwargs):
        
        assert 0. <= penalty_fraction <= 1.
        self.K = K
        self.penalty_fraction = penalty_fraction
        self.extra_oc_scale = extra_oc_scale
        super(LLGraphCondensationScore, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'K': self.K, 'penalty_fraction': self.penalty_fraction, 'extra_oc_scale': self.extra_oc_scale}
        base_config = super(LLGraphCondensationScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def loss(self, inputs):
        assert len(inputs) == 4
        score, coords, t_idx, rs = inputs
        
        # simply push down score globally, ut ask for >= 1 per KNN truth?
        # see if that works
        
        global_loss = tf.reduce_mean(score) #push down globally
        #if score has no entries, reduce_mean returns nan
        global_loss = tf.where(tf.math.is_finite(global_loss), global_loss ,0.)
        
        # now the more complicaed one
        nidx, _ = select_knn(self.K, coords, rs)
        
        n_score = select(nidx, score, 0.)
        n_t_idx = select(nidx, t_idx, -1)
        
        t_idx = tf.where(t_idx<0, -1000, t_idx)
        
        #among all that are the same as 0 and not -1 require at least one score ~ 1
        n_score = tf.where(t_idx[...,tf.newaxis] == n_t_idx, n_score, 0.) #mask other truth and noise
        
        #take the smooth max
        max_score = smooth_max(n_score,  axis=1)
        
        local_max_loss = tf.reduce_mean(1. - max_score)
        #if score has no entries, reduce_mean returns nan
        local_max_loss = tf.where(tf.math.is_finite(local_max_loss), local_max_loss ,0.)
        
        extra_score_loss = tf.zeros_like(local_max_loss)
        batch_counter = 0.*extra_score_loss
        ## add the OC beta penalty term to make sure there is at least one with highest beta
        for i in tf.range(tf.shape(rs)-1):
            rs_score = score[rs[i]:rs[i]+1]
            rs_t_idx = t_idx[rs[i]:rs[i]+1]
            Msel, *_ = oc_indices(rs_t_idx, calc_m_not=False)
            if Msel is None:
                continue
            rs_score_k_m = select(Msel, rs_score, 0.)
            max_rs_score_k = smooth_max(rs_score_k_m, axis=1)
            penalty = 1. - max_rs_score_k
        
            extra_score_loss += tf.reduce_sum(penalty) / (tf.reduce_sum( tf.cast( tf.shape(max_rs_score_k)[0], 'float32' ) ) + 1e-3 )
            batch_counter += 1.
        
        #this will be basically zero unless there are only few points left and objects might get lost
        extra_score_loss /= batch_counter + 1e-3
        
        self.add_prompt_metric(extra_score_loss, self.name+'_extra_score_loss')
        self.add_prompt_metric(global_loss, self.name+'_global_loss')
        self.add_prompt_metric(local_max_loss, self.name+'_local_max_loss')

        return self.extra_oc_scale * extra_score_loss + self.penalty_fraction * global_loss + (1. - self.penalty_fraction) * local_max_loss

    
graph_condensation_layers['LLGraphCondensationScore'] =   LLGraphCondensationScore  
        
        





    
    
    
    