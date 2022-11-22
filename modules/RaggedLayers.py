
'''
Only 'real' ragged layers here that take as input ragged tensors and output ragged tensors
'''

import tensorflow as tf
import time

from GravNetLayersRagged import CondensateToIdxs
from assign_condensate_op import calc_ragged_shower_indices, calc_ragged_cond_indices, collapse_ragged_noise

from ragged_tools import print_ragged_shape

ragged_layers = {}

class RaggedCreateCondensatesIdxs(CondensateToIdxs):
    
    def __init__(self, 
                 collapse_noise = True,
                 **kwargs):
        '''
        Quite complex layer with a lot going on, and a lot of output...
        
        :param collapse_noise: collapses noise index to one representative hit (if there is noise)
        
        Inputs:
        
        - beta
        - cluster space coords
        - individual distance scaler
        - no condensation mask (1: do not condensate point) (optional)
        - row splits
        
        Outputs:
        - gather indices to ragged [events, condensation point, hits, F]
        - gather indices to ragged [events, condensation point, F]
        - reverse ragged to flat indices (see calc_ragged_cond_indices)
        - reverse flat to flat indices (see calc_ragged_cond_indices)
        - association index per hit
        
        - dynamic t_d
        - dynamic t_b
        
        
        Options:
        - t_d: distance threshold for inference clustering
        - t_b: beta threshold for inference clustering
        - active: can switch off condensation, just for faster pre-training
                  in this case (off) it each input point will be its own condensation point
        - keepnoise: noise will not be assigned a negative association index but a 'self' index
        - return_thresholds: returns t_b and t_d in addition to the outputs
                             and initialises them dynamically, so they have to be trained
                             
        - 
        
        '''
        
        super(RaggedCreateCondensatesIdxs, self).__init__(**kwargs)
        self.return_full_assignment = True
        self.collapse_noise = collapse_noise
    
    def get_config(self):
            #when saving always assume active!
            config = {'collapse_noise': self.collapse_noise}
            base_config = super(RaggedCreateCondensatesIdxs, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, inputs):
        
        beta, ccoords, d, rs = inputs
        
        st = time.time()
        pred_sid, asso_idx, alpha ,ncond, *_ = super(RaggedCreateCondensatesIdxs, self).call(inputs)
        
        #print('idxs time a', time.time()-st, 's')
        st = time.time()
        
        try:
            ch_idx = calc_ragged_shower_indices(pred_sid, rs)
            c_idx, rev, revflat = calc_ragged_cond_indices(pred_sid, alpha, ncond, rs)
            
            #print('idxs time b', time.time()-st, 's')
            #print('c_idx, rev, revflat',c_idx, rev, revflat)
            
            if self.collapse_noise:
                #print(ch_idx.shape, ch_idx.row_splits, c_idx.shape, c_idx.row_splits)
                ch_idx = collapse_ragged_noise(ch_idx, tf.gather_nd(pred_sid[:,0], c_idx))
                
        except Exception as e:
            import pickle
            with open('test.pkl','wb') as f:
                pickle.dump(
                    {'pred_sid': pred_sid.numpy(),
                     'beta': beta.numpy(),
                     'ccoords': ccoords.numpy(),
                     'd': d.numpy(),
                     'alpha': alpha.numpy(),
                     'asso_idx': asso_idx.numpy(),
                     'ncond': ncond.numpy(),
                     'rs': rs.numpy(),
                     'ch_idx': ch_idx.numpy()
                        },f)
            raise e
    
        if self.return_thresholds:
            return ch_idx, c_idx, rev, revflat, pred_sid, asso_idx, self.dyn_t_d, self.dyn_t_b
        else:
            return ch_idx, c_idx, rev, revflat, pred_sid, asso_idx

#register
ragged_layers['RaggedCreateCondensatesIdxs'] = RaggedCreateCondensatesIdxs        

class RaggedSelectFromIndices(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        '''
        
        Inputs:
        - data (V x F)
        - indices (output from RaggedCreateCondensatesIdxs)
        
        Output:
        - ragged output (depending on indices)
        
        '''
        super(RaggedSelectFromIndices, self).__init__(**kwargs)
        
    def call(self, inputs):
        assert len(inputs) == 2
        data, idx = inputs
        return tf.gather_nd(data, idx)

#register
ragged_layers['RaggedSelectFromIndices'] = RaggedSelectFromIndices       

class RaggedDense(tf.keras.layers.Dense):
    
    def __init__(self, *args, **kwargs):
        '''
        Behaves like a standard dense layer on the last dimension of a ragged tensor.
        Last dimension must not be ragged.
        '''
        super(RaggedDense, self).__init__(*args, **kwargs)
        
    def build(self, input_shapes):
        super(RaggedDense, self).build([input_shapes[0],input_shapes[-1]])

    def unpack_ragged(self, rt):
        rs = []
        while hasattr(rt, 'values'):
            rs.append(rt.row_splits)
            rt = rt.values
        return rt, rs
    
    def pack_ragged(self, vals, rsl):
        rt = vals
        for i in reversed(range(len(rsl))):
            rt = tf.RaggedTensor.from_row_splits(rt, rsl[i], validate=False)
        return rt

    def call(self, inputs):
        vals, rsl = self.unpack_ragged(inputs)
        vals = super(RaggedDense, self).call(vals)
        return self.pack_ragged(vals,rsl)

#register
ragged_layers['RaggedDense'] = RaggedDense

class RaggedMixHitAndCondInfo(tf.keras.layers.Layer):
    
    def __init__(self, operation='subtract', **kwargs):
        '''
        Concat only supports same number of features for both
        
        Inputs:
        - ragged tensor of features of hits associated to condensation points: [None, None, None, F]
        - ragged tensor of features of condensation points: [None, None, F]
        
        Output:
        - ragged tensor where operation is applied to first input (broadcasted): 
             if add or subtract: [None, None, None, F]
             if concat: [None, None, None, F+F]
        '''
        assert operation == 'subtract' or operation == 'add' or operation == 'concat' or operation == 'mult'
        
        if operation == 'subtract':
            self.operation = self._sub
        if operation == 'add':
            self.operation = self._add
        if operation == 'concat':
            self.operation = self._concat
        if operation == 'mult':
            self.operation = self._mult
        self.operationstr = operation
            
        super(RaggedMixHitAndCondInfo, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'operation': self.operationstr}
        base_config = super(RaggedMixHitAndCondInfo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def _add(self, lhs, rhs): 
        return lhs + rhs
    def _sub(self, lhs, rhs): 
        return lhs - rhs
    def _concat(self, lhs, rhs):
        return tf.concat([lhs,lhs*0 + rhs], axis=-1)
    def _mult(self, lhs, rhs):
        return lhs * rhs
        
    def call(self, inputs):
        assert len(inputs) == 2
        hitsrt, cprt = inputs
        
        tf.assert_equal(hitsrt.shape[-1], cprt.shape[-1], self.name+" last shape must match.")
        tf.assert_equal(hitsrt.shape[0], cprt.shape[0], self.name+" first shape must match.")
        tf.assert_equal(hitsrt.shape.ndims, cprt.shape.ndims + 1, self.name+" shape issue.")
        tf.assert_equal(hitsrt.shape.ndims, 4, self.name+" shape issue.")
        #tf.assert_equal(tf.reduce_sum(hitsrt.shape * 0 +1), tf.reduce_sum(cprt.shape * 0 +1) + 1, self.name+" shape issue.")
        #print('hitsrt, cprt 1',hitsrt.shape, cprt.shape)
        cprt = tf.expand_dims(cprt, axis=2)
        rt=cprt
        rrt=hitsrt
        
        
        # breaks at
        # :  tf.Tensor([1 1 1 1], shape=(4,), dtype=int32) tf.Tensor([3 1 1 1], shape=(4,), dtype=int32) tf.Tensor([1392 1377 1352 1187], shape=(4,), dtype=int32)
        # so rt has wrong row splits
        try:
            return self.operation(hitsrt,cprt)
        except Exception as e:
            print('>>>')
            while hasattr(rt ,'values'):
                print(': ',rrt.row_lengths() ,rt.row_lengths(), rrt.values.row_lengths())
                #print(':: ',rt,rrt)
                rt = rt.values
                rrt = rrt.values
            print('<<<')
            raise e
        
    

#register
ragged_layers['RaggedMixHitAndCondInfo'] = RaggedMixHitAndCondInfo    


class RaggedCollapseHitInfo(tf.keras.layers.Layer):
    
    def __init__(self, operation='mean', **kwargs):
        '''
        Just a wrapper around tf.reduce_xxx(..., axis=2).
        operation can be either mean, max, sum, or any other callable (tf) function that takes an axis argument.
        
        Input:
        - data in [batch, ragged condensation point, ragged hit, F]
        
        Output:
        - mean or max along hit dimension
        
        '''
        
        #assert operation == 'mean' or operation == 'max' or operation == 'sum' or callable(operation)

        super(RaggedCollapseHitInfo, self).__init__(**kwargs)
        
        if operation == 'mean' or 'reduce_mean' in operation:
            self.operation = tf.reduce_mean
        elif operation == 'max' or 'reduce_max' in operation:
            self.operation = tf.reduce_max
        elif operation == 'sum' or 'reduce_sum' in operation:
            self.operation = tf.reduce_sum
        else:
            self.operation = operation
        
    def get_config(self):
        config = {'operation': self.operation}
        base_config = super(RaggedCollapseHitInfo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        return self.operation(inputs, axis=2)
        

ragged_layers['RaggedCollapseHitInfo'] = RaggedCollapseHitInfo




class RaggedPFCMomentum(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        Inputs:
        - mom correction 
        - energy sum
        - track mom
        - istrack
        '''
        super(RaggedPFCMomentum, self).__init__(**kwargs)
        
    def call(self, inputs):
        assert len(inputs) == 4
        c, esum, tmom, ist = inputs
        mom = c * esum
        return tf.where(ist > 0, c * tmom, mom)

ragged_layers['RaggedPFCMomentum'] = RaggedPFCMomentum

class RaggedPFCIsNoise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        Inputs:
        - asso index ragged
        
        Outputs:
        - is noise tensor as float
        
        '''
        super(RaggedPFCIsNoise, self).__init__(**kwargs)
        
    def call(self, inputs):
        asso = inputs[...,tf.newaxis]
        return tf.where(asso >= 0, 0. , tf.ones_like(asso, dtype='float32'))

ragged_layers['RaggedPFCIsNoise'] = RaggedPFCIsNoise


class RaggedToFlatRS(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        simply takes [events, ragged-something, F] and returns
        [X, F] + row splits.
        Only one ragged dimension here
        '''
        super(RaggedToFlatRS, self).__init__(**kwargs)
        
    def call(self, inputs):
        return inputs.values, inputs.row_splits
            
    
ragged_layers['RaggedToFlatRS'] = RaggedToFlatRS 

class FlatRSToRagged(tf.keras.layers.Layer):
    def __init__(self, validate=True, **kwargs):
        '''
        simply takes [X, F] + row splits and returns
        [events, ragged-something, F]
        Only one ragged dimension here
        '''
        super(FlatRSToRagged, self).__init__(**kwargs)
        self.validate = validate
        
    def get_config(self):
        config = {'validate': self.validate}
        base_config = super(FlatRSToRagged, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        return tf.RaggedTensor.from_row_splits(inputs[0],inputs[1], validate=self.validate)
            
    
ragged_layers['FlatRSToRagged'] = FlatRSToRagged      





class ForceExec(tf.keras.layers.Layer):
    
    def call(self, inputs):
        return inputs[0]
        
ragged_layers['ForceExec'] = ForceExec       
'''

Also need a garnet like layer to mix in condensation point info




scratch area

for condensation corrections:





'''