import tensorflow as tf
from object_condensation import oc_loss, OC_loss
from oc_helper_ops import SelectWithDefault
from oc_helper_ops import CreateMidx
import time
from baseModules import LayerWithMetrics
from binned_select_knn_op import BinnedSelectKnn
import time
from datastructures.TrainData_NanoML import id_str_to_idx

from ragged_tools import print_ragged_shape

def smooth_max(var, eps = 1e-3, **kwargs):
    return eps * tf.reduce_logsumexp(var/eps, **kwargs)

def one_hot_encode_id(t_pid, n_classes):
    valued_pids = tf.zeros_like(t_pid)+4 #defaults to 4 as unkown
    valued_pids = tf.where(tf.math.logical_or(t_pid==22, tf.abs(t_pid) == 11), 0, valued_pids) #isEM
    
    valued_pids = tf.where(tf.abs(t_pid)==211, 1, valued_pids) #isHad
    valued_pids = tf.where(tf.abs(t_pid)==2212, 1, valued_pids) #proton isChHad
    valued_pids = tf.where(tf.abs(t_pid)==321, 1, valued_pids) #K+
    
    valued_pids = tf.where(tf.abs(t_pid)==13, 2, valued_pids) #isMIP
    
    valued_pids = tf.where(tf.abs(t_pid)==111, 3, valued_pids) #pi0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==2112, 3, valued_pids) #neutron isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==130, 3, valued_pids) #K0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==310, 3, valued_pids) #K0 short
    valued_pids = tf.where(tf.abs(t_pid)==3122, 3, valued_pids) #lambda isNeutrHadOrOther
    
    valued_pids = tf.cast(valued_pids, tf.int32)[:,0]
    
    known = tf.where(valued_pids==4,tf.zeros_like(valued_pids),1)
    valued_pids = tf.where(known<1,3,valued_pids)#set to 3
    known = tf.expand_dims(known,axis=1) # V x 1 style
    
    depth = n_classes #If n_classes=pred_id.shape[1], should we add an assert statement? 
    return tf.one_hot(valued_pids, depth), known

def huber(x, d):
    losssq  = x**2   
    absx = tf.abs(x)                
    losslin = d**2 + 2. * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)

def quantile(x,tau):
    return tf.maximum(tau*x, (tau-1)*x)
    
def _calc_energy_weights(t_energy, t_pid=None, upmouns = True, alt_energy_weight=True):
    
        lower_cut = 0.5
        w = tf.where(t_energy > 10., 1., ((t_energy-lower_cut) / 10.)*10./(10.-lower_cut))
        w = tf.nn.relu(w)
        if alt_energy_weight:
            extra = t_energy/50. + 0.01
            #extra = tf.math.divide_no_nan(extra, tf.reduce_sum(extra,axis=0,keepdims=True)+1e-3)
            w *= extra
        return w
    
class AmbiguousTruthToNoiseSpectator(LayerWithMetrics):
    '''
    Sets the truth to noise spectators if it is ambiguous for a group of neighbours
    
    Technically, this is not a loss layer, but as it affects the truth
    this seems to be the most reasonable place to put it.
    
    Inputs: 
     - neighbour indicies (selected assuming the belong to the same object)
     - spectator weights
     - truth indices
     
    Outputs:
     - adapted spectator weights
     - adapted truth indices
    '''
    def __init__(self, threshold=0.68, return_score=False, active=True, **kwargs):
        super(AmbiguousTruthToNoiseSpectator, self).__init__(**kwargs)
        #self.record_metrics=True #DEBUG
        self.threshold  = threshold
        self.return_score = return_score
        self.active = active
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'return_score':self.return_score,
                  'active': self.active}
        base_config = super(AmbiguousTruthToNoiseSpectator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        
        nidx, sw, tidx, energy = None, None, None, None
        
        if len(inputs)==3:
            nidx, sw, tidx = inputs
            energy = tf.cast(tf.ones_like(sw),dtype='float32')
        elif len(inputs)==4:
            nidx, sw, tidx, energy = inputs
        else:
            raise ValueError("# inputs must be 3 or 4")
        
        if not self.active:
            if self.return_score:
                return sw, tidx, tf.ones_like(sw)
            else:
                return sw, tidx
        
        padding_default = -10000
        
        in_tidx = tidx
        
        n_tidx = SelectWithDefault(nidx, tidx, padding_default)
        n_energy = SelectWithDefault(nidx, energy, 0.)
        n_mask = SelectWithDefault(nidx, tf.ones_like(energy), 0.)
        group_energy = tf.reduce_sum(n_energy,axis=1)
        
        is_same = n_tidx[:,0:1]==n_tidx
        is_same = n_mask * tf.cast(is_same,dtype='float32')
        
        same_energy = tf.reduce_sum(n_energy*is_same,axis=1)
        sscore = tf.math.divide_no_nan(same_energy, group_energy)
        is_same =  sscore > self.threshold
        
        sw = tf.where(is_same, sw, 1.)#set ambiguous to spectator
        tidx = tf.where(is_same, tidx, -1)#set ambiguous to noise
        tidx = tf.where(in_tidx<0, in_tidx, tidx) # leave noise
        
        is_same = tf.cast(is_same,dtype='float32')
        same_f = tf.reduce_sum(is_same)/tf.reduce_sum(tf.ones_like(is_same))
        #metrics
        self.add_prompt_metric(same_f, 
                               self.name+'_non_amb_truth_fraction')
        if self.return_score:
            return sw, tidx, sscore
        else:
            return sw, tidx
        
from ragged_tools import normalise_index
class NormaliseTruthIdxs(tf.keras.layers.Layer):
    
    def __init__(self, active=True, add_rs_offset=True, **kwargs):
        '''
        changes arbitrary truth indices to well defined indices such that
        sort(unique(t_idx)) = -1, 0, 1, 2, 3, 4, 5, ... for each row split
        
        This should be called after every layer that could have modified
        the truth indices or removed hits, if the output needs to be regular.
        
        This Layer takes < 10ms usually so can be used generously.
        
        :param active: determines if it should be active. 
                       In pure inference mode that might not be needed
        
        Inputs: truth indices, row splits
        Output: new truth indices
        
        '''
        if 'dynamic' in kwargs:
            super(NormaliseTruthIdxs, self).__init__(**kwargs)
        else:
            super(NormaliseTruthIdxs, self).__init__(dynamic=True,**kwargs)
            
        self.active = active
        self.add_rs_offset = add_rs_offset
    
    def get_config(self):
        config = {'active': self.active,
                  'add_rs_offset': self.add_rs_offset}
        base_config = super(NormaliseTruthIdxs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
        
    def call(self, inputs):
        assert len(inputs) == 2
        t_idx, rs = inputs
        
        #double unique
        if not self.active or rs.shape[0] == None:
            return t_idx
        
        return normalise_index(t_idx, rs, self.add_rs_offset)
        

class LossLayerBase(LayerWithMetrics):
    """Base class for HGCalML loss layers.
    
    Use the 'active' switch to switch off the loss calculation.
    This needs to be done by hand, and is not handled by the TF 'training' flag, since
    it might be desirable to 
     (a) switch it off during training, or 
     (b) calculate the loss also during inference (e.g. validation)
     
     
    The 'scale' argument determines a global sale factor for the loss. 
    """
    
    def __init__(self, active=True, scale=1., 
                 print_loss=False,
                 print_batch_time=False,
                 return_lossval=False, 
                 print_time=False,#compat, has no effect
                 record_batch_time=False,
                 **kwargs):
        super(LossLayerBase, self).__init__(**kwargs)
        
        if print_time:
            print("print_time has no effect and is only for compatibility purposes")
        
        self.active = active
        self.scale = scale
        self.print_loss = print_loss
        self.print_batch_time = print_batch_time
        self.record_batch_time = record_batch_time
        self.return_lossval=return_lossval
        with tf.init_scope():
            now = tf.timestamp()
        self.time = tf.Variable(-now, name=self.name+'_time', 
                                trainable=False)
        
    def get_config(self):
        config = {'active': self.active ,
                  'scale': self.scale,
                  'print_loss': self.print_loss,
                  'print_batch_time': self.print_batch_time,
                  'record_batch_time': self.record_batch_time,
                  'return_lossval': self.return_lossval}
        base_config = super(LossLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        super(LossLayerBase, self).build(input_shape)
    
    def create_safe_zero_loss(self, a):
        zero_loss = tf.zeros_like(tf.reduce_mean(a))
        zero_loss = tf.where(tf.math.is_finite(zero_loss),zero_loss,0.)
        return zero_loss
    
    def call(self, inputs):
        lossval = tf.constant([0.],dtype='float32')
        a = None
        if not isinstance(inputs,list):
            a = inputs
        elif isinstance(inputs,list) and len(inputs) == 2:
            a,_ = inputs
        elif isinstance(inputs,list) and len(inputs) > 2:
            a,*_ = inputs
        else:
            raise ValueError("LossLayerBase: input not understood")
            
        if self.active:
            now = tf.timestamp() 
            
            #check for empty batches
                
            #generally protect against empty batches
            if a.shape[0] is None or a.shape[0] == 0:
                zero_loss = self.create_safe_zero_loss(a)
                print(self.name,'returning zero loss',zero_loss)
                lossval = zero_loss
            else:
                lossval = self.scale * self.loss(inputs)
                
            self.maybe_print_loss(lossval,now)
            
            lossval = tf.debugging.check_numerics(lossval, self.name+" produced inf or nan.")
            #this can happen for empty batches. If there are deeper problems, check in the losses themselves
            #lossval = tf.where(tf.math.is_finite(lossval), lossval ,0.)
            if not self.return_lossval:
                self.add_loss(lossval)
                
        self.add_prompt_metric(lossval, self.name+'_loss')
        if self.return_lossval:
            return a, lossval
        else:
            return a
    
    def loss(self, inputs):
        '''
        Overwrite this function in derived classes.
        Input: always a list of inputs, the first entry in the list will be returned, and should be the features.
        The rest is free (but will probably contain the truth somewhere)
        '''
        return tf.constant(0.,dtype='float32')
    
    def maybe_print_loss(self,lossval,stime=None):
        if self.print_loss:
            if hasattr(lossval, 'numpy'):
                print(self.name, 'loss', lossval.numpy())
            else:
                tf.print(self.name, 'loss', lossval)
                
        
        if self.print_batch_time or self.record_metrics:
            now = tf.timestamp() 
            prev = self.time
            prev = tf.where(prev<0.,now,prev)
            batchtime = now - prev            #round((time.time()-self.time)*1000.)/1000.
            losstime = 0.
            if stime is not None:
                losstime = now - stime
            tf.keras.backend.update(self.time,now)
            if self.print_batch_time:
                tf.print(self.name,'batch time',batchtime*1000.,'ms')
                if stime is not None:
                    tf.print(self.name,'loss time',losstime*1000.,'ms')
                print(self.name,'batch time',batchtime*1000.,'ms')
                if stime is not None:
                    print(self.name,'loss time',losstime*1000.,'ms')
            if self.record_batch_time and self.record_metrics:
                self.add_prompt_metric(batchtime, self.name+'_batch_time')
                if stime is not None:
                    self.add_prompt_metric(losstime, self.name+'_loss_time')
            

    def compute_output_shape(self, input_shapes):
        if self.return_lossval:
            return input_shapes[0], (None,)
        else:
            return input_shapes[0]



class LLDummy(LossLayerBase):
    
    def loss(self, inputs):
        return tf.reduce_mean(inputs)


class LLValuePenalty(LossLayerBase):
    
    def __init__(self, 
                 default : float = 0.,
                 **kwargs):
        '''
        Simple value penalty loss, tries to keep values around default using simple
        L2 regularisation; returns input
        '''
        
        super(LLValuePenalty, self).__init__(**kwargs)
        self.default = default
        
    def get_config(self):
        config = {'default': self.default}
        base_config = super(LLValuePenalty, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def loss(self, inputs):
        lossval = tf.reduce_mean((self.default - inputs)**2)
        return tf.where(tf.math.is_finite(lossval), lossval, 0.)#DEBUG




class CreateTruthSpectatorWeights(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold, 
                 minimum, 
                 active,
                 **kwargs):
        '''
        active: does not enable a loss, but acts similar to other layers using truth information
                      as a switch to not require truth information at all anymore (for inference)
                      
        Inputs: spectator score, truth indices
        Outputs: spectator weights (1-minimum above threshold, 0 else)
        
        '''
        super(CreateTruthSpectatorWeights, self).__init__(**kwargs)
        self.threshold = threshold
        self.minimum = minimum
        self.active = active
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'minimum': self.minimum,
                  'active': self.active,
                  }
        base_config = super(CreateTruthSpectatorWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0] 
    
    def call(self, inputs):
        if not self.active:
            return inputs[0]
        
        abovethresh = inputs[0] > self.threshold
        #notnoise = inputs[1] >= 0
        #noise can never be spectator
        return tf.where(abovethresh, tf.ones_like(inputs[0])-self.minimum, 0.)


#helper for OC theshold losses
def _thresh(x, d, t, invert=False):
    '''
    x: something to be multiplied by a threshold modifier
    d: distance to threshold
    t: threshold
    invert: False: threshold is an upper bound
            True: threshold is a lower bound
    '''
    a = 10. / (t + 1e-2)
    
    dmt = a * (d-t)
    if hasattr(dmt, 'row_splits'):
        dmt = tf.ragged.map_flat_values(tf.tanh,dmt)
    else:
        dmt = tf.tanh(dmt)
        
    if invert:
        return (0.5 * (1. + dmt)) * x
    return (0.5 * (1. - dmt)) * x


class LLPushTracks(LossLayerBase):
    
    def loss(self, inputs):
        x = None
        if len(inputs) == 4:
            beta, is_track, t_idx, x = inputs
        else:
            beta, is_track, t_idx = inputs
        
        
        beta = beta[is_track[:,0]>0.]
        t_idx = t_idx[is_track[:,0]>0.]
        
        if x is not None:
            print(self.name, beta, t_idx, x[is_track[:,0]>0.])
        
        
        tclass = tf.where(t_idx<0, 0., tf.ones_like(beta))
        
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tclass, beta))

from ragged_tools import  rconcat
class LLLocalEnergyConservation(LossLayerBase):
    
    def __init__(self, 
                 grad_rel_threshold=0.1, 
                 K = 64, 
                 dist_scale = 1.,
                 **kwargs):
        '''
        Inputs:
        weight, coords, pred_sid, pfc_energy, pfc_idx, t_idx, t_energy,  rs
        
        rs are for the non-ragged inputs:
        weight, coords, pred_sid, t_idx and t_energy
        
        ragged are only:
        pfc_energy, pfc_idx
        
        '''
        super(LLLocalEnergyConservation, self).__init__(**kwargs)
        self.grad_rel_threshold = grad_rel_threshold
        self.dist_scale = dist_scale
        self.K = K
        
    def get_config(self):
        config = {'grad_rel_threshold': self.grad_rel_threshold,
                  'dist_scale': self.dist_scale,
                  'K': self.K,
                  }
        base_config = super(LLLocalEnergyConservation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def w_ax1(self, x, w, epsilon=1e-4, **kwargs):
        x = tf.reduce_sum(x * w, axis=1, **kwargs)
        w = tf.reduce_sum(w, axis=1, **kwargs)
        return tf.math.divide_no_nan(x, w + epsilon)

    def max_arg_w(self,sel, w):
        sw_k_m = SelectWithDefault(sel, w, 0.)
        srange_k_m = SelectWithDefault(sel, tf.range(tf.shape(w)[0])[...,tf.newaxis], -1)
        maxarg_k = tf.argmax(sw_k_m,axis=1)
        return tf.gather_nd(srange_k_m, maxarg_k, batch_dims=1)
        
    
    #this is just to scale the gradient here
    def scale_gradient(self,energy_kp, t_energy_kp, grad_rel_threshold):
        rs = tf.cast(energy_kp.row_splits, 'int32')
        energy_kp = energy_kp.values
        t_energy_kp = t_energy_kp.values #not ragged
        reldiff = 1. - energy_kp/t_energy_kp # [kp, 1], use to define gradient reduction for well predicted showers
        reldiff = tf.clip_by_value(reldiff, -grad_rel_threshold, grad_rel_threshold)
        gradscaler = tf.abs(reldiff)/grad_rel_threshold
        gradscaler = tf.stop_gradient(gradscaler) #stop gradient for scaler itself
        energy_kp = gradscaler * energy_kp + (1. - gradscaler) * tf.stop_gradient(energy_kp)
        return tf.RaggedTensor.from_row_splits(energy_kp, rs, validate=False)

    
    
    def loss(self, inputs):
        weight, coords, pred_sid, pfc_energy, pfc_idx, t_idx, t_energy,  rs = inputs

        if rs.shape[0] is None:
            return 0.* tf.reduce_sum(weight)

        coords = tf.stop_gradient(coords)
        weight = tf.stop_gradient(weight)
        
        grad_rel_threshold = self.grad_rel_threshold
        K = self.K
        
        
        t_idx, t_nper_k = normalise_index(t_idx, rs, True, return_n_per=True)#now it's normalised and spanning row splits
        
        #get truth coords defined at max weight = beta (for local energy conservation)
        t_sel, _, _ = CreateMidx(t_idx, False)
        t_maxarg_k = self.max_arg_w(t_sel, weight)
        
        t_coords_k = tf.gather_nd(coords, t_maxarg_k)
        
        #t_sw_k_m = SelectWithDefault(t_sel, weight, 0.)
        #t_sumw_k = tf.reduce_sum(t_sw_k_m, axis=1)#sum weights = beta
        
        t_energy_k = tf.gather_nd(t_energy, t_maxarg_k)
        
        ### truth part done
        
        pred_sid, _ = normalise_index(pred_sid, rs, True, return_n_per=True)#now it's normalised and spanning row splits
        #print('>>>>>',pred_sid, nper, pfc_idx.row_splits)
        #print(pfc_idx)
        #print(tf.gather_nd( pfc_idx))
        
        #build the kp - predicted parts, these are ragged
        
        not_isnoise_kp = tf.gather_nd(pred_sid, pfc_idx) >= 0
        
        t_energy_kp = tf.gather_nd(t_energy, pfc_idx)
        t_energy_kp = tf.ragged.boolean_mask(t_energy_kp, not_isnoise_kp[...,0])
        
        energy_kp = tf.ragged.boolean_mask(pfc_energy, not_isnoise_kp[...,0])#pfc_energy laready ragged
        energy_kp = self.scale_gradient(energy_kp, t_energy_kp, grad_rel_threshold) #scale grad here
        
        coords_kp = tf.gather_nd(coords, pfc_idx)
        coords_kp = tf.ragged.boolean_mask(coords_kp, not_isnoise_kp[...,0])
        
        
        #print('>>>>>>>',coords_kp.row_splits, t_energy_kp.row_splits, energy_kp.row_splits, energy_kp, t_energy_kp)
        #exit()
        #create gradient scaler in non-ragged
        
        #p_rs = tf.cast(energy_kp.row_splits, 'int32')
        t_rs = tf.cast(t_nper_k, 'int32')
        
        #concat per rs
        t_energy_k = tf.RaggedTensor.from_row_splits(t_energy_k, t_rs, validate=False)#we know this works
        t_coords_k = tf.RaggedTensor.from_row_splits(t_coords_k, t_rs, validate=False)
        
        
        #create row splits for kNN from concat of nper and t_nper
        energy_kp  = rconcat([energy_kp, 0.*energy_kp ], axis=-1) # rec, truthzero
        t_energy_k = rconcat([0.*t_energy_k, t_energy_k ], axis=-1) # zerorec, truth
        
        allenergies = rconcat([energy_kp, t_energy_k],axis=1)
        allcoords = rconcat([coords_kp, t_coords_k],axis=1)
        
        totrs = tf.cast(allcoords.row_splits, 'int32')
        
        
        allenergies = allenergies.values
        allcoords = allcoords.values
        
        #now we have:
        # mixed coordinates [ V x C ]
        # mixed enegies [V x 2]
        # mixed row splits
        
        idx, distsq = BinnedSelectKnn(K, allcoords, totrs)
        
        #scale distance so that the Gaussian weighting makes sense
        distsq /= tf.reduce_mean(tf.reduce_max(distsq, axis=1)) #max for each vertex, min over all
        
        m_energy = SelectWithDefault(idx, allenergies, 0.)
        
        w_m_energy = tf.expand_dims(tf.exp(-6. * distsq / 2.),axis=2) * m_energy
        
        diff = tf.reduce_sum(w_m_energy, axis=1)
        diff = diff[...,0] - diff[...,1]
        reldiff = tf.math.divide_no_nan(diff , diff[...,1] + 1e-2)
        
        abs_mean_diff = tf.abs(tf.reduce_mean(reldiff))
        
        self.add_prompt_metric(abs_mean_diff, self.name + '_loss')
        
        return abs_mean_diff
        #print(reldiff, '\n', tf.reduce_mean(reldiff))


class LLDynOCThresholds(LossLayerBase):
    def __init__(self, 
                 purity_weight = 0.5,
                 object_energy_weight = 'modlog',
                 **kwargs):
        '''
        beta-weighted here
        '''
        self.object_energy_weight = object_energy_weight
        self.purity_weight = purity_weight
        
        super(LLDynOCThresholds, self).__init__(**kwargs)

    def get_config(self):
        config={'purity_weight': self.purity_weight,
                'object_energy_weight': self.object_energy_weight}
        base_config = super(LLFullOCThresholds, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss(self, inputs):
        assert len(inputs) == 9
        beta, ccoords, d, c_idx, ch_idx, t_idx, rs, t_d, t_b = inputs
        
        

     
class LLFullOCThresholds(LossLayerBase):
    def __init__(self, 
                 purity_weight = 0.5,
                 object_energy_weight = 'modlog',
                 **kwargs):
        '''
        creates gradients for t_d and t_b
        Inputs: a lot
        
        # betas: V x 1
        # coords: V x C
        # d: V x 1
        # c_idx: ragged to select condensation points
        # ch_idx: ragged to select condensation points and their hits
        # energy: hit energy, V x 1
        # t_idx: V x 1
        # t_depe: true deposited energy
        # rs: rs
        # t_d: ()
        # t_b: ()
        
        Output: pass-through of betas
        '''
        
        assert 0. < purity_weight < 1.
        self.purity_weight = purity_weight
        
        assert object_energy_weight == 'log' or object_energy_weight == 'lin'\
          or object_energy_weight == 'modlog' or object_energy_weight == 'none'
        if object_energy_weight == 'log':
            self.oweight = self._e_log
        if object_energy_weight == 'lin':
            self.oweight = self._e_lin
        if object_energy_weight == 'none':
            self.oweight = self._e_none
        if object_energy_weight == 'modlog':
            self.oweight = self._e_modlog
        self.object_energy_weight = object_energy_weight # to save it
        
        super(LLFullOCThresholds, self).__init__(**kwargs)
        
    def get_config(self):
        config={'purity_weight': self.purity_weight,
                'object_energy_weight': self.object_energy_weight}
        base_config = super(LLFullOCThresholds, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def _e_log(self, e):
        return tf.math.log1p(e + 1e-6)
    def _e_modlog(self, e):
        return tf.math.log1p(e + 1e-6)**2
    def _e_lin(self, e):
        return e
    def _e_none(self, e):
        return tf.ones_like(e)
    
    def loss(self, inputs):
        assert len(inputs) == 10
        # betas: V x 1
        # coords: V x C
        # d: V x 1
        # c_idx: ragged to select condensation points
        # ch_idx: ragged to select condensation points and their hits
        # energy: hit energy, V x 1
        # t_idx: V x 1
        # t_d: ()
        # t_b: ()
        betas, coords, d, c_idx, ch_idx, energy, t_idx, t_depe, t_d, t_b = inputs
        
        t_d, t_b = t_d[0], t_b[0] #remove extra dim
        
        print(f't_d {t_d}, t_b {t_b}')
        
        # distance from cp to hits:
        ch_coords = tf.gather_nd(coords, ch_idx) # [e, r, r, C]
        c_coords = tf.gather_nd(coords, c_idx)
        ch_dsq = (tf.expand_dims(c_coords,axis=2) - ch_coords)**2
        ch_dsq = tf.reduce_sum(ch_dsq, axis=-1, keepdims=True) # [e, r, r, 1]
        ch_d_n = tf.sqrt(ch_dsq + 1e-4)
        ch_d_n /= tf.expand_dims(tf.gather_nd(d, c_idx),axis=2) # [e, r, r, 1], normalised to distance scaler
        
        ch_energy = tf.gather_nd(energy, ch_idx)
        
        #create same tidx map for purity
        ch_t_idx = tf.gather_nd(t_idx, ch_idx)
        c_t_idx = tf.gather_nd(t_idx, c_idx)
        
        ch_noitnoise = tf.cast(ch_t_idx >= 0, 'float32')
        
        #this is still int

        ch_pure = tf.where(tf.expand_dims(c_t_idx,axis=2) == ch_t_idx, 
                           tf.ones_like(ch_t_idx), tf.zeros_like(ch_t_idx) )
        ch_pure = ch_noitnoise * tf.cast(ch_pure, 'float32') # noise never pure
        
        
        #now weight by distance w.r.t. threshold and energy
        #maybe also threshold the denominator..?
        t_h_p = ch_energy*ch_pure
        t_h_p = _thresh(t_h_p, ch_d_n, t_d)
        c_purity = tf.reduce_sum(t_h_p, axis=2)\
         / (tf.reduce_sum(ch_energy, axis=2) + 1e-3)
        
        ch_depe = tf.gather_nd(t_depe, ch_idx)
        c_mean_t_depe = tf.reduce_sum(ch_noitnoise * ch_energy * ch_depe, axis=2)# [e, r, 1]
        c_mean_t_depe /= tf.reduce_sum(ch_noitnoise * ch_energy, axis=2) + 1e-3
        
        # expected collected energy
        c_depe = c_mean_t_depe # [e, r, 1]
        c_eff = tf.reduce_sum(_thresh(ch_energy, ch_d_n, t_d), axis=2) /\
           tf.reduce_mean(_thresh(tf.ones_like(ch_energy), ch_d_n, t_d), axis=2)
        
        c_eff /= (c_depe + 1e-3)
        
        #weight both by betas
        c_betas = tf.gather_nd(betas, c_idx) # [e, r, 1]
        
        #object weights, will be automatically small for noise
        c_oweight = self.oweight(c_mean_t_depe)
        c_ones = tf.ones_like(c_purity)
        ow_sum = tf.reduce_sum(c_oweight * _thresh(c_ones, c_betas, t_b,True), axis=1) + 1e-6
        
        #make sure noise does not get included
        
        purity = tf.reduce_sum(c_oweight * _thresh(c_purity, c_betas, t_b,True), axis=1)
        purity /= ow_sum
        purity = tf.reduce_mean(purity)
        efficiency = tf.reduce_mean(c_oweight * _thresh(c_eff, c_betas, t_b,True), axis=1)
        efficiency /= ow_sum
        efficiency = tf.reduce_mean(efficiency)
        
        self.add_prompt_metric(purity, self.name+'_purity')
        self.add_prompt_metric(efficiency, self.name+'_efficiency')
        
        
        loss = self.purity_weight * (1. - purity) + (1.-self.purity_weight)* tf.abs(1. - efficiency)
        
        print(f'purity {purity} efficiency {efficiency} loss {loss}')
        
        return loss

    
class LLOCThresholds(LossLayerBase):
    def __init__(self, 
                 purity_target, 
                 lowest_t_b = 0.01,
                 lowest_t_d = 0.02,
                 highest_t_d = 5.,
                 **kwargs):
        
        self.purity_target = purity_target
        self.lowest_t_b = lowest_t_b
        self.lowest_t_d = lowest_t_d
        self.highest_t_d = highest_t_d
        
        assert highest_t_d > 0.
        assert lowest_t_d > 0.
        assert lowest_t_b >= 0.
        
        if 'dynamic' in kwargs:
            super(LLOCThresholds, self).__init__(**kwargs)
        else:
            super(LLOCThresholds, self).__init__(dynamic=True,**kwargs)
        
    def get_config(self):
        config={'purity_target': self.purity_target,
                'lowest_t_b': self.lowest_t_b,
                'lowest_t_d': self.lowest_t_d,
                'highest_t_d': self.highest_t_d}
        base_config = super(LLOCThresholds, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss(self, inputs):
        
        
        beta, asso_idx, d, x , tidx, t_d, t_b = inputs
        
        t_d, t_b =  t_d[0], t_b[0]
         
        if tf.shape(asso_idx)[0] == None:
            return tf.reduce_mean((1-t_d)**2  + (0.9-t_b)**2)
        
        #stop gradients:
        beta = tf.stop_gradient(beta)
        d = tf.stop_gradient(d)
        x = tf.stop_gradient(x)
        
        #use MSel from OC here
        # now creates a pseudo ragged asso index matrix
        # already takes care of row splits
        Msel,_,_ = CreateMidx(asso_idx, calc_m_not=False)
        
        #print('t_d',t_d, 't_b',t_b)
        #return tf.reduce_mean((1-t_d)**2  + (0.9-t_b)**2)
    
        if Msel is None or tf.shape(Msel)[0] == None:
            return 0.
        
        #OOM safe
        if Msel.shape[1] > 8192:
            Msel = Msel[:,:8192]
        if Msel.shape[0] > 8192: #just select a few
            Msel = Msel[:8192]
        
        beta_k_m = SelectWithDefault(Msel, beta, 0.) #K x V-obj x 1
        mask_k_m = SelectWithDefault(Msel, tf.ones_like(beta), 0.) #K x V-obj x 1
        x_k_m = SelectWithDefault(Msel, x, 0.) #K x V-obj x C
        d_k_m = SelectWithDefault(Msel, d, 0.)
        tidx_k_m = SelectWithDefault(Msel, tidx, -2) #K x V-obj x 1
        
        cluster_size = tf.reduce_sum(mask_k_m,axis=1) #K x 1
        non_singular = tf.cast(cluster_size > 1., 'float32') # K x 1
        
        #print('beta_k_m',beta_k_m.shape)
        #print('x_k_m',x_k_m.shape)
        #print('d_k_m',d_k_m.shape)
        #print('tidx_k_m',tidx_k_m.shape)
        
        alpha_k = tf.argmax(beta_k_m, axis=1)
        x_k = tf.gather_nd(x_k_m, alpha_k, batch_dims=1) #K x C
        d_k = tf.gather_nd(d_k_m, alpha_k, batch_dims=1) #K x 1
        b_k = tf.gather_nd(beta_k_m, alpha_k, batch_dims=1) #K x 1
        tidx_k = tf.gather_nd(tidx_k_m, alpha_k, batch_dims=1) #K x 1
        
        same_k_m = tf.cast( tf.expand_dims(tidx_k, axis=1)-tidx_k_m == 0 , 'float32' )
        #print('same_k_m',same_k_m.shape)
        
        #print('x_k',x_k.shape)
        #print('d_k',d_k.shape)
        #print('b_k',b_k.shape)
        
        dsq_k_m = tf.reduce_sum((tf.expand_dims(x_k, axis=1) - x_k_m)**2, axis=2, keepdims=True) # K x V-obj x 1
        #print('dsq_k_m',dsq_k_m.shape,d_k.shape)
        d_k = tf.expand_dims(d_k, axis=1)
        dsq_k_m = dsq_k_m/d_k**2 # K x V-obj x 1
        
        dsq_k_m = tf.sqrt(dsq_k_m+1e-3)
        # get truth
        
        
        den_k = tf.reduce_sum(_thresh(mask_k_m, dsq_k_m, t_d), axis=1) # K x 1
        p_k = tf.reduce_sum(_thresh(mask_k_m*same_k_m, dsq_k_m, t_d), axis=1) # K x 1
        
        p_k = tf.math.divide_no_nan(p_k, den_k) # K x 1
        
        #print('p_k',tf.reduce_mean(p_k),t_d,t_b)
        
        any_non_singular = tf.reduce_sum(non_singular) > 0.
        #now the same with beta
        p_w = tf.reduce_sum(non_singular * _thresh(p_k, b_k, t_b, True))
        den_w = tf.reduce_sum(non_singular * _thresh(tf.ones_like(p_k), b_k, t_b, True))
        p_w = tf.math.divide_no_nan(p_w, den_w + 1e-2) # ()
        
        p_w = tf.where(any_non_singular, p_w, 1.)
        
        self.add_prompt_metric(p_w, self.name+'_weighted_purity')
        
        mean_cluster_size = tf.reduce_mean(cluster_size)
        self.add_prompt_metric(mean_cluster_size, self.name+'_mean_cluster_size')
        
        #now the loss term, make it such that 0.1 -> 1. to have it on a good footing
        loss_a = tf.reduce_mean(tf.nn.relu(self.purity_target - p_w)**2)
        
        self.add_prompt_metric(loss_a, self.name+'_purity_loss')
        
        #now a generic loss pushing t_b and t_d down/up
        loss_b = 0.01*tf.abs(t_b) + 0.01 * (1. - tf.abs(t_d)/self.highest_t_d)
        loss_b += tf.nn.relu(1e-6 - t_b)*1e7 #keep it positive
        loss_b += tf.nn.relu(self.lowest_t_b - t_b)**2
        loss_b += tf.nn.relu(self.lowest_t_d - t_d)**2
        loss_b += tf.nn.relu(t_d - self.highest_t_d)**2
        
        
        self.add_prompt_metric(loss_b, self.name+'_threshold_loss')
        self.add_prompt_metric(t_b, self.name+'_t_b')
        self.add_prompt_metric(t_d, self.name+'_t_d')
        
        #print('mask_k_m',mask_k_m.shape)
        # just metrics here
        # non-weighted purity for reference
        p_k_t = tf.reduce_sum(mask_k_m*same_k_m, axis=1) / tf.reduce_sum(mask_k_m, axis=1)
        
        #print('p_k_t',p_k_t.shape)
        
        p_k_t = tf.reduce_sum(non_singular * p_k_t) / (tf.reduce_sum(non_singular) + 1e-6)
        p_k_t = tf.where(any_non_singular, p_k_t, 1.)
        self.add_prompt_metric(p_k_t, self.name+'_exact_purity')
        
    
        return loss_a + loss_b
        


class LLFillSpace(LossLayerBase):
    def __init__(self, 
                 maxhits: int=1000,
                 runevery: int=-1,
                 **kwargs):
        '''
        calculated a PCA of all points in coordinate space and 
        penalises very asymmetric PCs. 
        Reduces the risk of falling back to a (hyper)surface
        
        Inputs:
         - coordinates, row splits, (truth index - optional. then only applied to non-noise)
        Outputs:
         - coordinates (unchanged)
        '''
        print('INFO: LLFillSpace: this is actually a regulariser: move to right file soon.')
        assert maxhits>0
        self.maxhits = maxhits
        self.runevery = runevery
        self.counter=-1
        if runevery < 0:
            self.counter = -2
        if 'dynamic' in kwargs:
            super(LLFillSpace, self).__init__(**kwargs)
        else:
            super(LLFillSpace, self).__init__(dynamic=True,**kwargs)
        
    
    def get_config(self):
        config={'maxhits': self.maxhits,
                'runevery': self.runevery }
        base_config = super(LLFillSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @staticmethod
    def _rs_loop(coords, tidx, maxhits=1000):
        #only select a few hits to keep memory managable
        nhits = coords.shape[0]
        sel = None
        if nhits > maxhits:
            sel = tf.random.uniform(shape=(maxhits,), minval=0, maxval=coords.shape[0]-1, dtype=tf.int32)
        else:
            sel = tf.range(coords.shape[0], dtype=tf.int32)
        sel = tf.expand_dims(sel,axis=1)
        coords = tf.gather_nd(coords, sel) # V' x C
        if tidx is not None:
            tidx = tf.gather_nd(tidx, sel) # V' x C
            coords = coords[tidx[:,0] >= 0]
        #print('coords',coords.shape)
        means = tf.reduce_mean(coords,axis=0,keepdims=True)#1 x C
        coords -= means # V' x C
        #build covariance
        cov = tf.expand_dims(coords,axis=1)*tf.expand_dims(coords,axis=2)#V' x C x C
        cov = tf.reduce_mean(cov,axis=0,keepdims=False)# 1 x C x C
        #print('cov',cov)
        #get eigenvals
        eigenvals,_ = tf.linalg.eig(cov)#cheap because just once, no need for approx
        eigenvals = tf.cast(eigenvals, dtype='float32')
        #penalise one small EV (e.g. when building a surface)
        pen = tf.math.log((tf.math.divide_no_nan(tf.reduce_mean(eigenvals), 
                                    tf.reduce_min(eigenvals)+1e-6) - 1.)**2+1.)
        return pen
        
        
    @staticmethod
    def raw_loss(coords, rs, tidx, maxhits=1000):
        loss = tf.zeros([], dtype='float32')
        for i in range(len(rs)-1):
            rscoords = coords[rs[i]:rs[i+1]]
            loss += LLFillSpace._rs_loop(rscoords, tidx, maxhits)
        return tf.math.divide_no_nan(loss ,tf.cast(rs.shape[0],dtype='float32'))
    
    def loss(self, inputs):
        assert len(inputs) == 2 or len(inputs) == 3 #coords, rs
        tidx = None
        if len(inputs) == 3:
            coords, rs, tidx = inputs
        else:
            coords, rs = inputs
        if self.counter >= 0: #completely optimise away increment
            if self.counter < self.runevery:
                self.counter+=1
                return tf.zeros_like(coords[0,0])
            self.counter = 0
        lossval = LLFillSpace.raw_loss(coords, rs, tidx, self.maxhits)
        
        if self.counter == -1:
            self.counter+=1
        return lossval
    

class LLEnergySums(LossLayerBase):
    
    def _per_rs(self, energy, t_energy, t_isunique):
        t_energy = t_energy[t_isunique>0]
        sum_t = tf.reduce_sum(t_energy)
        sum_r = tf.reduce_sum(energy)
        return (sum_t-sum_r)**2/(tf.abs(sum_t)+1e-6)**2
    
    def loss(self, inputs):
        energy, is_track, t_idx, t_energy, t_isunique, t_pid, rs = inputs
        
        tot = tf.constant(0.,dtype='float32')
        if rs.shape[0] is None:
            return tot
        
        t_energy = tf.where(t_pid[:,id_str_to_idx['muon']:id_str_to_idx['muon']+1]>0,
                            0.,t_energy)#remove MIPS

        energy = tf.where(t_idx<0,0.,energy) #mask noise
        energy = tf.where(is_track>0,0.,energy) #mask tracks
        
        
        for i in range(len(rs)-1):
            tot += self._per_rs(energy[rs[i]:rs[i+1]], 
                                t_energy[rs[i]:rs[i+1]], 
                                t_isunique[rs[i]:rs[i+1]])
        tot /= tf.cast(rs.shape[0],'float32')
        return tot
    
#naming scheme: LL<what the layer is supposed to do>
class LLClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    Inputs:
    - coordinates
    - truth index
    - row splits
    '''
    def __init__(self, downsample:int = -1, 
                 ignore_noise=False, 
                 hinge_mode = False, 
                 **kwargs):
        if 'dynamic' in kwargs:
            super(LLClusterCoordinates, self).__init__(**kwargs)
        else:
            super(LLClusterCoordinates, self).__init__(dynamic=True,**kwargs)
            
        self.downsample = downsample
        self.ignore_noise = ignore_noise
        self.hinge_mode = hinge_mode
        
        #self.built = True #not necessary for loss layers

    def get_config(self):
        base_config = super(LLClusterCoordinates, self).get_config()
        return dict(list(base_config.items()) + list({'downsample':self.downsample, 
                                                      'ignore_noise': self.ignore_noise,
                                                      'hinge_mode': self.hinge_mode}.items()))
    
    def _attfunc(self, dsq):
        if self.hinge_mode:
            return tf.sqrt(dsq+1e-4)
        return tf.math.log(tf.math.exp(1.)*dsq+1.)
    
    def _rep_func(self,dsq):
        if self.hinge_mode:
            return tf.nn.relu( 1. - tf.sqrt(dsq + 1e-6) )
        
        return tf.exp(-tf.sqrt(dsq+1e-4)/(5.))
             
    #all this needs some cleaning up
    def _rs_loop(self,coords, tidx, specweight, energy):
        
        if tidx.shape[0] == 0:
            print(self.name, 'batch empty')
            return 3 * [self.create_safe_zero_loss(coords)]
        
        Msel, M_not, N_per_obj = CreateMidx(tidx, calc_m_not=True) #N_per_obj: K x 1
        
        if Msel is None or N_per_obj is None:
            print(self.name, 'no objects in batch')
            return 3 * [self.create_safe_zero_loss(coords)]
        
        
        #almost empty
        if Msel.shape[0] == 0 or (Msel.shape[0] == 1 and Msel.shape[1] == 1):
            print(self.name, 'just one point left')
            return 3 * [self.create_safe_zero_loss(coords)]
        
        if self.ignore_noise:
            e_tidx = tf.cast(tf.expand_dims(tidx,axis=0), 'float32')
            e_tidx *= M_not
            M_not = tf.where(e_tidx <0 , 0., M_not)
            
        
        
        N_per_obj = tf.cast(N_per_obj, dtype='float32')
        N_tot = tf.cast(tidx.shape[0], dtype='float32') 
        K = tf.cast(Msel.shape[0], dtype='float32') 
        
        padmask_m = SelectWithDefault(Msel, tf.ones_like(coords[:,0:1]), 0.)# K x V' x 1
        coords_m = SelectWithDefault(Msel, coords, 0.)# K x V' x C
        
        q = (1.-specweight) #*(1.+tf.math.log(tf.abs(energy)+1.))+1e-2
        q_m = SelectWithDefault(Msel, q, 0.)# K x V' x C
        q_k = tf.reduce_sum(q_m,axis=1)#K x 1
        
        #create average
        av_coords_m = tf.reduce_sum(coords_m * padmask_m * q_m,axis=1) # K x C
        av_coords_m = tf.math.divide_no_nan(av_coords_m, 
                                            tf.reduce_sum(padmask_m * q_m,axis=1) + 1e-3) #K x C
        av_coords_m = tf.expand_dims(av_coords_m,axis=1) ##K x 1 x C
        
        distloss = tf.reduce_sum((av_coords_m-coords_m)**2,axis=2)
        distloss = q_m[:,:,0] * self._attfunc(distloss) * padmask_m[:,:,0]
        distloss = tf.math.divide_no_nan(q_k[:,0] * tf.reduce_sum(distloss,axis=1),
                                         N_per_obj[:,0]+1e-3)#K
        distloss = tf.math.divide_no_nan(tf.reduce_sum(distloss),tf.reduce_sum(q_k)+1e-3)
        
        #check if Mnot is empty
        if M_not.shape[0] == 0 or tf.reduce_sum(M_not) == 0.:
            print(self.name, 'no repulsive loss')
            return distloss, distloss, self.create_safe_zero_loss(coords)
        
        repdist = tf.expand_dims(coords, axis=0) - av_coords_m #K x V x C
        repdist = tf.expand_dims(q,axis=0) * tf.reduce_sum(repdist**2,axis=-1,keepdims=True) #K x V x 1
        
        #add a long range part to it
        reploss = M_not * self._rep_func(repdist) #K x V x 1
        #downweight noise
        reploss = q_k * tf.reduce_sum(reploss,axis=1)/( N_tot-N_per_obj +1e-3)#K x 1
        reploss = tf.reduce_sum(reploss)/(tf.reduce_sum(q_k)+1e-3)
        
        return distloss+reploss, distloss, reploss
    
    
    def raw_loss(self,acoords, atidx, aspecw, aenergy, rs, downsample):
        
        lossval = tf.zeros_like(acoords[0,0])
        reploss = tf.zeros_like(acoords[0,0])
        distloss = tf.zeros_like(acoords[0,0])
        
        if rs.shape[0] is None:
            return lossval, distloss, reploss
        
        nbatches = rs.shape[0]-1
        for i in range(nbatches):
            coords = acoords[rs[i]:rs[i+1]]
            tidx = atidx[rs[i]:rs[i+1]]
            specw = aspecw[rs[i]:rs[i+1]]
            energy = aenergy[rs[i]:rs[i+1]]
            
            if downsample>0 and downsample < coords.shape[0]:
                sel = tf.random.uniform(shape=(downsample,), minval=0, maxval=coords.shape[0]-1, dtype=tf.int32)
                sel = tf.expand_dims(sel,axis=1)
                coords = tf.gather_nd(coords, sel)
                tidx = tf.gather_nd(tidx, sel)
                specw = tf.gather_nd(specw, sel)
                energy = tf.gather_nd(energy, sel)
            
            tlv, tdl, trl = self._rs_loop(coords,tidx,specw,energy)
            tlv = tf.where(tf.math.is_finite(tlv), tlv, 0.)
            tdl = tf.where(tf.math.is_finite(tdl), tdl, 0.)
            trl = tf.where(tf.math.is_finite(trl), trl, 0.)
            lossval += tlv
            distloss += tdl
            reploss += trl
        nbatches = tf.cast(nbatches,dtype='float32') + 1e-3
        return lossval/nbatches, distloss/nbatches, reploss/nbatches

    
        
    def loss(self, inputs):
        if len(inputs) == 5:
            coords, tidx, specw, energy, rs = inputs
        elif len(inputs) == 4:
            coords, tidx, energy, rs = inputs
            specw = tf.zeros_like(energy)
        else:
            raise ValueError("LLClusterCoordinates: expects 4 or 5 inputs")
        
        #maybe move this sort of protection to all loss layers: done!
        #moved, so not necessary here anymore
        #if rs.shape[0] is None or coords.shape[0] is None or coords.shape[0] == 0:
        #    zero_loss = 0.*tf.reduce_mean(coords)
        #    zero_loss = tf.where(tf.math.is_finite(zero_loss),zero_loss,0.)
        #    print(self.name,'returning zero loss',zero_loss)
        #    return zero_loss
            
        lossval,distloss, reploss = self.raw_loss(
            coords, tidx, specw, energy, rs, self.downsample)
        
        lossval = tf.where(tf.math.is_finite(lossval), lossval, 0.)#DEBUG
        
        self.add_prompt_metric(self.scale * distloss, self.name+'_att_loss')
        self.add_prompt_metric(self.scale * reploss, self.name+'_rep_loss')
        
        return lossval
    



class LLLocalClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    Inputs: dist, nidxs, tidxs, specweight
    
    Attractive and repulsive potential:
    - Att: log(sqrt(2)*dist**2+1.)
    - Rep: 1/(dist**2+0.1)
    
    The crossing point is at about 1,1.
    The ratio of repulse and attractive potential at 
     - dist**2 = 1. is about 1
     - dist = 0.85, dist**2 = 0.75 is about 2.
     - dist = 0.7, dist**2 = 0.5 is about 3.5
     - dist = 0.5, dist**2 = 0.25 is about 10
    (might be useful for subsequent distance cut-offs)
    
    '''
    def __init__(self, **kwargs):
        
        super(LLLocalClusterCoordinates, self).__init__(**kwargs)
        #if 'dynamic' in kwargs:
        #    super(LLLocalClusterCoordinates, self).__init__(**kwargs)
        #else:
        #    super(LLLocalClusterCoordinates, self).__init__(dynamic=kwargs['print_loss'],**kwargs)
        self.time = time.time()

    @staticmethod
    def raw_loss(dist, nidxs, tidxs, specweight, print_loss, name, scaler = None):
        
        sel_tidxs = SelectWithDefault(nidxs, tidxs, -1)[:,:,0]
        sel_spec = SelectWithDefault(nidxs, specweight, 1.)[:,:,0]
        active = tf.where(nidxs>=0, tf.ones_like(dist), 0.)
        notspecmask = 1. #(1. - 0.5*sel_spec)#only reduce spec #tf.where(sel_spec>0, 0., tf.ones_like(dist))
        
        probe_is_notnoise = tf.cast(tidxs>=0,dtype='float32') [:,0] #V
        notnoisemask = tf.where(sel_tidxs<0, 0., tf.ones_like(dist))
        notnoiseweight = notnoisemask + (1.-notnoisemask)*0.01
        #notspecmask *= notnoisemask#noise can never be spec
        #mask spectators
        sameasprobe = tf.cast(sel_tidxs[:,0:1] == sel_tidxs,dtype='float32')
        #sameasprobe *= notnoisemask #always push away noise, also from each other
        
        #only not noise can be attractive
        attmask = sameasprobe*notspecmask*active
        repmask = (1.-sameasprobe)*notspecmask*active
        
        attr = tf.math.log(tf.math.exp(1.)*dist+1.) * attmask
        rep =  tf.exp(-dist)* repmask * notnoiseweight # 1./(dist+1.) * repmask #2.*tf.exp(-3.16*tf.sqrt(dist+1e-6)) * repmask  #1./(dist+0.1)
        nattneigh = tf.reduce_sum(attmask,axis=1)
        nrepneigh = tf.reduce_sum(repmask,axis=1)
        
        attloss =  probe_is_notnoise * tf.reduce_sum(attr,axis=1) #tf.math.divide_no_nan(tf.reduce_sum(attr,axis=1), nattneigh)#same is always 0
        attloss = tf.math.divide_no_nan(attloss, nattneigh)
        reploss =  probe_is_notnoise * tf.reduce_sum(rep,axis=1) #tf.math.divide_no_nan(tf.reduce_sum(rep,axis=1), nrepneigh)
        reploss = tf.math.divide_no_nan(reploss, nrepneigh)
        #noise does not actively contribute
        lossval = attloss+reploss
        
        if scaler is not None:
            lossval *= scaler
        
        lossval = tf.math.divide_no_nan(tf.reduce_sum(probe_is_notnoise * lossval),tf.reduce_sum(probe_is_notnoise))
        
        if print_loss:
            avattdist = probe_is_notnoise * tf.math.divide_no_nan(tf.reduce_sum(attmask*tf.sqrt(dist),axis=1), nattneigh)
            avattdist = tf.reduce_sum(avattdist)/tf.reduce_sum(probe_is_notnoise)
            
            avrepdist = probe_is_notnoise * tf.math.divide_no_nan(tf.reduce_sum(repmask*tf.sqrt(dist),axis=1), nrepneigh)
            avrepdist = tf.reduce_sum(avrepdist)/tf.reduce_sum(probe_is_notnoise)
            
            if hasattr(lossval, "numpy"):
                print(name, 'loss', lossval.numpy(),
                      'mean att neigh',tf.reduce_mean(nattneigh).numpy(),
                      'mean rep neigh',tf.reduce_mean(nrepneigh).numpy(),
                      'att', tf.reduce_mean(probe_is_notnoise *attloss).numpy(),
                      'rep',tf.reduce_mean(probe_is_notnoise *reploss).numpy(),
                      'dist (same)', avattdist.numpy(),
                      'dist (other)', avrepdist.numpy(),
                      )
            else:
                tf.print(name, 'loss', lossval,
                'mean att neigh',tf.reduce_mean(nattneigh),
                'mean rep neigh',tf.reduce_mean(nrepneigh))
            
                
        return lossval            
         
    def maybe_print_loss(self, lossval):
        pass #overwritten here
         
    def loss(self, inputs):
        scaler = None
        if len(inputs) == 4:
            dist, nidxs, tidxs, specweight = inputs
        if len(inputs) == 5:
            dist, nidxs, tidxs, specweight, scaler = inputs
        return LLLocalClusterCoordinates.raw_loss(dist, nidxs, tidxs, specweight,
                                                  self.print_loss, 
                                                  self.name,scaler = scaler)





class LLNotNoiseClassifier(LossLayerBase):
    
    def __init__(self, purity_metrics_threshold = -1, record_efficiency=False, **kwargs):
        '''
        Inputs:
        - score
        - truth index (ignored if switched off)
        - spectator weights (optional)
        
        Returns:
        - score (unchanged)
        
        '''
        super(LLNotNoiseClassifier, self).__init__(**kwargs)
        self.purity_metrics_threshold = purity_metrics_threshold
        self.record_efficiency = record_efficiency
        
    
    def get_config(self):
        base_config = super(LLNotNoiseClassifier, self).get_config()
        return dict(list(base_config.items()) + list({'purity_metrics_threshold':self.purity_metrics_threshold,
                                                      'record_efficiency': self.record_efficiency}.items()))
        
    @staticmethod
    def raw_loss(score, tidx, specweight, name=""):
        truth = tf.cast(tidx >= 0,dtype='float32')
        classloss = (1.-specweight[:,0]) * tf.keras.losses.binary_crossentropy(truth, score)
        lossval =  tf.reduce_mean(classloss)
        lossval = tf.debugging.check_numerics(lossval, name+" produced inf or nan.")
        return lossval
        
    def loss(self, inputs):
        assert len(inputs) > 1 and len(inputs) < 4
        score, tidx, specweight = None, None, None
        if len(inputs) == 2:
            score, tidx = inputs
            specweight = tf.zeros_like(score)
        else:
            score, tidx, specweight = inputs
        lossval = LLNotNoiseClassifier.raw_loss(score, tidx, specweight, self.name)
        
        isnotnoise = tf.cast(tidx>=0,dtype='float32')
        accuracy = tf.reduce_sum(isnotnoise * tf.cast(score>0.5,dtype='float32'))/tf.reduce_sum(isnotnoise)
        self.add_prompt_metric(accuracy,self.name+'_accuracy')
        
        if self.purity_metrics_threshold > 0:
            isnoise = tf.cast(tidx<0,dtype='float32')
            selnoise = tf.cast(score<self.purity_metrics_threshold,dtype='float32')
            pur = tf.reduce_sum(isnoise * selnoise/tf.reduce_sum(selnoise))
            pur = tf.where(tf.reduce_sum(selnoise) == 0., 1., pur) #define no selection as 1 purity
            self.add_prompt_metric(pur,self.name+'_purity')
            
        if self.record_efficiency:
            redeff = tf.reduce_sum(isnoise * selnoise/tf.reduce_sum(isnoise))
            self.add_prompt_metric(redeff,self.name+'_red_efficiency')
        
        self.maybe_print_loss(lossval)
        return lossval
        

class LLNeighbourhoodClassifier(LossLayerBase):
    def __init__(self, **kwargs):
        '''
        Inputs:
        - score: high means neighbourhood is of same object as first point
        - neighbour indices (ignored if switched off)
        - truth index (ignored if switched off)
        - spectator weights (optional)
        
        Returns:
        - score (unchanged)
        
        '''
        super(LLNeighbourhoodClassifier, self).__init__(**kwargs)
    
    
    @staticmethod
    def raw_loss(score, nidx, tidxs, specweights):
        # score: V x 1
        # nidx: V x K
        # tidxs: V x 1
        # specweight: V x 1
        
        n_tidxs = SelectWithDefault(nidx, tidxs, -1)[:,:,0] # V x K
        tf.assert_equal(tidxs,n_tidxs[:,0:1])#sanity check to make sure the self reference is in the nidxs
        n_tidxs = tf.where(n_tidxs<0,-10,n_tidxs) #set noise to -10
        
        #the actual check
        n_good = tf.cast(n_tidxs==tidxs, dtype='float32')#noise is always bad
        
        #downweight spectators but don't set them to zero
        n_active = tf.where(nidx>=0, tf.ones_like(nidx,dtype='float32'), 0.) # V x K
        truthscore = tf.math.divide_no_nan(
            tf.reduce_sum(n_good,axis=1,keepdims=True),
            tf.reduce_sum(n_active,axis=1,keepdims=True)) #V x 1
        #cut at 90% same
        truthscore = tf.where(truthscore>0.9,1.,truthscore*0.) #V x 1
        
        lossval = tf.keras.losses.binary_crossentropy(truthscore, score)#V
        
        specweights = specweights[:,0]#V
        isnotnoise = tf.cast(tidxs>=0, dtype='float32')[:,0] #V
        obj_lossval = tf.math.divide_no_nan(tf.reduce_sum(specweights*isnotnoise*lossval) , tf.reduce_sum(specweights*isnotnoise))
        noise_lossval = tf.math.divide_no_nan(tf.reduce_sum((1.-isnotnoise)*lossval) , tf.reduce_sum(1.-isnotnoise))
        
        lossval = obj_lossval + 0.1*noise_lossval #noise doesn't really matter so much
    
        return lossval
        
    def loss(self, inputs):
        assert len(inputs) > 2 and len(inputs) < 5
        score, nidx, tidxs, specweights = None, None, None, None
        if len(inputs) == 3:
            score, nidx, tidxs = inputs
            specweights = tf.ones_like(score)
        else:
            score, nidx, tidxs, specweights = inputs
            
        lossval = LLNeighbourhoodClassifier.raw_loss(score, nidx, tidxs, specweights)
        self.maybe_print_loss(lossval)
        return lossval    


                
class LLEdgeClassifier(LossLayerBase):
    
    def __init__(self, downweight_spectators=False, 
                 fp_weight = 0.5, 
                 hifp_penalty= 0.,
                 lin_e_weight = False,
                 **kwargs):
        '''
        Noise hits (truth index < 0) are never classified as belonging together
        
        Inputs:
        - score
        - neighbour index
        - truth index (ignored if switched off)
        - spectator weights (optional)
        - energy (optional)
        
        Returns:
        - score (unchanged)
        '''
        
        assert 0. < fp_weight < 1.
        assert hifp_penalty >= 0.
        
        
        super(LLEdgeClassifier, self).__init__(**kwargs)
        self.downweight_spectators = downweight_spectators
        self.fp_weight = fp_weight
        self.hifp_penalty = hifp_penalty
        self.lin_e_weight = lin_e_weight
     
    
    def get_config(self):
        base_config = super(LLEdgeClassifier, self).get_config()
        return dict(list(base_config.items()) + list({'downweight_spectators':self.downweight_spectators,
                                                      'fp_weight': self.fp_weight,
                                                      'hifp_penalty': self.hifp_penalty,
                                                      'lin_e_weight': self.lin_e_weight}.items()))
            
    @staticmethod
    def raw_loss(score, nidx, tidx, specweight, energy, t_energy, 
                 downweight_spectators=False, fp_weight = .5, hifp_penalty=0.,
                 lin_e_weight=False):
        # nidx = V x K,
        # tidx = V x 1
        # specweight: V x 1
        # score: V x K-1 x 1
        n_tidxs = SelectWithDefault(nidx, tidx, -1)# V x K x 1
        tf.assert_equal(tidx,n_tidxs[:,0]) #check that the nidxs have self-reference
        
        #int32 goes up to -2.something e-9
        n_tidxs = tf.where(n_tidxs<0,-1000000000,n_tidxs)#set to -V for noise
        
        n_active = tf.where(nidx>=0, tf.ones_like(nidx,dtype='float32'), 0.)[:,1:] # V x K-1
        specweight = tf.clip_by_value(specweight,0.,1.)
        energy = tf.clip_by_value(energy,0.,100000.)
        n_specw = SelectWithDefault(nidx, specweight, 1.)[:,1:,0]# V x K-1
        n_energy = energy + SelectWithDefault(nidx, energy, 0.)[:,1:,0]# V x K-1, energy sum of edge
        
        n_weight = (1.+tf.math.log(n_energy+1.))+1e-2
        weight = (1.+tf.math.log(energy+1.))+1e-2
        
        if lin_e_weight:
            weight = _calc_energy_weights(t_energy)
        
        if downweight_spectators:
            n_weight = (1.-n_specw)*n_weight+1e-2
            weight = (1.-specweight)*weight+1e-2
        
        #now this will be false for all noise
        n_sameasprobe = tf.cast(tf.expand_dims(tidx, axis=2) == n_tidxs[:,1:,:], dtype='float32') # V x K-1 x 1
        
        n_mask_same = n_sameasprobe[...,0] * n_active
        n_mask_diff = (1. - n_sameasprobe[...,0] ) * n_active# V x K-1
        #get class weights from here n_sameasprobe is already a mask
        all_sum = tf.reduce_sum(n_active)
        w_all_same =  1. - tf.reduce_sum(n_sameasprobe[...,0] * n_active) / all_sum
        w_all_diff =  1. - w_all_same
        
        n_cl_weight =  2.*(1. - fp_weight) * n_sameasprobe[...,0] * n_active * w_all_same
        n_cl_weight += 2.*fp_weight * (1. - n_sameasprobe[...,0])  * n_active * w_all_diff
        
        #weight high scores more strongly
        score_w = 1. + hifp_penalty * tf.stop_gradient(score)[...,0]**2 
        score_w /= 1. + hifp_penalty
        
        lossval =  tf.keras.losses.binary_crossentropy(n_sameasprobe, score)# V x K-1
        lossval *= score_w
        lossval *= n_active
        lossval *= n_weight #reduce spectators, but don't remove them
        lossval *= 2. * n_cl_weight #class balance
        
        lossval = tf.math.divide_no_nan( tf.reduce_sum(lossval,axis=1), tf.reduce_sum(n_active,axis=1) ) # V 
        lossval *= weight[:,0]#V
        return tf.math.divide_no_nan(tf.reduce_sum(lossval), tf.reduce_sum(weight))
        
    def loss(self, inputs):
        assert len(inputs) == 6 or len(inputs) == 5
        if len(inputs) == 6:
            score, nidx, tidx,specweight,t_energy,energy = inputs
        else:
            score, nidx, tidx,specweight,energy = inputs
            t_energy = tf.ones_like(energy)

        if len(score.shape) < 3:
            score = tf.expand_dims(score, axis=2)# to fit the binary xentr
         
        if nidx.shape[1]  == score.shape[1]: #score should have one less, assume nidx does not contain 'self'
            nidx = tf.concat([tf.range(tf.shape(nidx)[0])[:,tf.newaxis], nidx], axis=1)
            
        lossval = LLEdgeClassifier.raw_loss(score, nidx, tidx, specweight, energy, t_energy,
                                            self.downweight_spectators, 
                                            self.fp_weight, self.hifp_penalty, self.lin_e_weight)
        self.maybe_print_loss(lossval)
        return lossval



class LLGoodNeighbourHood(LossLayerBase):
    
    def __init__(self, sampling = 0.005, distscale = 1., **kwargs):
        
        self.sampling = sampling
        self.distscale = distscale
        super(LLGoodNeighbourHood, self).__init__(**kwargs)
        
    def get_config(self):
        base_config = super(LLGoodNeighbourHood, self).get_config()
        return dict(list(base_config.items()) + list({'sampling':self.sampling,
                                                      'distscale':self.distscale}.items()))
        
    def loss(self, inputs):
        assert len(inputs) == 5
        score, coords, d, t_idx, rs = inputs
        
        coords = tf.stop_gradient(coords)
        d = tf.stop_gradient(d)
        
        d /= self.distscale
        
        zero = tf.reduce_sum(score)*0.
        
        if rs.shape[0] is None:
            return zero
        
        thisloss = zero
        cont = zero
        
        for i in range(len(rs)-1):
            
            rstidx = t_idx[rs[i]:rs[i+1]]
            rsd = d[rs[i]:rs[i+1]]
            rscoords = coords[rs[i]:rs[i+1]]
            rsscore = score[rs[i]:rs[i+1]]
            #create random score with rsscore shape and then tf.boolean_mask(inputs, bl)
            rand = tf.random.uniform(rsscore.shape)
            sel = (rand < self.sampling)[:,0]
            
            scoords = tf.boolean_mask(rscoords, sel)
            sd = tf.boolean_mask(rsd, sel) # S x 1
            sscore =  tf.boolean_mask(rsscore, sel)
            stidx = tf.boolean_mask(rstidx, sel)
            
            #now build a matrix for truth, 0 is the selected samples
            tsame = tf.cast(tf.expand_dims(stidx,axis=1) == tf.expand_dims(rstidx,axis=0), dtype='float32') # S x V x 1
            mdist = (tf.expand_dims(scoords,axis=1) - tf.expand_dims(rscoords,axis=0))**2
            mdist = tf.reduce_sum(mdist, axis=2) / sd**2 # S x V
            dist_weight = tf.exp(-mdist/2.)[:,:,tf.newaxis] # S x V x 1
            
            dist_weight = tf.cast(mdist[:,:,tf.newaxis] < 1., 'float32')
            
            sameprob = tf.math.divide_no_nan(
                tf.reduce_sum(tsame * dist_weight, axis=1), 
                tf.reduce_sum(dist_weight, axis=1) + 1e-3) # S x 1 
            
            print('(sameprob, sscore)',tf.reduce_mean(sameprob))
            bcloss = tf.keras.losses.binary_crossentropy(sameprob, sscore) # S x 1 ?
            #bcloss = 100.*(sameprob - sscore)**2 #simple MSE, but scaled reasonably
            
            #bcloss = tf.reduce_sum( bcloss*dist_weight, axis=1 )
            #bcloss = tf.math.divide_no_nan( bcloss, tf.reduce_sum(dist_weight, axis=1) + 1e-3 ) #S
            
            thisloss += tf.reduce_mean(bcloss)
            cont += tf.reduce_mean(sameprob)
            #reduce mean over S
            
        l = thisloss / (len(rs)-1)
        cont = cont / (len(rs)-1)
        self.add_prompt_metric(l, self.name +'_loss')
        self.add_prompt_metric(1.-cont, self.name +'_av_contamination')
        print(self.name,l,1.-cont)
        return l
            
class LLKnnSimpleObjectCondensation(LossLayerBase):
    
    def __init__(self, 
          K=196,
          purity_threshold = 0.95,
          b_weight=0.5,
          **kwargs):
        
        assert 0. <= b_weight <= 1.
        
        self.K = K
        self.purity_threshold = purity_threshold
        self.b_weight = b_weight
        super(LLKnnSimpleObjectCondensation, self).__init__(**kwargs)
    
    
    def get_config(self):
        base_config = super(LLKnnSimpleObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list({'K':self.K,
                                                      'purity_threshold':self.purity_threshold,
                                                      'b_weight':self.b_weight}.items()))
    
    def same_truth_mask(self, t_idx, idx):
        
        t_idx_k = SelectWithDefault(idx, t_idx, -1) # V x K x 1
        sametidx = t_idx_k[:,0:1] == t_idx_k
        notnoise = t_idx_k >= 0
        sameobj = tf.cast(tf.logical_and(sametidx, notnoise), 'float32')
        notsameobj = tf.cast(tf.logical_or(
            tf.math.logical_not(sametidx),
            tf.math.logical_not(notnoise)), 'float32')
        
        return sameobj, notsameobj
    
    def rep_func(self, dsq):
        return tf.exp(-dsq/(2. * 0.6**2))
    
    def att_func(self, dsq):
        return dsq #1. - tf.exp(-dsq/2.) + 0.01 * tf.sqrt(dsq+1e-3)
    
    def loss(self, inputs):
        assert len(inputs) == 5
        beta, coords, d, t_idx, rs = inputs
        
        idx, distsq = BinnedSelectKnn(self.K, coords, rs)
        
        #needed?
        ncoords = SelectWithDefault(idx, coords, 0.)
        distsq = (ncoords[:,0:1,:]-ncoords)**2
        distsq = tf.reduce_sum(distsq,axis=2)
        distsq = tf.where(idx<0, 0., distsq)
        
        scaled_dsq = tf.math.divide_no_nan(distsq ,  d**2 + 1e-4)
        
        mask_k = SelectWithDefault(idx, 0.*beta + 1., 0.) # V x K x 1
        mask_k_noself = tf.concat([ mask_k[:,0:1]*0., mask_k[:,1:] ], axis=1)

        sameobj, notsameobj = self.same_truth_mask(t_idx, idx) # V x K x 1
        
        #beta part
        
        pweight = self.rep_func(scaled_dsq)
        pweight = tf.stop_gradient(pweight)
        purity = pweight * sameobj[:,:,0] # V x K 
        purity = tf.math.divide_no_nan(tf.reduce_sum(purity,axis=1) , tf.reduce_sum(pweight,axis=1) + 1e-3) #K
        ispure = tf.cast( purity>self.purity_threshold, 'float32' )[:,tf.newaxis]
        
        mean_ispure = tf.reduce_mean(ispure)
        mean_ispure_no_noise = tf.reduce_sum(ispure) / tf.reduce_sum( tf.cast(t_idx>=0,'float32') )
        
        #print('mean_ispure',mean_ispure, mean_ispure_no_noise)
        
        self.add_prompt_metric(mean_ispure, name=self.name+'_truth_purity')
        self.add_prompt_metric(mean_ispure_no_noise, name=self.name+'_truth_nonoise_purity')
        
        #create classification for beta
        b_loss = tf.keras.losses.binary_crossentropy(ispure, beta)
        b_loss = tf.reduce_mean(b_loss)
        #print('beta_k',beta_k.shape)
        
        att = sameobj * mask_k_noself * self.att_func(scaled_dsq)[...,tf.newaxis]
        rep = notsameobj * mask_k_noself * self.rep_func(scaled_dsq)[...,tf.newaxis]
        
        
        att_loss = tf.reduce_mean(tf.reduce_sum(att, axis=1) / tf.reduce_sum(mask_k_noself, axis=1) + 1e-3) 
        rep_loss = tf.reduce_mean(tf.reduce_sum(rep, axis=1) / tf.reduce_sum(mask_k_noself, axis=1) + 1e-3)
        
        
        self.add_prompt_metric(b_loss, name=self.name+'_beta_loss')
        self.add_prompt_metric(att_loss, name=self.name+'_att_loss')
        self.add_prompt_metric(rep_loss, name=self.name+'_rep_loss')
        
        return self.b_weight * b_loss + (1. - self.b_weight) * (att_loss+rep_loss)
        
                
class LLKnnPushPullObjectCondensation(LossLayerBase):
    
    def __init__(self, 
          q_min=0.8, 
          s_b=1.,
          K=64,
          mode='doubleexp',
          **kwargs):
        
        self.q_min = q_min
        self.s_b = s_b
        self.K = K
        self.mode = mode
        
        assert mode == 'doubleexp' or mode == 'dippedsq'
        
        super(LLKnnPushPullObjectCondensation, self).__init__(**kwargs)
        
    def rep_potential(self,dsq):
        return tf.exp( - dsq/2.)
    
    def att_potential(self,dsq):
        if self.mode == 'doubleexp':
            return -tf.exp( - dsq/(2.*0.5**2))
        if self.mode == 'dippedsq':
            return dsq-1.
        else:
            raise ValueError("mode " + self.mode + " not implemented")
        
    def get_config(self):
        base_config = super(LLKnnPushPullObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list({'q_min':self.q_min,
                                                      's_b':self.s_b,
                                                      'K':self.K,
                                                      'mode':self.mode,}.items()))
                                                              
    def beta_loss(self, beta_ka):
        return tf.reduce_sum(beta_ka * 0.)
        
    def loss(self, inputs):
        assert len(inputs) == 5
        beta, coords, d, t_idx, rs = inputs
        
        idx, distsq = BinnedSelectKnn(self.K, coords, rs)
        
        scaled_dsq = tf.math.divide_no_nan(distsq ,  d**2 + 1e-4)  # V x K, [:,0] is self
        
        beta_k = SelectWithDefault(idx, beta, 0.) # V x K x 1
        mask_k = SelectWithDefault(idx, 0.*beta + 1., 0.) # V x K x 1
        mask_k_noself = tf.concat([ mask_k[:,0:1]*0., mask_k[:,1:] ], axis=1)
        q_k = mask_k * (tf.math.atanh(beta_k/1.002)**2 + self.q_min)
        t_idx_k = SelectWithDefault(idx, t_idx, -1) # V x K x 1
        
        #print('beta_k',beta_k.shape)
        
        sametidx = t_idx_k[:,0:1] == t_idx_k
        notnoise = t_idx_k >= 0
        sameobj = tf.cast(tf.logical_and(sametidx, notnoise), 'float32')
        notsameobj = tf.cast(tf.logical_or(
            tf.math.logical_not(sametidx),
            tf.math.logical_not(notnoise)), 'float32')
        
        #print('sameobj',sameobj.shape, sameobj)
        #print('notsameobj',notsameobj.shape, notsameobj)
        attscaler = (beta_k[:,0:1] + self.q_min) * (beta_k + self.q_min)
        repscaler = q_k[:,0:1] * q_k
        att = sameobj * mask_k_noself * attscaler * self.att_potential(scaled_dsq)[...,tf.newaxis]
        rep = notsameobj * mask_k_noself * repscaler * self.rep_potential(scaled_dsq)[...,tf.newaxis]
        
        att_loss = tf.reduce_mean(tf.reduce_sum(att, axis=1) / tf.reduce_sum(mask_k_noself, axis=1) + 1e-3) 
        rep_loss = tf.reduce_mean(tf.reduce_sum(rep, axis=1) / tf.reduce_sum(mask_k_noself, axis=1) + 1e-3)
        
        isnoise = tf.cast(t_idx < 0, 'float32')
        noise_loss = self.s_b * tf.math.divide_no_nan(tf.reduce_sum(beta * isnoise) , tf.reduce_sum(isnoise)+1e-3)
        
        
        self.add_prompt_metric(tf.reduce_mean(beta), name=self.name+'_av_beta')
        
        self.add_prompt_metric(att_loss, name=self.name+'_att_loss')
        self.add_prompt_metric(rep_loss, name=self.name+'_rep_loss')
        self.add_prompt_metric(noise_loss, name=self.name+'_noise_loss')
        
        return att_loss + rep_loss + noise_loss
        

class LLBasicObjectCondensation(LossLayerBase):
    
    def __init__(self, 
                 q_min=0.8, 
                 s_b=1.,
                 use_average_cc_pos=0.5,
                 implementation = 'std',
                 **kwargs):
        
        assert implementation == 'std' or \
               implementation == 'pushpull' or \
               implementation == 'precond' 
        
        super(LLBasicObjectCondensation, self).__init__(**kwargs)
        
        from object_condensation import Basic_OC_per_sample, PushPull_OC_per_sample, PreCond_kNNOC_per_sample, PreCond_OC_per_sample
        impl = Basic_OC_per_sample
        if implementation == 'pushpull':
            impl = PushPull_OC_per_sample
        if implementation == 'precond':
            impl = PreCond_OC_per_sample
        
        self.oc_loss_object = OC_loss(
            loss_impl = impl,
            q_min= q_min,
                 s_b=s_b,
                 use_mean_x=use_average_cc_pos,
                 spect_supp=1.
            )
        
    
    def loss(self, inputs):
        assert len(inputs) == 6
        beta, coords, d, spec, t_idx, rs = inputs
        
        spec = tf.clip_by_value(spec, 0., 1.)
        
        att, rep, noise, min_b, _, _ = self.oc_loss_object(
                beta=beta,
                x=coords,
                d=d,
                pll=tf.zeros_like(beta),
                truth_idx=t_idx,
                object_weight=tf.ones_like(beta),
                is_spectator_weight=spec,
                rs=rs)

        self.add_prompt_metric(att,self.name+'_att_loss')
        self.add_prompt_metric(rep,self.name+'_rep_loss')
        self.add_prompt_metric(noise,self.name+'_noise_loss')
        self.add_prompt_metric(min_b,self.name+'_min_b_loss')
        
        
        loss = att + rep + noise + min_b
        
        self.add_prompt_metric(loss,self.name+'_loss')
        
        
        self.add_prompt_metric(tf.reduce_mean(d),self.name+'_avg_dist')
        
        return loss
        


class LLFullObjectCondensation(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    
    This is a copy of the above, reducing the nested function calls.
    
    keep the individual loss definitions as separate functions, even if they are trivial.
    inherit from this class to implement different variants of the loss ingredients without
    making the config explode (more)
    '''

    def __init__(self, *, energy_loss_weight=1., 
                 use_energy_weights=True, 
                 alt_energy_weight=False,
                 train_energy_correction=True,
                 q_min=0.1, no_beta_norm=False,
                 noise_q_min=None,
                 potential_scaling=1., repulsion_scaling=1., s_b=1., position_loss_weight=1.,
                 classification_loss_weight=1., timing_loss_weight=1., use_spectators=True, beta_loss_scale=1.,
                 use_average_cc_pos=0.,
                  payload_rel_threshold=0.1, rel_energy_mse=False, smooth_rep_loss=False,
                 pre_train=False, huber_energy_scale=-1., downweight_low_energy=True, n_ccoords=2, energy_den_offset=1.,
                 noise_scaler=1., too_much_beta_scale=0., cont_beta_loss=False, log_energy=False, n_classes=0,
                 prob_repulsion=False,
                 phase_transition=0.,
                 phase_transition_double_weight=False,
                 alt_potential_norm=True,
                 payload_beta_gradient_damping_strength=0.,
                 payload_beta_clip=0.,
                 kalpha_damping_strength=0.,
                 cc_damping_strength=0.,
                 standard_configuration=None,
                 beta_gradient_damping=0.,
                 alt_energy_loss=False,
                 repulsion_q_min=-1.,
                 super_repulsion=False,
                 use_local_distances=True,
                 energy_weighted_qmin=False,
                 super_attraction=False,
                 div_repulsion=False,
                 dynamic_payload_scaling_onset=-0.005,
                 beta_push=0.,
                 **kwargs):
        """
        Read carefully before changing parameters

        :param energy_loss_weight:
        :param use_energy_weights:
        :param q_min:
        :param no_beta_norm:
        :param potential_scaling:
        :param repulsion_scaling:
        :param s_b:
        :param position_loss_weight:
        :param classification_loss_weight:
        :param timing_loss_weight:
        :param use_spectators:
        :param beta_loss_scale:
        :param use_average_cc_pos: weight (between 0 and 1) of the average position vs. the kalpha position 
        :param payload_rel_threshold:
        :param rel_energy_mse:
        :param smooth_rep_loss:
        :param pre_train:
        :param huber_energy_scale:
        :param downweight_low_energy:
        :param n_ccoords:
        :param energy_den_offset:
        :param noise_scaler:
        :param too_much_beta_scale:
        :param cont_beta_loss:
        :param log_energy:
        :param n_classes: give the real number of classes, in the truth labelling, class 0 is always ignored so if you
                          have 6 classes, label them from 1 to 6 not 0 to 5. If n_classes is 0, no classification loss
                          is applied
        :param prob_repulsion
        :param phase_transition
        :param standard_configuration:
        :param alt_energy_loss: introduces energy loss with very mild gradient for large delta. (modified 1-exp form)
        :param dynamic_payload_scaling_onset: only apply payload loss to well reconstructed showers. typical values 0.1 (negative=off)
        :param kwargs:
        """
        if 'dynamic' in kwargs:
            super(LLFullObjectCondensation, self).__init__(**kwargs)
        else:
            super(LLFullObjectCondensation, self).__init__(dynamic=True,**kwargs)
            
        assert use_local_distances #fixed now, if they should not be used, pass 1s
        
        if too_much_beta_scale==0 and cont_beta_loss:
            raise ValueError("cont_beta_loss must be used with too_much_beta_scale>0")
        
        if huber_energy_scale>0 and alt_energy_loss:
            raise ValueError("huber_energy_scale>0 and alt_energy_loss exclude each other")
        
        #configuration here, no need for all that stuff below 
        #as far as the OC part is concerned (still config for payload though)
        self.oc_loss_object = OC_loss(
            q_min= q_min,
                 s_b=s_b,
                 use_mean_x=use_average_cc_pos,
                 spect_supp=1.
            )
        #### the latter needs to be cleaned up

        self.energy_loss_weight = energy_loss_weight
        self.use_energy_weights = use_energy_weights
        self.train_energy_correction = train_energy_correction
        self.q_min = q_min
        self.noise_q_min = noise_q_min
        self.no_beta_norm = no_beta_norm
        self.potential_scaling = potential_scaling
        self.repulsion_scaling = repulsion_scaling
        self.s_b = s_b
        self.position_loss_weight = position_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.timing_loss_weight = timing_loss_weight
        self.use_spectators = use_spectators
        self.beta_loss_scale = beta_loss_scale
        self.use_average_cc_pos = use_average_cc_pos
        self.payload_rel_threshold = payload_rel_threshold
        self.rel_energy_mse = rel_energy_mse
        self.smooth_rep_loss = smooth_rep_loss
        self.pre_train = pre_train
        self.huber_energy_scale = huber_energy_scale
        self.downweight_low_energy = downweight_low_energy
        self.n_ccoords = n_ccoords
        self.energy_den_offset = energy_den_offset
        self.noise_scaler = noise_scaler
        self.too_much_beta_scale = too_much_beta_scale
        self.cont_beta_loss = cont_beta_loss
        self.log_energy = log_energy
        self.n_classes = n_classes
        self.prob_repulsion = prob_repulsion
        self.phase_transition = phase_transition
        self.phase_transition_double_weight = phase_transition_double_weight
        self.alt_potential_norm = alt_potential_norm
        self.payload_beta_gradient_damping_strength = payload_beta_gradient_damping_strength
        self.payload_beta_clip = payload_beta_clip
        self.kalpha_damping_strength = kalpha_damping_strength
        self.cc_damping_strength = cc_damping_strength
        self.beta_gradient_damping=beta_gradient_damping
        self.alt_energy_loss=alt_energy_loss
        self.repulsion_q_min=repulsion_q_min
        self.super_repulsion=super_repulsion
        self.use_local_distances = use_local_distances
        self.energy_weighted_qmin=energy_weighted_qmin
        self.super_attraction = super_attraction
        self.div_repulsion=div_repulsion
        self.dynamic_payload_scaling_onset = dynamic_payload_scaling_onset
        self.alt_energy_weight = alt_energy_weight
        self.loc_time=time.time()
        self.call_count=0
        self.beta_push = beta_push
        
        assert kalpha_damping_strength >= 0. and kalpha_damping_strength <= 1.

        if standard_configuration is not None:
            raise NotImplemented('Not implemented yet')
        
        
    def calc_energy_weights(self, t_energy, t_pid=None, upmouns = True):
        return _calc_energy_weights(t_energy, t_pid, upmouns, self.alt_energy_weight)
        
        
    
    def softclip(self, toclip, startclipat,softness=0.1):
        assert softness>0 and softness < 1.
        toclip /= startclipat
        soft = softness*tf.math.log((toclip-(1.-softness))/softness)+1.
        toclip = tf.where(toclip>1, soft , toclip)
        toclip *= startclipat
        return toclip
        
    def calc_energy_correction_factor_loss(self, t_energy, t_dep_energies, 
                                           pred_energy,pred_energy_low_quantile,pred_energy_high_quantile,
                                           return_concat=False): 
        
        ediff = (t_energy - pred_energy*t_dep_energies)/tf.sqrt(tf.abs(t_energy)+1e-3)
        
        ediff = tf.debugging.check_numerics(ediff, "eloss ediff")
        
        eloss = None
        if self.huber_energy_scale>0:
            eloss = huber(ediff, self.huber_energy_scale)
        else:
            eloss = tf.math.log(ediff**2 + 1. + 1e-5)
            
        #eloss = self.softclip(eloss, 0.4) 
        t_energy = tf.clip_by_value(t_energy,0.,1e12)
        t_dep_energies = tf.clip_by_value(t_dep_energies,0.,1e12)

        #calculate energy quantiles 
        
        #do not propagate the gradient for quantiles further up
        pred_energy = tf.stop_gradient(pred_energy)
        
        corrtruth = tf.math.divide_no_nan(t_energy, t_dep_energies+1e-3)
        
        print('corrtruth',tf.reduce_mean(corrtruth))
        corrtruth = tf.where(corrtruth>5.,5.,corrtruth)#remove outliers
        corrtruth = tf.where(corrtruth<.2,.2,corrtruth)
        resolution =(1-pred_energy/corrtruth) 
        l_low =  resolution - pred_energy_low_quantile
        l_high = resolution - pred_energy_high_quantile
        low_energy_tau = 0.16
        high_energy_tau = 0.84
        euncloss = quantile(l_low,low_energy_tau) + quantile(l_high,high_energy_tau)

        euncloss = tf.debugging.check_numerics(euncloss, "euncloss loss")
        eloss = tf.debugging.check_numerics(eloss, "eloss loss")
        
        if return_concat:
            return tf.concat([eloss, euncloss], axis=-1) # for ragged map flat values
        
        return eloss, euncloss
    
            
    def calc_energy_loss(self, t_energy, pred_energy): 
        
        #FIXME: this is just for debugging
        #return (t_energy-pred_energy)**2
        eloss=0
        
        t_energy = tf.clip_by_value(t_energy,1e-4,1e12)
        
        if self.huber_energy_scale > 0:
            l = tf.abs(t_energy-pred_energy)
            sqrt_t_e = tf.sqrt(t_energy+1e-3)
            l = tf.math.divide_no_nan(l, tf.sqrt(t_energy+1e-3) + self.energy_den_offset)
            eloss = huber(l, sqrt_t_e*self.huber_energy_scale)
        elif self.alt_energy_loss:
            ediff = (t_energy-pred_energy)
            l = tf.math.log(ediff**2/(t_energy+1e-3) + 1.)
            eloss = l
        else:
            eloss = tf.math.divide_no_nan((t_energy-pred_energy)**2,(t_energy + 1e-3))
        
        eloss = self.softclip(eloss, 0.2) 
        eloss = tf.debugging.check_numerics(eloss, "eloss loss")
        return eloss

    def calc_qmin_weight(self, hitenergy):
        if not self.energy_weighted_qmin:
            return self.q_min
        
    
    def calc_position_loss(self, t_pos, pred_pos):
        if tf.shape(pred_pos)[-1] == 2:#also has z component, but don't use it here
            t_pos = t_pos[:,0:2]
        if not self.position_loss_weight:
            return 0.*tf.reduce_sum((pred_pos-t_pos)**2,axis=-1, keepdims=True)
        #reduce risk of NaNs
        ploss = huber(tf.sqrt(tf.reduce_sum((t_pos-pred_pos) ** 2, axis=-1, keepdims=True)/(10**2) + 1e-2), 10.) #is in cm
        ploss = tf.debugging.check_numerics(ploss, "ploss loss")
        return ploss #self.softclip(ploss, 3.) 
    
    def calc_timing_loss(self, t_time, pred_time, pred_time_unc, t_dep_energy=None):
        if  self.timing_loss_weight==0.:
            return pred_time**2 + pred_time_unc**2
        
        pred_time_unc = tf.nn.relu(pred_time_unc)#safety
        
        tloss = tf.math.divide_no_nan((t_time - pred_time)**2 , (pred_time_unc**2 + 1e-1)) + pred_time_unc**2
        tloss = tf.debugging.check_numerics(tloss, "tloss loss")
        if t_dep_energy is not None:
            tloss = tf.where( t_dep_energy < 1., 0.,  tloss)
            
        return tloss
    
    def calc_classification_loss(self, orig_t_pid, pred_id, t_is_unique=None, hasunique=None):
        
        if self.classification_loss_weight <= 0:
            return tf.reduce_mean(pred_id,axis=1, keepdims=True)
        
        pred_id = tf.clip_by_value(pred_id, 1e-9, 1. - 1e-9)
        t_pid = tf.clip_by_value(orig_t_pid, 1e-9, 1. - 1e-9)
        classloss = tf.keras.losses.categorical_crossentropy(t_pid, pred_id)
        classloss = tf.where( orig_t_pid[:,-1]>0. , 0., classloss)#remove ambiguous, last class flag
        
        #take out undefined
        classloss = tf.where( tf.reduce_sum(t_pid,axis=1)>1. , 0., classloss)
        classloss = tf.where( tf.reduce_sum(t_pid,axis=1)<1.-1e-3 , 0., classloss)
        
        classloss = tf.debugging.check_numerics(classloss, "classloss")
        
        return classloss[...,tf.newaxis] # self.softclip(classloss[...,tf.newaxis], 2.)#for high weights
    

    def calc_beta_push(self, betas, tidx):
        if self.beta_push <=0. :
            return tf.reduce_mean(betas*0.)#dummy
        
        nonoise = tf.where(tidx>=0, betas*0.+ 1., betas*0.)
        nnonoise = tf.reduce_sum(nonoise)
        bpush = tf.nn.relu(self.beta_push - betas)
        bpush = tf.math.log( bpush/self.beta_push + 0.1)**2
        bsum = tf.reduce_sum(bpush*nonoise)
        l = tf.math.divide_no_nan(bsum, nnonoise+1e-2)
        l = tf.debugging.check_numerics(l, "calc_beta_push loss")
        return l #goes up to 0.1

    def loss(self, inputs):
        
        assert len(inputs)==21 or len(inputs)==20 
        hasunique = False
        if len(inputs) == 21:
            pred_beta, pred_ccoords, pred_distscale,\
            pred_energy, pred_energy_low_quantile,pred_energy_high_quantile,\
            pred_pos, pred_time, pred_time_unc, pred_id,\
            rechit_energy,\
            t_idx, t_energy, t_pos, t_time, t_pid, t_spectator_weights,t_fully_contained,t_rec_energy,\
            t_is_unique,\
            rowsplits = inputs
            hasunique=True
        elif len(inputs) == 20:
            pred_beta, pred_ccoords, pred_distscale,\
            pred_energy, pred_energy_low_quantile,pred_energy_high_quantile,\
            pred_pos, pred_time, pred_time_unc, pred_id,\
            rechit_energy,\
            t_idx, t_energy, t_pos, t_time, t_pid, t_spectator_weights,t_fully_contained,t_rec_energy,\
            rowsplits = inputs
            
            t_is_unique = tf.concat([t_idx[0:1]*0 + 1, t_idx[1:]*0],axis=0)
            hasunique=False
            print('WARNING. functions using unique will not work as expected')        
        
            #guard
            
        
        if rowsplits.shape[0] is None:
            return tf.constant(0,dtype='float32')
        
        energy_weights = self.calc_energy_weights(t_energy,t_pid)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights)+1.
            
        #reduce weight on not fully contained showers
        energy_weights = tf.where(t_fully_contained>0, energy_weights, energy_weights*0.01)
        
            
        #also kill any gradients for zero weight
        energy_loss,energy_quantiles_loss = None,None        
        if self.train_energy_correction:
            energy_loss,energy_quantiles_loss = self.calc_energy_correction_factor_loss(t_energy,t_rec_energy,pred_energy,pred_energy_low_quantile,pred_energy_high_quantile)
            energy_loss *= self.energy_loss_weight 
        else:
            energy_loss = self.energy_loss_weight * self.calc_energy_loss(t_energy, pred_energy)
            _, energy_quantiles_loss =  self.calc_energy_correction_factor_loss(t_energy,t_rec_energy,pred_energy,pred_energy_low_quantile,pred_energy_high_quantile)
        energy_quantiles_loss *= self.energy_loss_weight/2. 

        position_loss = self.position_loss_weight * self.calc_position_loss(t_pos, pred_pos)
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(t_time, pred_time, pred_time_unc,t_rec_energy)
        classification_loss = self.classification_loss_weight * self.calc_classification_loss(t_pid, pred_id, t_is_unique, hasunique)
        
        ##just for time metrics
        tdiff = (t_time-pred_time)
        tdiff -= tf.reduce_mean(tdiff,keepdims=True)
        tstd = tf.math.reduce_std(tdiff)
        self.add_prompt_metric(tstd,self.name+'_time_std')
        self.add_prompt_metric(tf.reduce_mean(pred_time_unc),self.name+'_time_pred_std')
        #end just for metrics
        
        full_payload = tf.concat([energy_loss,position_loss,timing_loss,classification_loss,energy_quantiles_loss], axis=-1)
        
        if self.payload_beta_clip > 0:
            full_payload = tf.where(pred_beta<self.payload_beta_clip, 0., full_payload)
            #clip not weight, so there is no gradient to push below threshold!
        
        is_spectator = t_spectator_weights #not used right now, and likely never again (if the truth remains ok)
        if is_spectator is None:
            is_spectator = tf.zeros_like(pred_beta)
        
        full_payload = tf.debugging.check_numerics(full_payload,"full_payload has nans of infs")
        pred_ccoords = tf.debugging.check_numerics(pred_ccoords,"pred_ccoords has nans of infs")
        energy_weights = tf.debugging.check_numerics(energy_weights,"energy_weights has nans of infs")
        pred_beta = tf.debugging.check_numerics(pred_beta,"beta has nans of infs")
        #safe guards
        with tf.control_dependencies(
            [tf.assert_equal(rowsplits[-1], pred_beta.shape[0]),
             
             tf.assert_equal(pred_beta>=0., True),
             tf.assert_equal(pred_beta<=1., True),
             
             tf.assert_equal(is_spectator<=1., True),
             tf.assert_equal(is_spectator>=0., True)]):
            
            #att, rep, noise, min_b, payload, exceed_beta = oc_loss(
            #                               x=pred_ccoords,
            #                               beta=pred_beta,
            #                               truth_indices=t_idx,
            #                               row_splits=rowsplits,
            #                               is_spectator=is_spectator,
            #                               payload_loss=full_payload,
            #                               Q_MIN=q_min,
            #                               S_B=self.s_b,
            #                               noise_q_min = self.noise_q_min,
            #                               distance_scale=pred_distscale,
            #                               energyweights=energy_weights,
            #                               use_average_cc_pos=self.use_average_cc_pos,
            #                               payload_rel_threshold=self.payload_rel_threshold,
            #                               cont_beta_loss=self.cont_beta_loss,
            #                               prob_repulsion=self.prob_repulsion,
            #                               phase_transition=self.phase_transition>0. ,
            #                               phase_transition_double_weight = self.phase_transition_double_weight,
            #                               #removed
            #                               #alt_potential_norm=self.alt_potential_norm,
            #                               payload_beta_gradient_damping_strength=self.payload_beta_gradient_damping_strength,
            #                               kalpha_damping_strength = self.kalpha_damping_strength,
            #                               beta_gradient_damping=self.beta_gradient_damping,
            #                               repulsion_q_min=self.repulsion_q_min,
            #                               super_repulsion=self.super_repulsion,
            #                               super_attraction = self.super_attraction,
            #                               div_repulsion = self.div_repulsion,
            #                               dynamic_payload_scaling_onset=self.dynamic_payload_scaling_onset
            #                               )
            att, rep, noise, min_b, payload, exceed_beta = self.oc_loss_object(
                beta=pred_beta,
                x=pred_ccoords,
                d=pred_distscale,
                pll=full_payload,
                truth_idx=t_idx,
                object_weight=energy_weights,
                is_spectator_weight=is_spectator,
                rs=rowsplits)

        self.add_prompt_metric(att+rep,self.name+'_dynamic_payload_scaling')
        
        att *= self.potential_scaling
        rep *= self.potential_scaling * self.repulsion_scaling
        min_b *= self.beta_loss_scale
        noise *= self.noise_scaler
        exceed_beta *= self.too_much_beta_scale

        #unscaled should be well in range < 1.
        #att = self.softclip(att, self.potential_scaling) 
        #rep = self.softclip(rep, self.potential_scaling * self.repulsion_scaling) 
        #min_b = self.softclip(min_b, 5.)  # not needed, limited anyway
        #noise = self.softclip(noise, 5.)  # not needed limited to 1 anyway
        
        
        energy_loss = payload[0]
        pos_loss    = payload[1]
        time_loss   = payload[2]
        class_loss  = payload[3]
        energy_unc_loss  = payload[4]
        
        
        #explicit cc damping
        ccdamp = self.cc_damping_strength * (0.02*tf.reduce_mean(pred_ccoords))**4# gently keep them around 0
        
        
        lossval = att + rep + min_b + noise + energy_loss + energy_unc_loss+ pos_loss + time_loss + class_loss + exceed_beta + ccdamp
        
        bpush = self.calc_beta_push(pred_beta,t_idx)    
            
        lossval = tf.reduce_mean(lossval)+bpush
        
        self.add_prompt_metric(att,self.name+'_attractive_loss')
        self.add_prompt_metric(rep,self.name+'_repulsive_loss')
        self.add_prompt_metric(min_b,self.name+'_min_beta_loss')
        self.add_prompt_metric(noise,self.name+'_noise_loss')
        self.add_prompt_metric(energy_loss,self.name+'_energy_loss')
        self.add_prompt_metric(energy_unc_loss,self.name+'_energy_unc_loss')
        self.add_prompt_metric(pos_loss,self.name+'_position_loss')
        self.add_prompt_metric(time_loss,self.name+'_time_loss')
        self.add_prompt_metric(class_loss,self.name+'_class_loss')
        self.add_prompt_metric(exceed_beta,self.name+'_exceed_beta_loss')
        self.add_prompt_metric(bpush,self.name+'_beta_push_loss')
        
        self.add_prompt_metric(tf.reduce_mean(pred_distscale),self.name+'_avg_dist')
        
        self.maybe_print_loss(lossval)

        return lossval

    def get_config(self):
        config = {
            'energy_loss_weight': self.energy_loss_weight,
            'alt_energy_weight': self.alt_energy_weight,
            'use_energy_weights': self.use_energy_weights,
            'train_energy_correction': self.train_energy_correction,
            'q_min': self.q_min,
            'no_beta_norm': self.no_beta_norm,
            'potential_scaling': self.potential_scaling,
            'repulsion_scaling': self.repulsion_scaling,
            's_b': self.s_b,
            'noise_q_min': self.noise_q_min,
            'position_loss_weight': self.position_loss_weight,
            'classification_loss_weight' : self.classification_loss_weight,
            'timing_loss_weight': self.timing_loss_weight,
            'use_spectators': self.use_spectators,
            'beta_loss_scale': self.beta_loss_scale,
            'use_average_cc_pos': self.use_average_cc_pos,
            'payload_rel_threshold': self.payload_rel_threshold,
            'rel_energy_mse': self.rel_energy_mse,
            'smooth_rep_loss': self.smooth_rep_loss,
            'pre_train': self.pre_train,
            'huber_energy_scale': self.huber_energy_scale,
            'downweight_low_energy': self.downweight_low_energy,
            'n_ccoords': self.n_ccoords,
            'energy_den_offset': self.energy_den_offset,
            'noise_scaler': self.noise_scaler,
            'too_much_beta_scale': self.too_much_beta_scale,
            'cont_beta_loss': self.cont_beta_loss,
            'log_energy': self.log_energy,
            'n_classes': self.n_classes,
            'prob_repulsion': self.prob_repulsion,
            'phase_transition': self.phase_transition,
            'phase_transition_double_weight': self.phase_transition_double_weight,
            'alt_potential_norm': self.alt_potential_norm,
            'payload_beta_gradient_damping_strength': self.payload_beta_gradient_damping_strength,
            'payload_beta_clip' : self.payload_beta_clip,
            'kalpha_damping_strength' : self.kalpha_damping_strength,
            'cc_damping_strength' : self.cc_damping_strength,
            'beta_gradient_damping': self.beta_gradient_damping,
            'repulsion_q_min': self.repulsion_q_min,
            'super_repulsion': self.super_repulsion,
            'use_local_distances': self.use_local_distances,
            'energy_weighted_qmin': self.energy_weighted_qmin,
            'super_attraction':self.super_attraction,
            'div_repulsion' : self.div_repulsion,
            'dynamic_payload_scaling_onset': self.dynamic_payload_scaling_onset,
            'beta_push': self.beta_push
        }
        base_config = super(LLFullObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LLGraphCondOCLoss(LLFullObjectCondensation):
    
    def __init__(self, *args, **kwargs):
        '''
        Same as FullOCLoss but using GraphCond_OC_per_sample object condensation 
        (mean beta per object instead of max beta per object)
        '''
        super(LLGraphCondOCLoss, self).__init__(*args, **kwargs)
        
        from object_condensation import GraphCond_OC_per_sample
        
        self.oc_loss_object = OC_loss(
            loss_impl=GraphCond_OC_per_sample,
            
            q_min = self.oc_loss_object.loss_impl.q_min,
                 s_b=self.oc_loss_object.loss_impl.s_b,
                 use_mean_x=self.oc_loss_object.loss_impl.use_mean_x,
                 spect_supp=1.
            )
    


from ragged_tools import rwhere, add_ragged_offset_to_flat
class PFTruthCondensateBuilder(object):
    
    def _hit_en_mean(self, x, pfc_h_energy, pfc_hitensum, pf_h_idx=None, pf_idx=None, pfc_is_track=None): #x is [e, p, h, f]
        x_orig = x
        if pf_h_idx is not None:
            x = tf.gather_nd(x, pf_h_idx)
        x = x * pfc_h_energy
        x = tf.reduce_sum(x, axis=2)
        x = tf.math.divide_no_nan(x, pfc_hitensum + 1e-3)
        if pfc_is_track is not None:
            assert pf_idx is not None
            x = tf.where(pfc_is_track > 0., tf.gather_nd(x_orig, pf_idx), x)
        return x
    
    
    def _truth_purity(self,
                     pf_h_idx,
                     pf_idx,
                     t_idx,
                     pfc_h_energy, 
                     pfc_hitensum
                     ):
        '''
        inputs: pf_h_idx, pf_idx (ragged), rest per hit
        
        output: ragged (in pfc) purity measure, corrected truth energy #[e, pf,  1]
        '''
        pf_h_t_idx = tf.gather_nd(t_idx, pf_h_idx)
        pf_h_t_idx = rwhere(pf_h_t_idx < 0, -2, pf_h_t_idx)#mask noise -2
        
        pfc_t_idx = tf.gather_nd(t_idx, pf_idx)
        pfc_h_same = tf.expand_dims(pfc_t_idx, axis=2) == pf_h_t_idx
        pfc_h_same = tf.cast(pfc_h_same, 'float32') #[e, pf, h, 1]
        
        purity = self._hit_en_mean(pfc_h_same, pfc_h_energy, pfc_hitensum)
        
        return purity
    
        
    
    def build_pf_truth(self, 
                       pf_h_idx, pf_idx,
                       pfc_istrack, #ragged already
                       
                       #these are all per hit
                       hit_energy,
                       is_track,
                       
                       t_idx,
                       t_energy,
                       t_pos,
                       t_time,
                       t_pid,
                       t_fully_contained,
                       t_rec_energy
                       ):
        
        '''
        The output will be flattened in 
        [npf, x]  with row splits!
        '''
        
        #globally useful inputs
        pfc_h_energy = tf.gather_nd(hit_energy, pf_h_idx) #[e, p, h, 1]
        pfc_h_istrack = tf.gather_nd(is_track, pf_h_idx)
        pfc_h_energy *= (1. - pfc_h_istrack) #tf.where(pfc_h_istrack  > 0., 0., pfc_h_energy)#mask track energies
        pfc_hitensum = tf.reduce_sum(pfc_h_energy, axis=2)
        
        encorr = tf.math.divide_no_nan(t_energy, t_rec_energy)

        out = {}
        #basically all get passed through
        pfc_t_idx = tf.gather_nd(t_idx, pf_idx)
        out['pfc_t_idx'] = pfc_t_idx.values
        out['rs'] = pfc_t_idx.row_splits
        
        #use hit energy weighted mean correction for neutrals otherwise exact track correction for charged
        out['pfc_t_encorr'] = self._hit_en_mean(
            encorr, pfc_h_energy, pfc_hitensum, pf_h_idx, pf_idx, pfc_istrack).values
        
        out['pfc_t_energy'] = self._hit_en_mean(
            t_energy, pfc_h_energy, pfc_hitensum, pf_h_idx, pf_idx, pfc_istrack).values
        
        out['pfc_t_rec_energy'] = self._hit_en_mean(
            t_rec_energy, pfc_h_energy, pfc_hitensum, pf_h_idx, pf_idx, pfc_istrack).values
        
        out['pfc_t_pos'] = self._hit_en_mean(
            t_pos, pfc_h_energy, pfc_hitensum, pf_h_idx, pf_idx, pfc_istrack).values
        
        out['pfc_t_time'] = tf.gather_nd(t_time, pf_idx).values#direct
        
        out['pfc_t_id'] =  tf.gather_nd(t_pid, pf_idx).values#direct
        
        out['pfc_t_fully_contained'] = tf.gather_nd(t_fully_contained, pf_idx).values#direct
        
        out['pfc_t_purity'] = self._truth_purity(pf_h_idx, pf_idx, t_idx,
                     pfc_h_energy, pfc_hitensum).values
        
        return out

class LLPFCondensates(LLFullObjectCondensation):
    
    
    def __init__(self, purity_threshold = 0.9, **kwargs):
        super(LLPFCondensates, self).__init__(**kwargs)
        self.truthbuilder = PFTruthCondensateBuilder()
    
        self.purity_threshold = purity_threshold
    
    def energy_conservation_loss(self, pfc_ccoords, t_idx, t_energy, rs):
        ## select unique truth indices
        # create both truth and pred collection
        # assign truth_e = 0 to pred and pred_e = 0 to truth
        # concat both and run kNN. 
        # evaluate for both truth and pred e
        # weight by gaussian and force same
        # take into account how good the prediction on single particle
        # level already is. Force only badly predicted ones to change a lot
        # keep good ones
        pass
    
    
    def loss(self, inputs):
        '''
        inputs:
        
        
        #this is all ragged  [e, pfc-ragged, X]
        
        - pfc_mom_corr
        - pfc_ensum
        - pfc_energy_low_quantile
        - pfc_energy_high_quantile
        - pfc_pos
        - pfc_time
        - pfc_time_unc
        - pfc_id
        - pfc_ccoords
        - pfc_istrack
        - pfc_isnoise
        
        - rechit_energy #for weighting
        
        - pf_h_idx
        - pf_idx
        # - pf_rs #that's part of the ragged tensors, no need
        
        #this is not ragged, yet (that's why the indices above
        
        - is_track
        
        - t_idx
        - t_energy
        - t_pos
        - t_time
        - t_pid
        - t_fully_contained
        - t_rec_energy
        
        #- rs not really needed
        
        
        this should be factorise. Maybe we can build a PF truth first?
        
        '''
        
        pfc_mom_corr, pfc_ensum, pfc_energy_low_quantile, pfc_energy_high_quantile,\
        pfc_pos, pfc_time, pfc_time_unc, pfc_id,\
        pfc_ccoords, pfc_istrack, pfc_isnoise,\
        \
        rechit_energy, \
        pf_h_idx, pf_idx, is_track, t_idx, t_energy, t_pos, t_time,\
        t_pid, t_fully_contained, t_rec_energy = inputs
        
        
        
        #the last paragraph is only used by PF truth builder
        st = time.time()
        
        pftruth = self.truthbuilder.build_pf_truth(pf_h_idx, pf_idx, pfc_istrack, 
                                                   rechit_energy, is_track, t_idx, t_energy, t_pos, 
                                                   t_time, t_pid, t_fully_contained, t_rec_energy)
        print('pftruth', time.time()-st)
        #flatten all pfc
        pfc_mom_corr = pfc_mom_corr.values
        pfc_ensum = pfc_ensum.values
        pfc_energy_low_quantile = pfc_energy_low_quantile.values
        pfc_energy_high_quantile = pfc_energy_high_quantile.values
        pfc_pos = pfc_pos.values
        pfc_time = pfc_time.values
        pfc_time_unc = pfc_time_unc.values
        pfc_id = pfc_id.values
        
        #for metrics?
        pfc_istrack = pfc_istrack.values
        pfc_isnoise = pfc_isnoise.values
        
        pfc_t_rec_energy = tf.gather_nd(t_rec_energy, pf_idx).values
        
        #safety
        pfc_ccoords = None
        
        #now it's all the same dimensionality
        notnoise = 1. - pfc_isnoise
        ispure = tf.where(pftruth['pfc_t_purity'] < self.purity_threshold, 0., tf.ones_like(pftruth['pfc_t_purity']))
        
        eweight = self.calc_energy_weights(pftruth['pfc_t_energy'], pftruth['pfc_t_id'], False)
        
        #only to pure
        closs = ispure * notnoise * self.classification_loss_weight * self.calc_classification_loss(
            pftruth['pfc_t_id'], pfc_id)
        
        #don't require timing for MIPS (yet)
        #time loss only for pure
        tloss = ispure * notnoise * self.timing_loss_weight * \
            self.calc_timing_loss(pftruth['pfc_t_time'], pfc_time, pfc_time_unc, pfc_t_rec_energy)
        
        ploss = notnoise * self.position_loss_weight * \
            self.calc_position_loss(pftruth['pfc_t_pos'], pfc_pos)
        
        e_andunc_loss = notnoise * self.energy_loss_weight * \
            self.calc_energy_correction_factor_loss(
            pftruth['pfc_t_energy'], pftruth['pfc_t_rec_energy'], 
            pfc_mom_corr, pfc_energy_low_quantile, pfc_energy_high_quantile, return_concat=True)
        
        eloss, euncloss = e_andunc_loss[...,0:1], e_andunc_loss[...,1:2]
        euncloss /= 2. #same as above
        
        self.add_prompt_metric(tf.reduce_mean(eweight * closs), self.name + '_class_loss')
        self.add_prompt_metric(tf.reduce_mean(eweight * tloss), self.name + '_time_loss')
        self.add_prompt_metric(tf.reduce_mean(eweight * ploss), self.name + '_pos_loss')
        self.add_prompt_metric(tf.reduce_mean(eweight * eloss), self.name + '_momentum_loss')
        self.add_prompt_metric(tf.reduce_mean(eweight * euncloss), self.name + '_momentum_unc_loss')
        
        #this is as if there weren't any modifications
        pfc_t_energy = tf.gather_nd(t_energy, pf_idx).values
        
        ## a bunch of metrics that could be nice
        resranges = [2., 10., 20, 50, 100., 200.]
        for i in range(len(resranges)-1):
            sel = tf.logical_and( pftruth['pfc_t_energy'] >= resranges[i], pftruth['pfc_t_energy']< resranges[i+1])
            sel = sel[...,0]
            namestr = str(resranges[i]) + '_to_' + str(resranges[i+1])
            
            s_pfc_mom_corr = tf.ragged.boolean_mask(pfc_mom_corr, sel)
            s_pfc_t_encorr = tf.ragged.boolean_mask(pftruth['pfc_t_encorr'], sel)
            
            offset = tf.reduce_mean( s_pfc_mom_corr - s_pfc_t_encorr )
            var = tf.math.reduce_std(s_pfc_mom_corr - s_pfc_t_encorr - offset)
            
            self.add_prompt_metric(offset, self.name + '_binned_en_offset_'+namestr)
            self.add_prompt_metric(var, self.name + '_binned_en_std_'+namestr)
            
            s_pfc_energy = s_pfc_mom_corr * tf.ragged.boolean_mask(pfc_ensum, sel)
            s_pfc_t_energy = tf.ragged.boolean_mask(pfc_t_energy, sel)
            
            offset = tf.reduce_mean( (s_pfc_energy - s_pfc_t_energy)/s_pfc_t_energy )
            var = tf.math.reduce_std((s_pfc_energy - s_pfc_t_energy)/s_pfc_t_energy - offset)
            
            self.add_prompt_metric(offset, self.name + '_binned_t_en_offset_'+namestr)
            self.add_prompt_metric(var, self.name + '_binned_t_en_std_'+namestr)
            
        
        allloss = eweight * (closs + tloss + ploss + eloss + euncloss)
        
        return tf.reduce_mean(allloss)
        
        
        













