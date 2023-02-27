# -*- coding: utf-8 -*-

import tensorflow as tf
from oc_helper_ops import CreateMidx, SelectWithDefault
from binned_select_knn_op import BinnedSelectKnn



def huber(x, d):
    losssq  = x**2   
    absx = tf.abs(x)                
    losslin = d**2 + 2. * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)

def remove_zero_length_elements_from_ragged_tensors(row_splits):
    lengths = row_splits[1:] - row_splits[:-1]
    row_splits = tf.concat(([0], tf.cumsum(tf.gather_nd(lengths, tf.where(tf.not_equal(lengths, 0))))), axis=0)
    return row_splits



# create a few with fixed number and then make an if statement selecting the right one
# @tf.function
def normalize_weights(weights):
    '''
    Input is X x V x X
    Outputs are X x V x X
    normalises across V
    '''
    weight_sum = tf.reduce_sum(weights, axis=1, keepdims=True) #K x 1 x X
    return tf.math.divide_no_nan(weights,weight_sum)
    

def payload_weight_function(w, w2, threshold):
    '''
    Input/output  K x V x 1
    normalised in V
    '''
    w = tf.clip_by_value(w, 0., 1.-1e-4)
    w = w2*tf.math.atanh(w)**2
    w_max = tf.reduce_max(w, axis = 1, keepdims=True) # K x 1 x 1
    w = tf.math.divide_no_nan(w, w_max)
    if threshold>0:
        w = tf.nn.relu(w-threshold)
    return normalize_weights(w)
    

def gather_for_obj_from_vert(v_prop, ids):
    '''
    In: V x X , ids
    Out: K x 1 x X
    '''
    gathered = tf.gather_nd(tf.tile(tf.expand_dims(v_prop, axis=0), [ids.shape[0]]+[1]*len(v_prop.shape)), ids, batch_dims=1)
    gathered = tf.expand_dims(gathered, axis=1)
    return gathered

def mean_N_K(x, N, K):
    '''
    Input: K x V x X
    Output: X
    '''
    red = tf.reduce_sum(x, axis=0)#K
    red = tf.reduce_sum(red, axis=0)#V
    return tf.math.divide_no_nan(red, tf.expand_dims(N*K, axis=0))


class Basic_OC_per_sample(object):
    def __init__(self, 
                 
                 q_min,
                 s_b,
                 use_mean_x,
                 spect_supp=None #None means same as noise
                 ):
        
        self.q_min = q_min
        self.s_b = s_b
        self.use_mean_x = use_mean_x
        if spect_supp is None:
            spect_supp = s_b
        self.spect_supp = spect_supp
        
        self.valid=False #constants not created
        
        
    #helper
    def _create_x_alpha_k(self): 
        x_kalpha_m = tf.gather_nd(self.x_k_m,self.alpha_k, batch_dims=1) # K x C
        if self.use_mean_x>0:
            w_k_m = self.q_k_m * self.mask_k_m
            x_kalpha_m_m = tf.reduce_sum(w_k_m * self.x_k_m,axis=1) # K x C
            x_kalpha_m_m = tf.math.divide_no_nan(x_kalpha_m_m, tf.reduce_sum(w_k_m, axis=1)+1e-4)
            x_kalpha_m = self.use_mean_x * x_kalpha_m_m + (1. - self.use_mean_x)*x_kalpha_m
        
        return x_kalpha_m 
    
    def create_Ms(self, truth_idx):
        self.Msel, self.Mnot, _ = CreateMidx(truth_idx, calc_m_not=True)
    
    def set_input(self, 
                         beta,
                         x,
                         d,
                         pll,
                         truth_idx,
                         object_weight,
                         is_spectator_weight,
                         calc_Ms=True,
                         ):
        self.valid=True
        #used for pll and q
        self.tanhsqbeta = tf.math.atanh(beta/(1.01))**2
        
        self.beta_v = tf.debugging.check_numerics(beta,"OC: beta input")
        self.d_v = tf.debugging.check_numerics(d,"OC: d input")
        self.x_v = tf.debugging.check_numerics(x,"OC: x input")
        self.pll_v = tf.debugging.check_numerics(pll,"OC: pll input")
        self.sw_v = tf.debugging.check_numerics(is_spectator_weight,"OC: is_spectator_weight input")

        object_weight = tf.debugging.check_numerics(object_weight,"OC: object_weight input")
        
        self.isn_v = tf.where(truth_idx<0, tf.zeros_like(truth_idx,dtype='float32')+1., 0.)
        
        #spectators do not participate in the potential losses
        self.q_v = (self.tanhsqbeta + self.q_min)*tf.clip_by_value(1.-is_spectator_weight, 0., 1.)
        
        if calc_Ms:
            self.create_Ms(truth_idx)
        if self.Msel is None:
            self.valid=False
            return
        #if self.Msel.shape[0] < 2:#less than two objects - can be dangerous
        #    self.valid=False
        #    return
        
        self.mask_k_m = SelectWithDefault(self.Msel, tf.zeros_like(beta)+1., 0.) #K x V-obj x 1
        self.beta_k_m = SelectWithDefault(self.Msel, self.beta_v, 0.) #K x V-obj x 1
        self.x_k_m = SelectWithDefault(self.Msel, self.x_v, 0.) #K x V-obj x C
        self.q_k_m = SelectWithDefault(self.Msel, self.q_v, 0.)#K x V-obj x 1
        self.d_k_m = SelectWithDefault(self.Msel, self.d_v, 0.)
        
        self.alpha_k = tf.argmax(self.q_k_m, axis=1)# high beta and not spectator -> large q
        
        self.beta_k = tf.gather_nd(self.beta_k_m, self.alpha_k, batch_dims=1) # K x 1
        self.x_k = self._create_x_alpha_k() #K x C
        self.q_k = tf.gather_nd(self.q_k_m, self.alpha_k, batch_dims=1) # K x 1
        self.d_k = tf.gather_nd(self.d_k_m, self.alpha_k, batch_dims=1) # K x 1
        
        #just a temp
        ow_k_m = SelectWithDefault(self.Msel, object_weight, 0.)
        self.ow_k = tf.gather_nd(ow_k_m, self.alpha_k, batch_dims=1) # K x 1
        
    
    ### the following functions should not modify any of the constants and must only depend on them
    
    #for override through inheriting
    def att_func(self,dsq_k_m):
        return tf.math.log(tf.math.exp(1.)*dsq_k_m/2. + 1.)
    
    def V_att_k(self):
        '''
        '''
        x_k_e = tf.expand_dims(self.x_k,axis=1)
        
        N_k =  tf.reduce_sum(self.mask_k_m, axis=1)
        
        dsq_k_m = tf.reduce_sum((self.x_k_m - x_k_e)**2, axis=-1, keepdims=True) #K x V-obj x 1
        
        sigma = self.weighted_d_k_m(dsq_k_m) #create gradients for all
        
        dsq_k_m = tf.math.divide_no_nan(dsq_k_m, sigma + 1e-4)
            
        V_att = self.att_func(dsq_k_m) * self.q_k_m * self.mask_k_m  #K x V-obj x 1
    
        V_att = self.q_k * tf.reduce_sum( V_att ,axis=1)  #K x 1
        V_att = tf.math.divide_no_nan(V_att, N_k+1e-3)  #K x 1
        
        #print(tf.reduce_mean(self.d_v),tf.reduce_max(self.d_v))
        
        return V_att
    
    def rep_func(self,dsq_k_v):
        return tf.math.exp(-dsq_k_v/2.)
    
    def weighted_d_k_m(self, dsq): # dsq K x V x 1
        return tf.expand_dims(self.d_k, axis=1) # K x 1 x 1
        
    def V_rep_k(self):
        
        
        N_k = tf.reduce_sum(self.Mnot, axis=1)
        #future remark: if this gets too large, one could use a kNN here
        
        dsq = tf.expand_dims(self.x_k, axis=1) - tf.expand_dims(self.x_v, axis=0) #K x V x C
        dsq = tf.reduce_sum(dsq**2, axis=-1, keepdims=True)  #K x V x 1
        
        # nogradbeta = tf.stop_gradient(self.beta_k_m)
        #weight. tf.reduce_sum( tf.exp(-dsq) * d_v_e, , axis=1) / tf.reduce_sum( tf.exp(-dsq) )
        sigma = self.weighted_d_k_m(dsq) #create gradients for all, but prefer k vertex
        
        dsq = tf.math.divide_no_nan(dsq, sigma + 1e-4) #K x V x 1
        
        V_rep = self.rep_func(dsq) * self.Mnot * tf.expand_dims(self.q_v,axis=0)  #K x V x 1
        
        V_rep = self.q_k * tf.reduce_sum(V_rep, axis=1) #K x 1
        V_rep = tf.math.divide_no_nan(V_rep, N_k+1e-3)  #K x 1
        
        return V_rep
    
    def Pll_k(self):
        
        tanhsqbeta = self.beta_v**2 #softer here
        tanhsqbeta = tf.debugging.check_numerics(tanhsqbeta, "OC: pw b**2")
        pw = tanhsqbeta * tf.clip_by_value((1.-tf.clip_by_value(self.isn_v+self.sw_v,0.,1.)),0.,1.) + 1e-6
        
        pw = tf.debugging.check_numerics(pw, "OC: pw")
        
        pll_k_m = SelectWithDefault(self.Msel, self.pll_v, 0.) #K x V_perobj x P
        pw_k_m = SelectWithDefault(self.Msel, pw, 0.) #K x V-obj x P
        pw_k_sum = tf.reduce_sum(pw_k_m, axis=1)
        pw_k_sum = tf.where(pw_k_sum <= 0., 1e-2, pw_k_sum)
        
        pll_k = tf.math.divide_no_nan(tf.reduce_sum(pll_k_m * pw_k_m, axis=1), 
                                             pw_k_sum  )#K x P
        return pll_k
    
    def Beta_pen_k(self):
        #use continuous max approximation through LSE
        eps = 1e-3
        beta_pen = 1. - eps * tf.reduce_logsumexp(self.beta_k_m/eps, axis=1)#sum over m
        #for faster convergence  
        beta_pen += 1. - tf.clip_by_value(tf.reduce_sum(self.beta_k_m, axis=1), 0., 1)
        beta_pen = tf.debugging.check_numerics(beta_pen, "OC: beta pen")
        return beta_pen
        
    def Noise_pen(self):
        
        nsupp_v = self.beta_v * self.isn_v
        nsupp = tf.math.divide_no_nan(tf.reduce_sum(nsupp_v), 
                                      tf.reduce_sum(self.isn_v)+1e-3) # nodim
        
        specsupp_v = self.beta_v * self.sw_v
        specsupp = tf.math.divide_no_nan(tf.reduce_sum(specsupp_v), 
                                      tf.reduce_sum(self.sw_v)+1e-3) # nodim
        
        return self.s_b * nsupp + self.spect_supp * specsupp
        
    
    # doesn't do anything in this implementation
    def high_B_pen_k(self):
        return 0.* self.beta_k
    
    # override with more complex through inheritance
    def pll_weight_k(self, ow_k, vatt_k, vrep_k):
        return ow_k
    
    
        
    def add_to_terms(self,
                     V_att, 
                     V_rep,
                     Noise_pen, 
                     B_pen, 
                     pll,
                     high_B_pen
                     ):
        
        zero_tensor = tf.zeros_like(tf.reduce_mean(self.q_v,axis=0))
        
        if not self.valid: # no objects
            zero_payload = tf.zeros_like(tf.reduce_mean(self.pll_v,axis=0))
            print('WARNING: no objects in sample, continue to next')
            return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_payload, zero_tensor
    
        K = tf.reduce_sum(tf.ones_like(self.q_k)) # > 0
        
        V_att_k = self.V_att_k()
        V_rep_k = self.V_rep_k()
        
        V_att += tf.reduce_sum(self.ow_k * V_att_k)/K
        V_rep += tf.reduce_sum(self.ow_k * V_rep_k)/K
        Noise_pen += self.Noise_pen()
        B_pen += tf.reduce_sum(self.ow_k * self.Beta_pen_k())/K
        
        pl_ow_k = self.pll_weight_k(self.ow_k, V_att_k, V_rep_k)
        pll += tf.reduce_sum(pl_ow_k * self.Pll_k(),axis=0)/K 
        
        high_B_pen += tf.reduce_sum(self.ow_k *self.high_B_pen_k())/K
        
        return V_att, V_rep, Noise_pen, B_pen, pll, high_B_pen
        
        


class PushPull_OC_per_sample(Basic_OC_per_sample):
    
    def __init__(self, **kwargs):
        super(PushPull_OC_per_sample, self).__init__(**kwargs)
        
    def V_att_k(self):
         
        x_k_e = tf.expand_dims(self.x_k,axis=1)
        d_k_e = tf.expand_dims(self.d_k,axis=1)
        
        N_k =  tf.reduce_sum(self.mask_k_m, axis=1)
        
        dsq_k_m = tf.reduce_sum((self.x_k_m - x_k_e)**2, axis=-1, keepdims=True) #K x V-obj x 1
        
        sigma = 0.95 * d_k_e**2 + 0.05 * self.d_k_m**2 #create gradients for all
        
        dsq_k_m = tf.math.divide_no_nan(dsq_k_m, sigma + 1e-4)
            
        # attractive here scales with beta *not* with q!! (otherwise same)
        V_att = self.att_func(dsq_k_m) * (self.beta_k_m + self.q_min/10.) * self.mask_k_m  #K x V-obj x 1
    
        # attractive here scales with beta *not* with q!! (otherwise same)
        V_att = (self.beta_k + self.q_min/10.) * tf.reduce_sum( V_att ,axis=1)  #K x 1
        V_att = tf.math.divide_no_nan(V_att, N_k+1e-3)  #K x 1
        
        #print(tf.reduce_mean(self.d_v),tf.reduce_max(self.d_v))
        
        return V_att
        
    def att_func(self,dsq_k_m):
        return tf.where(dsq_k_m==0, 0., - self.rep_func(dsq_k_m/0.5**2)) #create a probability well
    
    def Beta_pen_k(self):
        return 0.*(1. - self.beta_k) #always zero
    
class PreCond_OC_per_sample(Basic_OC_per_sample):
    
    def __init__(self, 
                 att_scale=0.1,
                 **kwargs):
        self.att_scale = att_scale
        super(PreCond_OC_per_sample, self).__init__(**kwargs)
        
    def att_func(self,dsq_k_m):
        return 1. - tf.math.exp(-dsq_k_m/(2.* self.att_scale**2)) + 0.1*tf.sqrt(dsq_k_m + 1e-6)
    
    
class PreCond_kNNOC_per_sample(PreCond_OC_per_sample):
    
    def __init__(self, 
                 K=128,
                 **kwargs):
        self.K = K
        self.Mnotsel = None
        super(PreCond_kNNOC_per_sample, self).__init__(**kwargs)
            
    
    def create_Ms(self, truth_idx):
        #this is now based on kNN and needs as output indices
        #
        '''
        already defined:
        
        self.beta_v = tf.debugging.check_numerics(beta,"OC: beta input")
        self.d_v = tf.debugging.check_numerics(d,"OC: d input")
        self.x_v = tf.debugging.check_numerics(x,"OC: x input")
        self.pll_v = tf.debugging.check_numerics(pll,"OC: pll input")
        '''
        rs = tf.concat([0, tf.shape(self.x_v)[0] ],axis=0)
        idx, _ = BinnedSelectKnn(self.K, self.x_v, rs)
        
        selfidx = idx[:,0:1]
        nnidx = idx[:,1:]
        
        nntidx = SelectWithDefault(nnidx, truth_idx, -1)
        
        same = tf.where(truth_idx == nntidx, nnidx, -1)
        nonoise_and_same = tf.where(truth_idx < 0, -1, same)
        
        self.Msel = tf.concat([selfidx, nonoise_and_same],axis=1)
        
        self.Mnot = None
        self.Mnotsel = tf.where(truth_idx != nntidx, nnidx, -1)

    #needs adjustment to avoid N**2
    def V_rep_k(self):
        
        d_k_e = tf.expand_dims(self.d_k, axis=1) # K x 1 x 1
        d_v_e = SelectWithDefault(self.Mnotsel, self.d_v, 0.) # K x k x 1
        
        x_v = SelectWithDefault(self.Mnotsel, self.d_v, 0.) # K x k x C
        
        N_k = tf.cast(self.K, 'float32')
        
        dsq = tf.expand_dims(self.x_k, axis=1) - x_v #K x k x C
        dsq = tf.reduce_sum(dsq**2, axis=-1, keepdims=True)  #K x k x 1
        
        sigma = 0.95 * d_k_e**2 + 0.05 * d_v_e**2 #create gradients for all, but prefer k vertex
        
        dsq = tf.math.divide_no_nan(dsq, sigma + 1e-4) #K x V x 1
        
        V_rep = self.rep_func(dsq) * SelectWithDefault(self.Mnotsel, self.q_v, 0.)  #K x k x 1
        
        V_rep = self.q_k * tf.reduce_sum(V_rep, axis=1) #K x 1
        V_rep = tf.math.divide_no_nan(V_rep, N_k+1e-3)  #K x 1
        
        return V_rep
    
class GraphCond_OC_per_sample(Basic_OC_per_sample):
    
    def set_input(self, beta,
                  x,
                         d,
                         pll,
                         truth_idx,
                         *args,**kwargs
                         ):
        
        #replace beta with per-object normalised value
        self.create_Ms(truth_idx)
        beta_max = tf.reduce_max(self.Mnot * tf.expand_dims(beta,axis=0),axis=1, keepdims=True)  #K x 1 x 1
        beta_max *= self.Mnot  #K x V x 1
        
        # this is the same reduced in K given the others are zero with exception of noise (filtered later)
        beta_max = tf.reduce_max(beta_max, axis=0) # V x 1 
        beta_normed = tf.math.divide_no_nan(beta, beta_max) # V x 1 
        beta = tf.where(truth_idx<0, beta, beta_normed) #only for not noise
        
        super(GraphCond_OC_per_sample, self).set_input(beta, x, d, pll, truth_idx, *args,**kwargs, calc_Ms=False)
    
    def Beta_pen_k(self):
        #simple mean beta per object
        pen = tf.math.divide_no_nan(tf.reduce_sum(self.mask_k_m * self.beta_k_m, axis=1), 
                                     tf.reduce_sum(self.mask_k_m, axis=1) + 1e-3)
        return tf.debugging.check_numerics(pen,"GCOC: beta penalty")
    

class OC_loss(object):
    def __init__(self, 
                 loss_impl=Basic_OC_per_sample,
                 **kwargs
                 ):
        self.loss_impl=loss_impl(**kwargs)


    def __call__(self, beta,
                         x,
                         d,
                         pll,
                         truth_idx,
                         object_weight,
                         is_spectator_weight,
                         
                         rs): #rs last
        
        tot_V_att, tot_V_rep, tot_Noise_pen, tot_B_pen, tot_pll,tot_too_much_B_pen = 6*[tf.constant(0., tf.float32)]
        #batch loop
            
        if rs.shape[0] is None or rs.shape[0] < 2:
            return tot_V_att, tot_V_rep, tot_Noise_pen, tot_B_pen, tot_pll,tot_too_much_B_pen
        batch_size = rs.shape[0] - 1
    
        for b in tf.range(batch_size):
            
            self.loss_impl.set_input( 
                 beta[rs[b]:rs[b + 1]],
                 x[rs[b]:rs[b + 1]],
                 d[rs[b]:rs[b + 1]],
                 pll[rs[b]:rs[b + 1]],
                 truth_idx[rs[b]:rs[b + 1]],
                 object_weight[rs[b]:rs[b + 1]],
                 is_spectator_weight[rs[b]:rs[b + 1]]
                 )
            
            tot_V_att, tot_V_rep, tot_Noise_pen, tot_B_pen, tot_pll,tot_too_much_B_pen = self.loss_impl.add_to_terms(
                tot_V_att, tot_V_rep, tot_Noise_pen, tot_B_pen, tot_pll,tot_too_much_B_pen
                )
            
        bs = tf.cast(batch_size, dtype='float32') + 1e-3
        out = [a/bs for a in [tot_V_att, tot_V_rep, tot_Noise_pen, tot_B_pen, tot_pll,tot_too_much_B_pen]]
        
        return out


#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
###### below old for comparison #################
#################################################
#################################################
#################################################
#################################################
#################################################

def oc_per_batch_element(
        beta,
        x,
        q_min,
        object_weights, # V x 1 !!
        truth_idx,
        is_spectator,
        payload_loss,
        S_B=1.,
        noise_q_min=None,
        distance_scale=None,
        payload_weight_function = None,  #receives betas as K x V x 1 as input, and a threshold val
        payload_weight_threshold = 0.8,
        use_mean_x = 0.,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        payload_beta_gradient_damping_strength=0.,
        kalpha_damping_strength=0.,
        beta_gradient_damping=0.,
        soft_q_scaling=True,
        weight_by_q=False, 
        repulsion_q_min=-1.,
        super_repulsion=False,
        super_attraction=False,
        div_repulsion=False,
        soft_att=True,
        dynamic_payload_scaling_onset=-0.03
        ):
    '''
    all inputs
    V x X , where X can be 1
    
    is_ambiguous is applied to (nullifies for ambiguous hits):
    - potentials
    - payload losses
    - beta loss
    is_ambiguous = 1 does receive a beta penalty (same as noise) though
    
    '''
    tf.assert_equal(True, is_spectator>=0.)
    tf.assert_equal(True, beta>=0.)
    
    if prob_repulsion:
        raise ValueError("prob_repulsion not implemented")
    if phase_transition_double_weight:
        raise ValueError("phase_transition_double_weight not implemented")
    if payload_weight_function is not None:
        raise ValueError("payload_weight_function not implemented")
        
    #set all spectators invalid here, everything scales with beta, so:
    if beta_gradient_damping > 0.:
        beta = beta_gradient_damping * tf.stop_gradient(beta) + (1. - beta_gradient_damping)*beta
    beta_in = beta
    beta = tf.clip_by_value(beta, 0.,1.-1e-4)
    
    q_min *= (1. - is_spectator)
    
    qraw = tf.math.atanh(beta)**2 
    if soft_q_scaling:
        qraw = tf.math.atanh(beta_in/1.002)**2 #beta_in**4 *20.
    
    is_noise = tf.where(truth_idx<0, tf.zeros_like(truth_idx,dtype='float32')+1., 0.)#V x 1
    if noise_q_min is not None:
        q_min = (1.-is_noise)*q_min + is_noise*noise_q_min
    
    q_min = tf.where(q_min<0,0.,q_min)#just safety in case there are some numerical effects
    
    q = qraw + q_min # V x 1
    #q = tf.where(beta_in<1.-1e-4, q, tf.math.atanh(1.-1e-4)**2 + q_min + beta_in) #just give the rest above clip a gradient
    
    N = tf.cast(beta.shape[0], dtype='float32')
    
    Msel, M_not, N_per_obj = CreateMidx(truth_idx, calc_m_not=True)
    #use eager here
    if Msel is None:
        #V_att, V_rep, Noise_pen, B_pen, pll, too_much_B_pen
        print('>>> WARNING: Event has no objects, only noise! Will return zero loss. <<<')
        zero_tensor = tf.reduce_mean(q,axis=0)*0.
        zero_payload = tf.reduce_mean(payload_loss,axis=0)*0.
        return zero_tensor,zero_tensor,zero_tensor,zero_tensor,zero_payload,zero_tensor
    
    N_per_obj = tf.cast(N_per_obj, dtype='float32') # K x 1
    
    K = tf.cast(Msel.shape[0], dtype='float32') 
    
    ########################################################
    #sanity check, use none of the following for the loss calculation
    truth_m = SelectWithDefault(Msel, truth_idx, -2)#K x V-obj x 1
    truth_same = truth_m[:,0:1] == truth_m
    truth_same = tf.where(truth_m==-2, True, truth_same)
    tf.assert_equal( tf.reduce_all(truth_same), True , 
                     message="truth indices do not match object selection, serious bug")
    #end sanity check
    ########################################################
    
    padmask_m = SelectWithDefault(Msel, tf.zeros_like(beta_in)+1., 0.) #K x V-obj x 1
    x_m = SelectWithDefault(Msel, x, 0.) #K x V-obj x C
    beta_m = SelectWithDefault(Msel, beta, 0.) #K x V-obj x 1
    is_spectator_m = SelectWithDefault(Msel, is_spectator, 0.) #K x V-obj x 1
    q_m = SelectWithDefault(Msel, q, 0.)#K x V-obj x 1
    object_weights_m = SelectWithDefault(Msel, object_weights, 0.)
    
    distance_scale += 1e-3
    distance_scale_m = SelectWithDefault(Msel, distance_scale, 1.)
    
    tf.assert_greater(distance_scale_m, 0.,message="predicted distances must be greater zero")
    
    kalpha_m = tf.argmax((1.-is_spectator_m)*beta_m, axis=1) # K x 1
    
    x_kalpha_m = tf.gather_nd(x_m,kalpha_m, batch_dims=1) # K x C
    if use_mean_x>0:
        x_kalpha_m_m = tf.reduce_sum(beta_m * q_m * x_m * padmask_m,axis=1) # K x C
        x_kalpha_m_m = tf.math.divide_no_nan(x_kalpha_m_m, tf.reduce_sum(beta_m * q_m * padmask_m, axis=1)+1e-9)
        x_kalpha_m = use_mean_x * x_kalpha_m_m + (1. - use_mean_x)*x_kalpha_m
    
    if kalpha_damping_strength > 0:
        x_kalpha_m = kalpha_damping_strength * tf.stop_gradient(x_kalpha_m) + (1. - kalpha_damping_strength)*x_kalpha_m
    
    q_kalpha_m = tf.gather_nd(q_m,kalpha_m, batch_dims=1) # K x 1
    beta_kalpha_m = tf.gather_nd(beta_m,kalpha_m, batch_dims=1) # K x 1
    
    object_weights_kalpha_m = tf.gather_nd(object_weights_m,kalpha_m, batch_dims=1) # K x 1
    
    #make the distance scale a beta weighted mean so that there is more than 1 impact per object
    distance_scale_kalpha_m = tf.math.divide_no_nan(
        tf.reduce_sum(distance_scale_m*beta_m*padmask_m, axis=1),
        tf.reduce_sum(beta_m*padmask_m,axis=1)+1e-3
        )+1e-3 #K x 1
    #distance_scale_kalpha_m = tf.gather_nd(distance_scale_m,kalpha_m, batch_dims=1) # K x 1
    
    
    distance_scale_kalpha_m_exp = tf.expand_dims(distance_scale_kalpha_m, axis=2) # K x 1 x 1
    
    distancesq_m = tf.reduce_sum( (tf.expand_dims(x_kalpha_m, axis=1) - x_m)**2, axis=-1, keepdims=True) #K x V-obj x 1
    distancesq_m = tf.math.divide_no_nan(distancesq_m, 2.*distance_scale_kalpha_m_exp**2+1e-6)
    
    absdist = tf.sqrt(distancesq_m + 1e-6)
    huberdistsq = huber(absdist, d=4) #acts at 4
    if super_attraction:
        huberdistsq += 1. - tf.math.exp(-100.*absdist)
        
    V_att = q_m * tf.expand_dims(q_kalpha_m,axis=1) * huberdistsq #K x V-obj x 1
    
    if soft_att:
        V_att = q_m * tf.math.log(tf.math.exp(1.)*distancesq_m+1.)
        
    V_att = V_att * tf.expand_dims(object_weights_kalpha_m,axis=1) #K x V-obj x 1
    
    if weight_by_q:
        V_att = tf.math.divide_no_nan(tf.reduce_sum(padmask_m * V_att,axis=1), tf.reduce_sum(q_m, axis=1)) # K x 1
    else:
        V_att = tf.math.divide_no_nan(tf.reduce_sum(padmask_m * V_att,axis=1), N_per_obj+1e-9) # K x 1
    
    # opt. used later in payload loss
    V_att_K = V_att
    V_att = tf.math.divide_no_nan(tf.reduce_sum(V_att,axis=0), K+1e-9) # 1
    
    
    #what if Vatt and Vrep are weighted by q, not scaled by it?
    q_rep = q
    if repulsion_q_min >= 0:
        raise ValueError("repulsion_q_min >= 0: spectators TBI")
        q_rep = (qraw + repulsion_q_min)*(1.- is_spectator)
        q_kalpha_m += repulsion_q_min - q_min
    
    #now the bit that needs Mnot
    Mnot_distances = tf.expand_dims(x_kalpha_m, axis=1) #K x 1 x C
    Mnot_distances = Mnot_distances - tf.expand_dims(x, axis=0) #K x V x C
    
    rep_distances = tf.reduce_sum(Mnot_distances**2, axis=-1, keepdims=True)  #K x V x 1
    
    rep_distances = tf.math.divide_no_nan(rep_distances, 2.*distance_scale_kalpha_m_exp**2+1e-6)
    
    V_rep =  tf.math.exp(-rep_distances) #1. / (V_rep + 0.1) #-2.*tf.math.log(1.-tf.math.exp(-V_rep/2.)+1e-5)
    
    if super_repulsion:
        V_rep += 10.*tf.math.exp(-100.* tf.sqrt(rep_distances+1e-6))
        
    if div_repulsion:
        V_rep = 1. / (rep_distances + 0.1)
    
    #spec weights are in q
    V_rep *= M_not * tf.expand_dims(q_rep, axis=0) #K x V x 1
    V_rep = tf.reduce_sum(V_rep, axis=1) #K x 1
    
    V_rep *= object_weights_kalpha_m * q_kalpha_m #K x 1
    
    if weight_by_q:
        sumq = tf.reduce_sum(M_not * tf.expand_dims(q_rep, axis=0), axis=1)
        V_rep = tf.math.divide_no_nan(V_rep, sumq) # K x 1
    else:
        V_rep = tf.math.divide_no_nan(V_rep, 
                                  tf.expand_dims(tf.expand_dims(N,axis=0),axis=0) - N_per_obj+1e-9) # K x 1
    # opt used later in payload loss
    V_rep_K = V_rep
    V_rep = tf.math.divide_no_nan(tf.reduce_sum(V_rep,axis=0), K+1e-9) # 1
    
    B_pen = None
    
    def bpenhelp(b_m, exponent : int):
        b_mes = tf.reduce_sum(b_m**exponent, axis=1)
        if not exponent==1:
            b_mes = (b_mes+1e-16)**(1./float(exponent))
        return tf.math.log((1.-b_mes)**2+1.+1e-8) 
    
    if phase_transition:
    ## beta terms
        B_pen = - tf.reduce_sum(padmask_m * 1./(20.*distancesq_m + 1.),axis=1) # K x 1
        B_pen += 1. #remove self-interaction term (just for offset)
        B_pen *= object_weights_kalpha_m * beta_kalpha_m
        B_pen = tf.math.divide_no_nan(B_pen, N_per_obj+1e-9) # K x 1
        #now 'standard' 1-beta
        B_pen -= 0.2*object_weights_kalpha_m * (tf.math.log(beta_kalpha_m+1e-9))#tf.math.sqrt(beta_kalpha_m+1e-6) 
        #another "-> 1, but slower" per object
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(B_pen,axis=0), K+1e-9) # 1
    
    else:
        B_pen_po = object_weights_kalpha_m * (1. - beta_kalpha_m)
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(B_pen_po,axis=0), K+1e-9)#1
        #get out of random gradients in the beginning
        #introduces gradients on all betas of hits rather than just the max one
        B_up = tf.math.divide_no_nan(tf.reduce_sum((1.-is_noise)*(1.-beta_in)), N - tf.reduce_sum(is_noise) )
        B_pen += 0.01*B_pen*B_up #if it's high try to elevate all betas
        
    if cont_beta_loss:
        B_pen =   bpenhelp(beta_m,2) + bpenhelp(beta_m,4)
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(object_weights_kalpha_m * B_pen,axis=0), K+1e-9)   
    
    too_much_B_pen = object_weights_kalpha_m * bpenhelp(beta_m,1) #K x 1, don't make it steep
    too_much_B_pen = tf.math.divide_no_nan(tf.reduce_sum(too_much_B_pen), K+1e-9)
    
    Noise_pen = S_B*tf.math.divide_no_nan(tf.reduce_sum(is_noise * beta_in), tf.reduce_sum(is_noise)+1e-3)
    
    #explicit payload weight function here, the old one was odd
    
    #too aggressive scaling is bad for high learning rates. 
    p_w = padmask_m * tf.math.atanh(beta_m/1.002)**2 #this is well behaved
    
    if payload_beta_gradient_damping_strength > 0:
        p_w = payload_beta_gradient_damping_strength * tf.stop_gradient(p_w) + \
        (1.- payload_beta_gradient_damping_strength)* p_w
        
    payload_loss_m = p_w * SelectWithDefault(Msel, (1.-is_noise)*payload_loss, 0.) #K x V_perobj x P
    payload_loss_m = object_weights_kalpha_m * tf.reduce_sum(payload_loss_m, axis=1) # K x P
        
    #here normalisation per object
    payload_loss_m = tf.math.divide_no_nan(payload_loss_m, tf.reduce_sum(p_w, axis=1))
    
    #print('dynamic_payload_scaling_onset',dynamic_payload_scaling_onset)
    if dynamic_payload_scaling_onset > 0:
        #stop gradient
        V_scaler = tf.stop_gradient(V_rep_K + V_att_K)    # K x 1
        #print('N_per_obj[V_scaler=0]',N_per_obj[V_scaler==0])
        #max of V_scaler is around 1 given the potentials
        scaling = tf.exp(-tf.math.log(2.)*V_scaler/(dynamic_payload_scaling_onset/5.))
        #print('affected fraction',tf.math.count_nonzero(scaling>0.5,dtype='float32')/K,'max',tf.reduce_max(V_scaler,axis=0,keepdims=True))
        payload_loss_m *= scaling#basically the onset of the rise
    #pll = tf.math.divide_no_nan(payload_loss_m, N_per_obj+1e-9) # K x P #really?
    pll = tf.math.divide_no_nan(tf.reduce_sum(payload_loss_m,axis=0), K+1e-3) # P
    
    return V_att, V_rep, Noise_pen, B_pen, pll, too_much_B_pen



   
def oc_loss(
        x, 
        beta, 
        truth_indices, 
        row_splits, 
        is_spectator, 
        payload_loss,
        Q_MIN=0.1, 
        S_B=1.,
        noise_q_min=None,
        distance_scale=None,
        energyweights=None,
        use_average_cc_pos=False,
        payload_rel_threshold=0.1,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        payload_beta_gradient_damping_strength=0.,
        kalpha_damping_strength=0.,
        beta_gradient_damping=0.,
        repulsion_q_min=-1,
        super_repulsion=False,
        super_attraction=False,
        div_repulsion=False,
        dynamic_payload_scaling_onset=-0.1
        ):   
    
    if energyweights is None:
        energyweights = tf.zeros_like(beta)+1.
        
    if distance_scale is None:
        distance_scale = tf.zeros_like(beta)+1.
        
    if row_splits.shape[0] is None or row_splits.shape[0] < 2:
        return tf.constant(0,dtype='float32')
    batch_size = row_splits.shape[0] - 1
    
    
    
    V_att, V_rep, Noise_pen, B_pen, pll,to_much_B_pen = 6*[tf.constant(0., tf.float32)]
    
    for b in tf.range(batch_size):
        att,rep,noise,bp,pl,tmb = oc_per_batch_element(
            
            beta[row_splits[b]:row_splits[b + 1]],
            x[row_splits[b]:row_splits[b + 1]],
            
            Q_MIN,
            
            energyweights[row_splits[b]:row_splits[b + 1]],
            truth_indices[row_splits[b]:row_splits[b + 1]],
            is_spectator[row_splits[b]:row_splits[b + 1]],
            payload_loss[row_splits[b]:row_splits[b + 1]],
            
            payload_weight_threshold=payload_rel_threshold,
            
            use_mean_x=use_average_cc_pos,
            cont_beta_loss=cont_beta_loss,
            S_B=S_B,
            noise_q_min=noise_q_min,
            distance_scale=distance_scale,
            prob_repulsion=prob_repulsion,
            phase_transition=phase_transition,
            phase_transition_double_weight=phase_transition_double_weight,
            #alt_potential_norm=alt_potential_norm,
            payload_beta_gradient_damping_strength=payload_beta_gradient_damping_strength,
            kalpha_damping_strength=kalpha_damping_strength,
            beta_gradient_damping=beta_gradient_damping,
            repulsion_q_min=repulsion_q_min,
            super_repulsion=super_repulsion,
            super_attraction=super_attraction,
            div_repulsion=div_repulsion,
            dynamic_payload_scaling_onset=dynamic_payload_scaling_onset
            )
        V_att += att
        V_rep += rep
        Noise_pen += noise
        B_pen += bp
        pll += pl
        to_much_B_pen += tmb
    
    bsize = tf.cast(batch_size, dtype='float32') + 1e-3
    V_att /= bsize
    V_rep /= bsize
    Noise_pen /= bsize
    B_pen /= bsize
    pll /= bsize
    to_much_B_pen /= bsize
    
    return V_att, V_rep, Noise_pen, B_pen, pll, to_much_B_pen
