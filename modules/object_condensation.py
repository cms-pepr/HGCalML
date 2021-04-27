# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import time
from oc_helper_ops import CreateMidx, SelectWithDefault


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
    

def oc_per_batch_element(
        beta,
        x,
        q_min,
        object_weights, # V x 1 !!
        truth_idx,
        is_spectator,
        payload_loss,
        S_B=1.,
        payload_weight_function = None,  #receives betas as K x V x 1 as input, and a threshold val
        payload_weight_threshold = 0.8,
        use_mean_x = False,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        alt_potential_norm=False,
        cut_payload_beta_gradient=False
        ):
    '''
    all inputs
    V x X , where X can be 1
    '''
    
    if not alt_potential_norm:
        raise ValueError("not alt_potential_norm not implemented")
    if not prob_repulsion:
        raise ValueError("not prob_repulsion not implemented")
    if not phase_transition:
        raise ValueError("not phase_transition not implemented")
    if phase_transition_double_weight:
        raise ValueError("phase_transition_double_weight not implemented")
    if cont_beta_loss:
        raise ValueError("cont_beta_loss not implemented")
    if payload_weight_function is not None:
        raise ValueError("payload_weight_function not implemented")
        
        
    
    #set all spectators invalid here, everything scales with beta, so:
    beta_in = beta
    beta = tf.clip_by_value(beta, 0.,1.-1e-4)
    beta *= (1. - is_spectator)
    qraw = tf.math.atanh(beta)**2 
    q = qraw + q_min * (1. - is_spectator) # V x 1
    #q = tf.where(beta_in<1.-1e-4, q, tf.math.atanh(1.-1e-4)**2 + q_min + beta_in) #just give the rest above clip a gradient
    
    N = tf.cast(beta.shape[0], dtype='float32')
    is_noise = tf.where(truth_idx<0, tf.zeros_like(truth_idx,dtype='float32'), 1.)#V x 1
    
    Msel, M_not, N_per_obj = CreateMidx(truth_idx, calc_m_not=True)
    
    N_per_obj = tf.cast(N_per_obj, dtype='float32') # K x 1
    
    K = tf.cast(Msel.shape[0], dtype='float32') 
    
    padmask_m = SelectWithDefault(Msel, tf.zeros_like(beta_in)+1., 0) #K x V-obj x 1
    x_m = SelectWithDefault(Msel, x, 0.) #K x V-obj x C
    beta_m = SelectWithDefault(Msel, beta_in, 0.) #K x V-obj x 1
    q_m = SelectWithDefault(Msel, q, 0.)#K x V-obj x 1
    object_weights_m = SelectWithDefault(Msel, object_weights, 0.)
    
    kalpha_m = tf.argmax(beta_m, axis=1) # K x 1
    
    x_kalpha_m = tf.gather_nd(x_m,kalpha_m, batch_dims=1) # K x C
    if use_mean_x:
        x_kalpha_m = tf.reduce_sum(q_m * x_m * padmask_m,axis=1) # K x C
        x_kalpha_m = tf.math.divide_no_nan(x_kalpha_m, tf.reduce_sum(q_m * padmask_m, axis=1))
    
    q_kalpha_m = tf.gather_nd(q_m,kalpha_m, batch_dims=1) # K x 1
    beta_kalpha_m = tf.gather_nd(beta_m,kalpha_m, batch_dims=1) # K x 1
    
    object_weights_kalpha_m = tf.gather_nd(object_weights_m,kalpha_m, batch_dims=1) # K x 1
    
    distancesq_m = tf.reduce_sum( (tf.expand_dims(x_kalpha_m, axis=1) - x_m)**2, axis=-1, keepdims=True) #K x V-obj x 1
    V_att = q_m * tf.expand_dims(q_kalpha_m,axis=1) * distancesq_m #K x V-obj x 1
    V_att = V_att * tf.expand_dims(object_weights_kalpha_m,axis=1) #K x V-obj x 1
    
    V_att = tf.math.divide_no_nan(tf.reduce_sum(padmask_m * V_att,axis=1), N_per_obj) # K x 1
    V_att = tf.math.divide_no_nan(tf.reduce_sum(V_att,axis=0), K) # 1
    
    
    #now the bit that needs Mnot
    V_rep = tf.expand_dims(x_kalpha_m, axis=1) #K x 1 x C
    V_rep = V_rep - tf.expand_dims(x, axis=0) #K x V x C
    V_rep = tf.reduce_sum(V_rep**2, axis=-1, keepdims=True)  #K x V x 1
    
    V_rep = -2.*tf.math.log(1.-tf.math.exp(-V_rep/2.)+1e-5)
    V_rep *= M_not * tf.expand_dims(q, axis=0) #K x V x 1
    V_rep = tf.reduce_sum(V_rep, axis=1) #K x 1
    
    V_rep *= object_weights_kalpha_m * q_kalpha_m #K x 1
    
    V_rep = tf.math.divide_no_nan(V_rep, 
                                  tf.expand_dims(tf.expand_dims(N,axis=0),axis=0) - N_per_obj) # K x 1
    V_rep = tf.math.divide_no_nan(tf.reduce_sum(V_rep,axis=0), K) # 1
    
    
    ## beta terms
    B_pen = - tf.reduce_sum(padmask_m * 1./(20.*distancesq_m + 1.),axis=1) # K x 1
    B_pen += 1. #remove self-interaction term (just for offset)
    B_pen *= object_weights_kalpha_m * beta_kalpha_m
    B_pen = tf.math.divide_no_nan(B_pen, N_per_obj) # K x 1
    #now 'standard' 1-beta
    #B_pen -= object_weights_kalpha_m * tf.math.sqrt(beta_kalpha_m+1e-6) 
    #another "-> 1, but slower" per object
    B_pen = tf.math.divide_no_nan(tf.reduce_sum(B_pen,axis=0), K) # 1
    
    
    too_much_B_pen = tf.constant([0.],dtype='float32')
    
    Noise_pen = S_B*tf.math.divide_no_nan(tf.reduce_sum(is_noise * beta_in), tf.reduce_sum(is_noise))
    
    #explicit payload weight function here, the old one was odd
    
    p_w = tf.math.atanh(padmask_m * tf.clip_by_value(beta_m, 1e-4, 1.-1e-4))**2 #already zero-padded  , K x V_perobj x 1
    p_w = tf.math.divide_no_nan(p_w, tf.reduce_max(p_w, axis=1, keepdims=True)) #normalise to maximum
    
    if cut_payload_beta_gradient:
        p_w = tf.stop_gradient(p_w)
        
    payload_loss_m = p_w * SelectWithDefault(Msel, payload_loss, 0.) #K x V_perobj x P
    payload_loss_m = tf.reduce_sum(payload_loss_m, axis=1)
    
    pll = tf.math.divide_no_nan(payload_loss_m, N_per_obj) # K x P
    pll = tf.math.divide_no_nan(tf.reduce_sum(pll,axis=0), K) # P
    
    return V_att, V_rep, Noise_pen, B_pen, pll, too_much_B_pen



def oc_per_batch_element_old(
        beta,
        x,
        q_min,
        object_weights, # V x 1 !!
        truth_idx,
        is_spectator,
        payload_loss,
        S_B=1.,
        payload_weight_function = payload_weight_function,  #receives betas as K x V x 1 as input, and a threshold val
        payload_weight_threshold = 0.8,
        use_mean_x = False,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        alt_potential_norm=False,
        cut_payload_beta_gradient=False
        ):
    '''
    all inputs
    V x X , where X can be 1
    '''
    
    #set all spectators invalid here, everything scales with beta, so:
    beta_in = beta
    beta = tf.clip_by_value(beta, 0.,1.-1e-4)
    beta *= (1. - is_spectator)
    qraw = tf.math.atanh(beta)**2 
    q = qraw + q_min * (1. - is_spectator) # V x 1
    #q = tf.where(beta_in<1.-1e-4, q, tf.math.atanh(1.-1e-4)**2 + q_min + beta_in) #just give the rest above clip a gradient
    
    #determine object associations
    obj_ids,_ = tf.unique(tf.squeeze(truth_idx,axis=1)) # K+1
    obj_ids = obj_ids[obj_ids>=0]
    
    K = tf.cast(obj_ids.shape[0], dtype='float32') 
    N = tf.cast(beta.shape[0], dtype='float32')
    
    #print('objects',K)
    
    obj_ids = tf.expand_dims(obj_ids, axis=1) #K x 1
    
    is_noise = tf.where(truth_idx<0, tf.zeros_like(truth_idx,dtype='float32'), 1.)#V x 1
    N_nonoise = N - tf.cast(tf.math.count_nonzero(is_noise), dtype='float32') #()
    
    M = tf.expand_dims(obj_ids, axis=1) - tf.expand_dims(truth_idx, axis=0) # K x V x 1
    M_not = tf.where(tf.abs(M) > 0, tf.zeros_like(M,dtype='float32') + 1, tf.zeros_like(M,dtype='float32'))
    M = tf.where(tf.abs(M) == 0, tf.zeros_like(M,dtype='float32') + 1, tf.zeros_like(M,dtype='float32'))
    N_per_obj = tf.cast(tf.reduce_sum(M, axis=1), dtype='float32') # K x 1
    
    kalpha = tf.argmax(M * tf.expand_dims(beta_in, axis=0), axis=1) # K x 1
    
    x_kalpha = gather_for_obj_from_vert(x, kalpha) # K x 1 x C
    if use_mean_x: #q weighted mean here
        x_kalpha = tf.reduce_sum( M * tf.expand_dims(q, axis=0) * tf.expand_dims(x,axis=0), axis=1, keepdims=True) # K x 1 x C
        x_kalpha = tf.math.divide_no_nan(x_kalpha, tf.reduce_sum(M * tf.expand_dims(q, axis=0), axis=1, keepdims=True)) # K x 1 x C
    
    q_kalpha = gather_for_obj_from_vert(q, kalpha) # K x 1 x 1
    qraw_kalpha = gather_for_obj_from_vert(qraw, kalpha) # K x 1 x 1
    beta_kalpha = gather_for_obj_from_vert(beta_in, kalpha) # K x 1 x 1
    
    object_weights_kalpha = gather_for_obj_from_vert(object_weights, kalpha)# K x 1 x 1
    
    distancesq = tf.reduce_sum((x_kalpha - tf.expand_dims(x, axis=0)) ** 2, axis=-1, keepdims=True)# K x V x 1
    
    V_att = object_weights_kalpha * M * q_kalpha * tf.expand_dims(q, axis=0) * distancesq # K x V x 1
    
    if alt_potential_norm:
        V_att = tf.math.divide_no_nan(tf.reduce_sum(V_att,axis=1), N_per_obj) # K x 1
        V_att = tf.math.divide_no_nan(tf.reduce_sum(V_att,axis=0), K) # 1
    else:
        V_att = mean_N_K(V_att, N, K) # ()
    
    V_rep=None
    if prob_repulsion:
        # comes from 1-Gaus in LH space
        V_rep = -2.*tf.math.log(1.-tf.math.exp(-distancesq/2.)+1e-5)
        #V_rep = tf.exp(- distancesq * 6./2.)
    else:
        V_rep = tf.nn.relu(1. - tf.sqrt(distancesq + 1e-4))
    
    V_rep = object_weights_kalpha * V_rep * M_not * q_kalpha * tf.expand_dims(q, axis=0)     # K x V x 1
    
    if alt_potential_norm:
        V_rep = tf.math.divide_no_nan(tf.reduce_sum(V_rep,axis=1), 
                                      tf.expand_dims(tf.expand_dims(N,axis=0),axis=0)-N_per_obj) # K x 1
        V_rep = tf.math.divide_no_nan(tf.reduce_sum(V_rep,axis=0), K) # 1
    else:
        V_rep = mean_N_K(V_rep, N, K) # ()
    
    ##beta penalty
    B_pen = 0.

    if cont_beta_loss:
        assert not phase_transition

        c = M * tf.expand_dims(beta_in, axis=0) #K x V x 1
        c = tf.reduce_max(c,axis=1)*tf.reduce_max(tf.nn.softmax(c,axis=1),axis=1)
        c = tf.expand_dims(0.5/(c+0.25)-1. ,axis=2) #K x 1 x 1
        c *= object_weights_kalpha #K x 1 x 1
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(c), K) 
        #b_exp = M * tf.expand_dims(beta, axis=0)
        #maxb = beta_kalpha
        #meanb = tf.math.divide_no_nan(tf.reduce_sum(b_exp, axis = 1, keepdims=True),
        #                              tf.reduce_sum(M,  axis = 1, keepdims=True))
        #sqsum = tf.reduce_sum(b_exp**2, axis = 1, keepdims=True)
        #beta_kalpha_sm = 1. - (tf.math.divide_no_nan(meanb+0.2, maxb+0.1) + tf.math.exp(-sqsum))
        #b_exp = M * tf.expand_dims(beta, axis=0)
        #b_exp_sum = tf.reduce_sum(b_exp, axis=1, keepdims=True) # K x 1 x 1
        #b_exp_prod = tf.reduce_prod(b_exp, axis=1, keepdims=True)  # K x 1 x 1
        #
        ## 1 - X here to compensate 1 - down there
        #beta_kalpha_sm = 1. - tf.math.exp(-b_exp_sum)+0.5*b_exp_prod-tf.math.exp(tf.constant([[[-1.]]])) # K x 1 x 1
        
    
    if phase_transition:
        # K x V x 1
        # does not scale with q of each vertex, but only with beta_kalpha
        #
        # for double scale phase transition also scale with linear beta, 
        # add qmin and allow for self-potential (more smooth)
        #
        B_pen = 0.#tf.exp(-10. * tf.sqrt(distancesq + 1e-9))
        if phase_transition_double_weight:
            B_pen = - object_weights_kalpha * M * (1. + beta_kalpha) * tf.math.exp(-20*distancesq)
            B_pen *= 1. + tf.expand_dims(beta,axis=0)
        else:
            B_pen = - object_weights_kalpha * M * beta_kalpha * 1./(20*distancesq + 1.)
            B_pen = tf.where(distancesq==0. , 0., B_pen) #exclude exact self-potential (not needed in the other cases)
        #B_pen = mean_N_K(B_pen, N, K)
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(B_pen,axis=1), N_per_obj) # K x 1
        B_pen = tf.math.divide_no_nan(tf.reduce_sum(B_pen,axis=0), K) # 1
    else:
        B_pen += tf.math.divide_no_nan(tf.reduce_sum(object_weights_kalpha*(1. - beta_kalpha)), 
                                  tf.reduce_sum(object_weights_kalpha)) # ()
        

    # beta_in V x 1
    # object_weights V x 1
    
    to_much_B_pen = tf.constant([0.],dtype='float32')
    if not cont_beta_loss:
        to_much_B_pen = tf.reduce_sum( beta_in*object_weights ) - tf.reduce_sum(object_weights_kalpha)
        to_much_B_pen = tf.nn.relu(to_much_B_pen)#min=0
        to_much_B_pen = tf.math.divide_no_nan(to_much_B_pen,tf.reduce_sum(object_weights))
    
    ##noise penalty
    Noise_pen = S_B*tf.math.divide_no_nan(tf.reduce_sum(is_noise * beta_in), tf.reduce_sum(is_noise))
    
    #### payload
    ## beta_nograd kill beta gradient? maybe..?
    
    p_w = object_weights_kalpha * payload_weight_function(M * tf.expand_dims(beta,axis=0), 
                                  tf.expand_dims(object_weights,axis=0), 
                                  payload_weight_threshold) #K x V x 1
    if cut_payload_beta_gradient:
        p_w = tf.stop_gradient(p_w)
    pll = p_w * tf.expand_dims(payload_loss, axis=0) #K x V x X
    pll = tf.math.divide_no_nan(
        tf.reduce_sum(pll,axis=[0,1]), K  )# (), weights are already normalised in V
    
    return V_att, V_rep, Noise_pen, B_pen, pll, to_much_B_pen


   
def oc_loss(
        x, 
        beta, 
        truth_indices, 
        row_splits, 
        is_spectator, 
        payload_loss,
        Q_MIN=0.1, 
        S_B=1.,
        energyweights=None,
        use_average_cc_pos=False,
        payload_rel_threshold=0.1,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        alt_potential_norm=False,
        cut_payload_beta_gradient=False,
        ):   
    
    if energyweights is None:
        energyweights=tf.zeros_like(beta)+1.
        
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
            prob_repulsion=prob_repulsion,
            phase_transition=phase_transition,
            phase_transition_double_weight=phase_transition_double_weight,
            alt_potential_norm=alt_potential_norm,
            cut_payload_beta_gradient=cut_payload_beta_gradient
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
