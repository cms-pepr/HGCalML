# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import time


from Loss_tools import huber

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
    w = tf.clip_by_value(w, 0., 1.-1e-6)
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
        payload_weight_function = payload_weight_function,  #receives betas as K x V x 1 as input, and a threshold val
        payload_weight_threshold = 0.8,
        use_mean_x = False,
        cont_beta_loss=False,
        ):
    '''
    all inputs
    V x X , where X can be 1
    '''
    
    #set all spectators invalid here, everything scales with beta, so:
    beta_in = beta
    beta = tf.clip_by_value(beta, 1e-9,1.-1e-4)
    beta *= (1. - is_spectator)
    q = tf.math.atanh(beta)**2 + q_min * (1. - is_spectator) # V x 1
    
    
    #determine object associations
    obj_ids,_ = tf.unique(tf.squeeze(truth_idx,axis=1)) # K+1
    obj_ids = obj_ids[obj_ids>-0.1]
    
    K = tf.cast(obj_ids.shape[0], dtype='float32') 
    N = tf.cast(beta.shape[0], dtype='float32')
    
    #print('objects',K)
    
    obj_ids = tf.expand_dims(obj_ids, axis=1) #K x 1
    
    is_noise = tf.where(truth_idx<0., tf.zeros_like(truth_idx), 1.)#V x 1
    N_nonoise = N - tf.cast(tf.math.count_nonzero(is_noise), dtype='float32') #()
    
    M = tf.expand_dims(obj_ids, axis=1) - tf.expand_dims(truth_idx, axis=0) # K x V x 1
    M_not = tf.where(tf.abs(M) > 0.1, tf.zeros_like(M) + 1., tf.zeros_like(M))
    M = tf.where(tf.abs(M) < 0.1, tf.zeros_like(M) + 1., tf.zeros_like(M))
    
    
    kalpha = tf.argmax(M * tf.expand_dims(beta_in, axis=0), axis=1) # K x 1
    
    x_kalpha = gather_for_obj_from_vert(x, kalpha) # K x 1 x C
    if use_mean_x: #q weighted mean here
        x_kalpha = tf.reduce_sum( M * tf.expand_dims(q, axis=0) * tf.expand_dims(x,axis=0), axis=1, keepdims=True) # K x 1 x C
        x_kalpha = tf.math.divide_no_nan(x_kalpha, tf.reduce_sum(M * tf.expand_dims(q, axis=0), axis=1, keepdims=True)) # K x 1 x C
    
    q_kalpha = gather_for_obj_from_vert(q, kalpha) # K x 1 x 1
    
    beta_kalpha = gather_for_obj_from_vert(beta_in, kalpha) # K x 1 x 1
    object_weights_kalpha = gather_for_obj_from_vert(object_weights, kalpha)# K x 1 x 1
    
    distance = tf.sqrt(tf.reduce_sum((x_kalpha - tf.expand_dims(x, axis=0)) ** 2, axis=-1, keepdims=True) + 1e-6)# K x V x 1
    
    V_att = M * q_kalpha * tf.expand_dims(q, axis=0) * huber(distance, 8)  # K x V x 1
    V_att = mean_N_K(V_att, N, K) # ()
    
    V_rep = M_not * q_kalpha * tf.expand_dims(q, axis=0) * tf.nn.relu(1. - distance)   # K x V x 1
    V_rep = mean_N_K(V_rep, N, K) # ()
    
    ##beta penalty
    
    beta_kalpha_sm = beta_kalpha
    if cont_beta_loss:
        
        b_exp = M * tf.expand_dims(beta, axis=0)
        maxb = beta_kalpha
        meanb = tf.math.divide_no_nan(tf.reduce_sum(b_exp, axis = 1, keepdims=True),
                                      tf.reduce_sum(M,  axis = 1, keepdims=True))
        sqsum = tf.reduce_sum(b_exp**2, axis = 1, keepdims=True)
        beta_kalpha_sm = 1. - (tf.math.divide_no_nan(meanb+0.2, maxb+0.1) + tf.math.exp(-sqsum))
        #b_exp = M * tf.expand_dims(beta, axis=0)
        #b_exp_sum = tf.reduce_sum(b_exp, axis=1, keepdims=True) # K x 1 x 1
        #b_exp_prod = tf.reduce_prod(b_exp, axis=1, keepdims=True)  # K x 1 x 1
        #
        ## 1 - X here to compensate 1 - down there
        #beta_kalpha_sm = 1. - tf.math.exp(-b_exp_sum)+0.5*b_exp_prod-tf.math.exp(tf.constant([[[-1.]]])) # K x 1 x 1
        

    B_pen = tf.math.divide_no_nan(tf.reduce_sum(object_weights_kalpha*(1. - beta_kalpha_sm)), 
                                  tf.reduce_sum(object_weights_kalpha)) # ()
    
    # beta_in V x 1
    # object_weights V x 1
    
    to_much_B_pen = tf.constant([0.],dtype='float32')
    if not cont_beta_loss:
        to_much_B_pen = tf.reduce_sum( beta_in*object_weights ) - tf.reduce_sum(object_weights_kalpha)
        to_much_B_pen = tf.nn.relu(to_much_B_pen)#min=0
        to_much_B_pen = tf.math.divide_no_nan(to_much_B_pen,tf.reduce_sum(object_weights))
    
    ##noise penalty
    Noise_pen = tf.math.divide_no_nan(tf.reduce_sum(is_noise * beta_in), tf.reduce_sum(is_noise))
    
    #### payload
    ## beta_nograd kill beta gradient? maybe..?
    
    p_w = payload_weight_function(M * tf.expand_dims(beta,axis=0), 
                                  tf.expand_dims(object_weights,axis=0), 
                                  payload_weight_threshold) #K x V x 1
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
        cont_beta_loss=False
        ):   
    
    if energyweights is None:
        energyweights=tf.zeros_like(beta)+1.
        
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
            
            payload_weight_function=payload_weight_function,
            payload_weight_threshold=payload_rel_threshold,
            
            use_mean_x=use_average_cc_pos,
            cont_beta_loss=cont_beta_loss
            )
        V_att += att
        V_rep += rep
        Noise_pen += noise
        B_pen += bp
        pll += pl
        to_much_B_pen += tmb
    
    bsize = tf.cast(batch_size, dtype='float32') + 1e-6
    V_att /= bsize
    V_rep /= bsize
    Noise_pen /= bsize
    B_pen /= bsize
    pll /= bsize
    to_much_B_pen /= bsize
    
    return V_att, V_rep, Noise_pen, B_pen, pll, to_much_B_pen
