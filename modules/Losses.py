
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras
import keras.backend as K

def energy_weighting(e, usesqrt, weightfactor):
    
    if usesqrt:
        e = tf.sqrt(tf.abs(e)+K.epsilon())
    if weightfactor<=0:
        e = tf.zeros_like(e)+1.
    return e

def weighted_frac_loss( truth, pred, usesqrt, weightfactor=1.):
    
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = truth[:,:,-1]
    r_energy  = energy_weighting(r_energy, usesqrt, weightfactor)
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_energy  = tf.expand_dims(r_energy, axis=2)*t_sigfrac
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    t_isnoise = (1.-t_issc)
    r_sumnoise_energy = tf.reduce_sum(t_isnoise * r_energy, axis=-1)
    
    #t_sigfrac, p_sigfrac : B x V x Fracs
    #r_energy             : B x V 
    #t_issc               : B x V  
    #t_energy             : B x V x Fracs
    #t_sumenergy          : B x Fracs
    #t_isnoise            : B x V
    #r_sumnoise_energy    : B
    
    # B x Fracs
    sc_loss   = tf.reduce_sum(t_energy * (t_sigfrac-p_sigfrac)**2, axis=1)/(t_sumenergy+K.epsilon())
    # B : mean over sim clusters
    sc_loss   = tf.reduce_mean(sc_loss, axis=-1)
    # B 
    rest_loss = tf.reduce_sum(tf.reduce_sum(tf.expand_dims(t_isnoise,axis=2) * tf.expand_dims(r_energy,axis=2) * (t_sigfrac-p_sigfrac)**2, axis=1),axis=1)\
                         /(r_sumnoise_energy+K.epsilon())
    
    ## additional penalties
    
    # B x V
    sum_p_sigfrac = tf.reduce_sum(p_sigfrac, axis=-1)
    penalty = r_energy * ( t_issc *  (sum_p_sigfrac - 1.)**2 + t_isnoise * sum_p_sigfrac**2)
    penalty = tf.reduce_mean(penalty, axis=1)
    
    
    ## weighting goes here
    loss = 2. * sc_loss + 0.1 * rest_loss + 2. * penalty
    
    loss   = tf.Print(loss,[tf.reduce_mean(loss), 
                            tf.reduce_mean(sc_loss),
                            tf.reduce_mean(rest_loss),
                            tf.reduce_mean(penalty),
                            tf.reduce_mean(t_issc *  (sum_p_sigfrac - 1.)),
                            tf.reduce_mean(t_isnoise *  (sum_p_sigfrac)),
                            ],
                            'loss, sc_loss, rest_loss, penalty, mean err(pred fracs) SC, mean err(pred fracs) Noise ')
    
    return loss
    

def fraction_loss_noweight( truth, pred):
    return weighted_frac_loss(truth, pred, True,-1.)
global_loss_list['fraction_loss_noweight']=fraction_loss_noweight
    

def fraction_loss( truth, pred):
    return weighted_frac_loss(truth, pred, True)
global_loss_list['fraction_loss']=fraction_loss
    

def fraction_loss_lin( truth, pred):
    return weighted_frac_loss(truth, pred, False)
global_loss_list['fraction_loss_lin']=fraction_loss_lin

