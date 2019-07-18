
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
    
    ## old
    
    
    
    #pred can contain more stuff, just take the first entries
    # the LAST entry of truth is energy
    sigfrac = truth[:,:,0:-1]
    #sigfrac = tf.Print(sigfrac,[sigfrac],'sigfrac',summarize=300)
    e_weight = truth[:,:,-1]
    non_exp_weight = e_weight
    e_weight = tf.expand_dims(e_weight, axis=2)
    #e_weight = tf.Print(e_weight,[e_weight],'e_weight',summarize=300)
    weight=e_weight
    if usesqrt:
        weight = tf.sqrt(tf.abs(weight)+K.epsilon())
    if weightfactor<=0:
        weight = tf.zeros_like(weight)+1.
    
    
    
    # 
    
    asso_weight = tf.squeeze(tf.zeros_like(weight)+1., axis=-1)
    #asso_weight = tf.where(non_exp_weight>0., asso_weight, tf.zeros_like(asso_weight))
    asso_weight = non_exp_weight
    
    asso_norm = tf.reduce_sum(asso_weight, axis=1) + K.epsilon()
    
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    
    ### p_sigfrac       : B x R x Frac (20)
    ### sigfrac         : B x R x Frac (20)
    ### weight          : B x R x 1
    ### non_exp_weight  : B x R 
    
    
    ### penalise all-zero predictions
    
    is_sc = tf.reduce_sum(sigfrac,axis=-1) #1 for SC, 0 else
    
    sim_associated_penalty =  is_sc  *(tf.reduce_sum(p_sigfrac,axis=-1) - 1.)
                                      
    no_sim_penalty    = (1. - is_sc )  * (tf.reduce_sum(p_sigfrac,axis=-1))
    
    #sim_associated_penalty = tf.Print(sim_associated_penalty,[tf.reduce_mean(sim_associated_penalty),
    #                                                          tf.reduce_mean(no_sim_penalty)],
    #                                  'mean(sim_associated_penalty),mean(no_sim_penalty)')

    sim_associated_penalty = tf.reduce_sum(asso_weight*sim_associated_penalty, axis=1)/asso_norm
    no_sim_penalty = tf.reduce_sum(asso_weight*no_sim_penalty, axis=1)/asso_norm
    
    
    #sim_associated_penalty = tf.Print(sim_associated_penalty,[tf.reduce_mean(sim_associated_penalty),
    #                                                          tf.reduce_mean(no_sim_penalty)],
    #                                  'weighted sim_associated_penalty,weighted no_sim_penalty')
    
    asso_penalty = 10. * tf.reduce_mean(sim_associated_penalty**2+no_sim_penalty**2)
    
    
    diffs = sigfrac - p_sigfrac
    
    diffsw = diffs * diffs * weight
    
    e_weight = tf.tile(e_weight, [1,1,tf.shape(sigfrac)[-1]])
    diffs  = tf.where(e_weight>0, diffs,  tf.zeros_like(diffs))
    diffsw = tf.where(e_weight>0, diffsw, tf.zeros_like(diffsw))
    
    sumsig = tf.reduce_sum(sigfrac*weight, axis=1) + K.epsilon()
    
    #sumsig = tf.Print(sumsig,[sumsig],'sumsig')
    
    sumdiffs = tf.reduce_sum(diffsw, axis=1)
    
    #sumdiffs = tf.Print(sumdiffs,[sumdiffs],'sumdiffs')
    #sumsig = tf.Print(sumsig,[sumsig],'sumsig')
    
    #this loss is a bit different from the paper one
    loss =   tf.reduce_mean(sumdiffs/sumsig) + asso_penalty#tf.sign(sumsig)*sumdiffs/sumsig) #the sign masks out the zeros
    
    
    #########
    
    
    loss = tf.Print(loss,[asso_penalty,loss],'assopen, loss')
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

