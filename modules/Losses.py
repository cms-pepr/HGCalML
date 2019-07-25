
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras
import keras.backend as K


def sortFractions(fracs, energies, to_sort):
    '''
    
    fracs      : B x V x Fracs
    energies   : B x V x 1
    to_sort    : B x V x 1
        
    '''
    frac_energies   = energies*fracs
    frac_sumenergy  = tf.reduce_sum(frac_energies, axis=1)
    
    #frac_energies    : B x V x Fracs
    #frac_sumenergy   : B x Fracs
    
    weighted_to_sort = tf.reduce_sum(frac_energies * to_sort, axis=1)/(frac_sumenergy+K.epsilon())
    
    #set the zero entries to something big to make them appear at the end of the list
    weighted_to_sort = tf.where(tf.abs(weighted_to_sort)>0., weighted_to_sort, tf.zeros_like(weighted_to_sort)+500.)
    
    ranked_to_sort, ranked_indices = tf.nn.top_k(-weighted_to_sort, tf.shape(fracs)[2])
    
    #ranked_indices = tf.Print(ranked_indices,[weighted_to_sort, ranked_to_sort],'weighted_to_sort, ranked_to_sort ', summarize=200)
    
    ranked_indices = tf.expand_dims(ranked_indices, axis=2)
    
    batch_range = tf.range(0, tf.shape(fracs)[0])
    batch_range = tf.expand_dims(batch_range, axis=1)
    batch_range = tf.expand_dims(batch_range, axis=1)
        
    batch_indices = tf.tile(batch_range, [1, tf.shape(fracs)[2], 1]) # B x Fracs x 1
    indices = tf.concat([batch_indices, ranked_indices], axis=-1) # B x Fracs x 2
    
    identity_matrix = tf.eye(tf.shape(fracs)[2]) #Fracs x Fracs 1 matrix
    identity_matrix = tf.expand_dims(identity_matrix, axis=0) # 1 x F x F
    identity_matrix = tf.tile(identity_matrix, [tf.shape(fracs)[0],1,1])  # B x F x F
    sorted_identity_matrix = tf.gather_nd(identity_matrix, indices) # B x F x F
    
    # sorted_identity_matrix : B x Fm x Fm
    # predicted_fracs        : B x V  x Ff
    
    # B x Fm x Fm --> B x V x Fm x Fm
    sorted_identity_matrix = tf.expand_dims(sorted_identity_matrix, axis=1)
    sorted_identity_matrix = tf.tile(sorted_identity_matrix, [1,tf.shape(fracs)[1],1,1])
    # B x V x Fm x Fm
    
    # predicted_fracs   : B x V  x Ff --> B x V x Ff x 1
    sorted_predicted_fractions = tf.expand_dims(fracs, axis=3)
    
    out = tf.squeeze(tf.matmul(sorted_identity_matrix, sorted_predicted_fractions), axis=-1)
    
    #out = tf.Print(out, [fracs[0,0,:], out[0,0,:]], 'in / out ', summarize=300)
    
    return out



def energy_weighting(e, usesqrt, weightfactor):
    
    if usesqrt:
        e = tf.sqrt(tf.abs(e)+K.epsilon())
    if weightfactor<=0:
        e = tf.zeros_like(e)+1.
    return e

def weighted_frac_loss( truth, pred, usesqrt, weightfactor=1., sort_truth_by_eta=False, sort_pred_by_eta=False):
    
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = truth[:,:,-1]
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    if sort_truth_by_eta or sort_pred_by_eta:
        etas      = tf.expand_dims(pred[:,:,-1], axis=2)
        if sort_pred_by_eta:
            p_sigfrac = sortFractions(p_sigfrac, tf.expand_dims(r_energy,axis=2), etas)
        if sort_truth_by_eta:
            t_sigfrac = sortFractions(t_sigfrac, tf.expand_dims(r_energy,axis=2), etas)
    
    r_energy  = energy_weighting(r_energy, usesqrt, weightfactor)
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_energy  = tf.expand_dims(r_energy, axis=2)*t_sigfrac
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    t_isnoise = (1.-t_issc)
    t_isnoise = tf.where(t_isnoise>0., t_isnoise, tf.zeros_like(t_isnoise))
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
    
    #penalty = tf.Print(penalty, [t_issc, r_energy], 't_issc, r_energy] ', summarize=200)
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
    return weighted_frac_loss(truth, pred, usesqrt=True,weightfactor=-1.)
global_loss_list['fraction_loss_noweight']=fraction_loss_noweight
    

def fraction_loss( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True)
global_loss_list['fraction_loss']=fraction_loss
    

def fraction_loss_sorted( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, sort_truth_by_eta=True)
global_loss_list['fraction_loss']=fraction_loss

def fraction_loss_sorted_all( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, sort_truth_by_eta=True, sort_pred_by_eta=True)
global_loss_list['fraction_loss']=fraction_loss

def fraction_loss_lin( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=False)
global_loss_list['fraction_loss_lin']=fraction_loss_lin

