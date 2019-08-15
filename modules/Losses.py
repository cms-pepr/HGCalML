
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras
import keras.backend as K

from Loss_tools import create_loss_dict,sortFractions, deltaPhi, deltaR2, makeDR2Matrix, weightedCenter, makeDR2Matrix_SC_hits



def scale_weight(weight, scale):
    s = (scale+0.5*scale)/scale
    weight = s * scale*(weight-0.5)/(1. + scale* (tf.abs(weight-0.5)+K.epsilon())) + 0.5
    weight = tf.clip_by_value(weight, 0.+K.epsilon(), 1.-K.epsilon())
    return weight

def applyMatrixToFracs(matrix, fracs):
    
    matrix = tf.expand_dims(matrix, axis=1)
    matrix = tf.tile(matrix, [1,tf.shape(fracs)[1],1,1])
    # B x V x Fm x Fm
    # predicted_fracs   : B x V  x Ff --> B x V x Ff x 1
    fracs = tf.expand_dims(fracs, axis=3)
    
    fracs = tf.matmul(matrix, fracs)
    fracs = tf.squeeze(fracs, axis=-1)
    
    return fracs

def _DR_loss( t_sigfrac, r_energy, p_sigfrac, etas, phis , scaling = 200.):
    '''
    Expects:
    
    t_sigfrac :  B x V x F
    r_energy  :  B x V x 1
    p_sigfrac :  B x V x F
    etas      :  B x V x 1
    phis      :  B x V x 1
    '''
    #t_sigfrac  = tf.Print(t_sigfrac, [tf.shape(t_sigfrac), 't_sigfrac s ']  )
    #r_energy   = tf.Print(r_energy, [tf.shape(r_energy), 'r_energy s ']     )
    #p_sigfrac  = tf.Print(p_sigfrac, [tf.shape(p_sigfrac), 'p_sigfrac s ']  )
    #etas       = tf.Print(etas, [tf.shape(etas), 'etas s ']                 )
    #phis       = tf.Print(phis, [tf.shape(phis), 'phis s ']                 )
    
    t_sigfrac = sortFractions(t_sigfrac, tf.expand_dims(r_energy,axis=2), etas)
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_issc    = tf.where(t_issc>0., t_issc, tf.zeros_like(t_issc))
    t_issc    = tf.where(t_issc>1., tf.zeros_like(t_issc), t_issc)
    
    sum_p_sigfrac = tf.reduce_sum(p_sigfrac, axis=-1)
    penalty = ( t_issc *  (sum_p_sigfrac - 1.)**2 + (1.-t_issc) * sum_p_sigfrac**2)
    penalty = tf.reduce_mean(penalty, axis=1)
    
    
    t_eta = weightedCenter(r_energy, t_sigfrac, etas)
    t_phi = weightedCenter(r_energy, t_sigfrac, phis, isPhi=True)
    
    r_eta = weightedCenter(r_energy, p_sigfrac, etas)
    r_phi = weightedCenter(r_energy, p_sigfrac, phis, isPhi=True)
    
    #DR2   = makeDR2Matrix(t_eta,t_phi,r_eta,r_phi) # B x F x F
    #DR2 = tf.exp(-DR2*scaling)
    t_DR2 = makeDR2Matrix(t_eta,t_phi,t_eta,t_phi)
    t_DR2 = tf.exp(-t_DR2*scaling) 
    
    
    r_energy = tf.sqrt(r_energy + K.epsilon())
    t_energy  =   r_energy * t_sigfrac
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    
    p_sigfrac = applyMatrixToFracs(t_DR2,  p_sigfrac)
    t_sigfrac = applyMatrixToFracs(t_DR2, t_sigfrac)
    sc_loss   = tf.reduce_sum(t_energy * (t_sigfrac-p_sigfrac)**2, axis=1)/(t_sumenergy+K.epsilon())
    sc_loss   = tf.reduce_mean(sc_loss, axis=-1)
    
    loss = 1. * sc_loss + 2. * penalty 
    
    loss   = tf.Print(loss,[tf.reduce_mean(loss), 
                            tf.reduce_mean(sc_loss),
                            tf.reduce_mean(penalty),
                            tf.reduce_mean(t_issc *    (sum_p_sigfrac - 1.)),
                            tf.reduce_mean((1. - t_issc) *  (sum_p_sigfrac)),
                            ],
                            'loss, sc_loss, penalty, n_loss, mean err(pred fracs) SC, mean err(pred fracs) Noise ')
    
    return tf.reduce_mean(loss)
    





def _simple_energy_loss(r_energy, t_sigfrac, p_sigfrac):
    
    #t_sigfrac, p_sigfrac : B x V x Fracs
    #r_energy             : B x V x 1
    
    r_energy = energy_weighting(r_energy,usesqrt=True)
    tot_energy = tf.squeeze(tf.reduce_sum(r_energy, axis=1), axis=1)
    
    n_withen = tf.squeeze(tf.cast(tf.count_nonzero(r_energy, axis=1), dtype='float32'), axis=1)
    #B ?
    
    diff = (t_sigfrac - p_sigfrac)**2
    mean_diff = tf.reduce_mean(tf.reduce_sum(diff,axis=1),axis=1) / (n_withen+K.epsilon())
    w_diff = r_energy * diff
    #B x V x Fracs
    print(w_diff.shape)
    w_diff = tf.reduce_mean(w_diff, axis=2)#over fracs
    print(w_diff.shape)
    w_diff = tf.reduce_sum(w_diff, axis=1)
    print(w_diff.shape)
    print('n_withen',n_withen.shape)
    
    #w_diff = w_diff / (n_withen + K.epsilon())
    w_diff = w_diff / (tot_energy+ K.epsilon())
    w_diff = tf.Print(w_diff,[tf.reduce_mean(w_diff), tf.reduce_mean(mean_diff)],'loss, avg fracdiff ')
    return w_diff
    
def simple_energy_loss( truth, pred):
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = tf.expand_dims(truth[:,:,-1],axis=2)
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    etas      = pred[:,:,tf.shape(truth)[2]:tf.shape(truth)[2]+1]
    t_sigfrac = sortFractions(t_sigfrac, r_energy , etas)
    
    return _simple_energy_loss(r_energy, t_sigfrac, p_sigfrac)
    

def weighted_frac_loss( truth, pred, usesqrt, weightfactor=1., 
                        sort_truth_by_eta=False, 
                        sort_pred_by_eta=False,
                        eta_penalty=False,
                        card_penalty=False,
                        distance_pen=False,
                        focus_on_multi_showers=False):
    
    
    
    
    
    
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = truth[:,:,-1]
    unweighted_r_energy = r_energy
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    
    
    etas      = pred[:,:,tf.shape(pred)[2]-2:tf.shape(pred)[2]-1] #second but last
    
    if sort_truth_by_eta or sort_pred_by_eta:
        if sort_pred_by_eta:
            p_sigfrac = sortFractions(p_sigfrac, tf.expand_dims(r_energy,axis=2), etas)
        if sort_truth_by_eta:
            t_sigfrac = sortFractions(t_sigfrac, tf.expand_dims(r_energy,axis=2), etas)
    
    r_energy  = energy_weighting(r_energy, usesqrt, weightfactor)
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_issc    = tf.where(t_issc>0., t_issc, tf.zeros_like(t_issc))
    t_issc    = tf.where(t_issc>1., tf.zeros_like(t_issc), t_issc)
    t_energy  = tf.expand_dims(r_energy, axis=2)*t_sigfrac
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    t_isnoise = (1.-t_issc)
    t_isnoise = tf.where(t_isnoise>0., t_isnoise, tf.zeros_like(t_isnoise))
    t_isnoise = tf.where(t_isnoise>1., tf.zeros_like(t_isnoise)+1., t_isnoise)
    r_sumnoise_energy = tf.reduce_sum(t_isnoise * r_energy, axis=-1)
    
    n_unmasked = tf.cast(tf.count_nonzero(r_energy, axis=1), dtype='float32')
    print(n_unmasked.shape)
    
    n_true = tf.cast(tf.count_nonzero(tf.reduce_sum(t_sigfrac, axis=1), axis=1), dtype='float32')
        
    #t_sigfrac, p_sigfrac : B x V x Fracs
    #r_energy             : B x V 
    #t_issc               : B x V  
    #t_energy             : B x V x Fracs
    #t_sumenergy          : B x Fracs
    #t_isnoise            : B x V
    #r_sumnoise_energy    : B
    #n_unmasked           : B
    
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
    loss = 2. * sc_loss + 0.0001 * rest_loss + 1. * penalty
    
    if focus_on_multi_showers:
        loss *= n_true
    
    
    if eta_penalty:
        #Inputs:
        #energy: B x V x 1
        #fracs:  B x V x F
        #var:    B x V x 1
        unweighted_r_energy = tf.expand_dims(unweighted_r_energy,axis=2)
        pred_eta = weightedCenter(unweighted_r_energy,p_sigfrac, etas)
        true_eta = weightedCenter(unweighted_r_energy,t_sigfrac, etas)
        
        diffsq = (pred_eta-true_eta)**2
        diffsq = tf.where(tf.abs(true_eta) > 1.,diffsq , tf.zeros_like(diffsq))
        
        eta_penalty_val = tf.reduce_sum(diffsq, axis=1) / (tf.cast(tf.count_nonzero(diffsq, axis=1), dtype='float32')+K.epsilon())
        
        loss = tf.reduce_mean(loss) +  0.1 * tf.reduce_mean(eta_penalty_val)
        
        loss   = tf.Print(loss,[tf.reduce_mean(loss), 
                            tf.reduce_mean(sc_loss),
                            tf.reduce_mean(rest_loss),
                            tf.reduce_mean(penalty),
                            tf.reduce_mean(t_issc *  (sum_p_sigfrac - 1.)),
                            tf.reduce_mean(tf.abs(eta_penalty_val)),
                            ],
                            'loss, sc_loss, rest_loss, penalty, mean err(pred fracs) SC, abs sqrt eta_penalty_val ')
    
        
    else:
        loss   = tf.Print(loss,[tf.reduce_mean(loss), 
                            tf.reduce_mean(sc_loss),
                            tf.reduce_mean(rest_loss),
                            tf.reduce_mean(penalty),
                            tf.reduce_mean(t_issc *  (sum_p_sigfrac - 1.)),
                            tf.reduce_mean(t_isnoise *  (sum_p_sigfrac)),
                            ],
                            'loss, sc_loss, rest_loss, penalty, mean err(pred fracs) SC, mean err(pred fracs) Noise ')
    
    if card_penalty:
        n_pred = tf.reduce_max(p_sigfrac, axis=1)
        n_pred = tf.where(n_pred > 0.2, n_pred, tf.zeros_like(n_pred))
        n_pred = tf.cast(tf.count_nonzero(n_pred, axis=1), dtype='float32')
        card_penalty_val = (n_true-n_pred)**2 / (n_true+K.epsilon())
        loss += 0.001 * tf.reduce_mean(card_penalty_val)
        loss = tf.Print(loss,[card_penalty_val, n_pred, n_true],' card_penalty_val, n_pred, n_true ')
        
    if distance_pen:
        from Loss_tools import weightedCoordLoss
        last_four_coords= pred[:,:,tf.shape(pred)[2]-(2+4):tf.shape(pred)[2]-2]
        coordloss = tf.reduce_mean(weightedCoordLoss(t_sigfrac, r_energy, last_four_coords))
        loss += 10. * coordloss
        loss = tf.Print(loss,[coordloss, loss],'coordloss, total loss ')
        
    
    return tf.reduce_mean(loss)
    

def fraction_loss_noweight( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True,weightfactor=-1.,sort_truth_by_eta=True)
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


def fraction_loss_eta_penalty( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, eta_penalty=True, sort_truth_by_eta=True)
global_loss_list['fraction_loss_eta_penalty']=fraction_loss_eta_penalty


def fraction_loss_eta_penalty_lin( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=False, eta_penalty=True, sort_truth_by_eta=True)
global_loss_list['fraction_loss_eta_penalty_lin']=fraction_loss_eta_penalty_lin


def fraction_loss_eta_penalty_distance_pen( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, eta_penalty=True, sort_truth_by_eta=True,
                              distance_pen=True)
global_loss_list['fraction_loss_eta_penalty_distance_pen']=fraction_loss_eta_penalty_distance_pen




def fraction_loss_eta_penalty_sort_both( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, eta_penalty=True, sort_truth_by_eta=True, sort_pred_by_eta=True)
global_loss_list['fraction_loss_eta_penalty_sort_both']=fraction_loss_eta_penalty_sort_both


def fraction_loss_eta_penalty_card_pen( truth, pred):
    return weighted_frac_loss(truth, pred, usesqrt=True, eta_penalty=True, sort_truth_by_eta=True, sort_pred_by_eta=True, card_penalty=True)
global_loss_list['fraction_loss_eta_penalty_card_pen']=fraction_loss_eta_penalty_card_pen



def coordinate_loss(truth, pred):
    from Loss_tools import weightedCoordLoss
    
    ldict = create_loss_dict(truth, pred)
    '''
    outputs:
    
    t_sigfrac, p_sigfrac : B x V x Fracs
    r_energy             : B x V 
    t_energy             : B x V x Fracs
    t_sumenergy          : B x Fracs
    t_n_rechits          : B 
    r_eta                : B x V 
    r_phi                : B x V 
    t_issc               : B x V  
    '''
    
    
    last_four_coords= pred[:,:,tf.shape(pred)[2]-(2+4):tf.shape(pred)[2]-2]
    
    coordloss = weightedCoordLoss(ldict['t_sigfrac'], ldict['r_energy'], last_four_coords)
    coordloss = tf.reduce_sum(coordloss, axis=1)/(tf.reduce_sum(ldict['r_energy'],axis=1)+K.epsilon())**2
    coordloss = tf.Print(coordloss,[coordloss, ldict['t_n_rechits']],'coordloss, t_n_rechits ')
    return  coordloss

global_loss_list['coordinate_loss']=coordinate_loss

   


def DR_loss(truth, pred):
    
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = tf.expand_dims(truth[:,:,-1], axis=2)
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    etas      = tf.expand_dims(pred[:,:,-2], axis=2)
    phis      = tf.expand_dims(pred[:,:,-1], axis=2)
    
    drloss = _DR_loss( t_sigfrac, r_energy, p_sigfrac, etas, phis )
    fracloss = fraction_loss_sorted(truth, pred)
    fracloss = tf.reduce_mean(fracloss)
    fracloss = tf.Print(fracloss,[fracloss],'fracloss ')
    
    return  drloss + 0.1*fracloss
    
    
global_loss_list['DR_loss']=DR_loss
    
    
    
def Indiv_DR_loss(truth, pred):  
     
    t_sigfrac = truth[:,:,0:-1]
    r_energy  = tf.expand_dims(truth[:,:,-1], axis=2)
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    etas      = tf.expand_dims(pred[:,:,-2], axis=2)
    phis      = tf.expand_dims(pred[:,:,-1], axis=2) 
    
    sc_eta = weightedCenter(r_energy, t_sigfrac, etas, isPhi=False)
    sc_phi = weightedCenter(r_energy, t_sigfrac, phis, isPhi=True)
    
    #get DR
    DR2m = makeDR2Matrix_SC_hits(sc_eta, sc_phi, etas, phis) # B x V x N_SC
    DR2m = DR2m + 1. #set a minimum, still penalise far away wrong associations more
    DR2m = tf.sqrt(DR2m) # go to actual DR+1
    #B x V x F
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_issc    = tf.where(t_issc>0., t_issc, tf.zeros_like(t_issc))
    t_issc    = tf.where(t_issc>1., tf.zeros_like(t_issc), t_issc)
    t_energy  = r_energy*t_sigfrac
    
    
    sum_p_sigfrac = tf.reduce_sum(p_sigfrac, axis=-1)
    penalty = tf.squeeze(r_energy, axis=-1) * ( t_issc *  (sum_p_sigfrac - 1.)**2 + (1. - t_issc) * sum_p_sigfrac**2)
    penalty = tf.reduce_mean(penalty, axis=1)
    
    #B x V x F
    
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    # B x Fracs
    sc_loss   = tf.reduce_sum(DR2m * t_energy * (t_sigfrac-p_sigfrac)**2, axis=1)/(t_sumenergy+K.epsilon())
    
   
    
    no_dr_sc_loss   = tf.reduce_sum( t_energy * (t_sigfrac-p_sigfrac)**2, axis=1)/(t_sumenergy+K.epsilon())
    # B : mean over sim clusters
    sc_loss   = tf.reduce_mean(sc_loss, axis=-1)
    
    loss = sc_loss + 4. * penalty
    # B 
    loss = tf.Print(loss,[tf.reduce_mean(loss), 
                            tf.reduce_mean(sc_loss), 
                            tf.reduce_mean(penalty),
                            tf.reduce_mean(no_dr_sc_loss)], 'loss, sc_loss, penalty, no_dr_sc_loss ')
        
    return loss
    
    
    
from Loss_implementations import *    
global_loss_list['n_shower_loss']=n_shower_loss
global_loss_list['frac_loss']=frac_loss
    
    





