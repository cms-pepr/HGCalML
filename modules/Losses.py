
'''
Define custom losses here and add them to the global_loss_list dict (important!)
Keep this file to functions with a few lines adding loss components
Do the actual implementation of the loss components in Loss_implementations.py

'''
global_loss_list = {}

import tensorflow as tf

def fraction_loss_with_penalties(truth, pred):
    from Loss_implementations import frac_loss, good_frac_sum_loss, good_frac_range_loss
    fracloss = frac_loss(truth, pred) 
    fracsumloss = good_frac_sum_loss(truth,pred) 
    fracrangeloss = good_frac_range_loss(truth, pred)
    
    loss = fracloss + fracsumloss + 100.* fracrangeloss
    
    loss = tf.Print(loss, [tf.reduce_mean(loss), 
                           tf.reduce_mean(fracloss), 
                           tf.reduce_mean(fracsumloss), 
                           tf.reduce_mean(fracrangeloss)],
                    'loss, fracloss, fracsumloss, fracrangeloss ')
    return loss
global_loss_list['fraction_loss_with_penalties']=fraction_loss_with_penalties

def fraction_loss_with_penalties_sort_pred(truth, pred):
    from Loss_implementations import frac_loss_sort_pred, good_frac_sum_loss, good_frac_range_loss
    fraclosssortpred = frac_loss_sort_pred(truth, pred) 
    fracsumloss = good_frac_sum_loss(truth,pred) 
    fracrangeloss = good_frac_range_loss(truth, pred)
    
    loss = fraclosssortpred + fracsumloss + 100.* fracrangeloss
    
    loss = tf.Print(loss, [tf.reduce_mean(loss), 
                           tf.reduce_mean(fraclosssortpred), 
                           tf.reduce_mean(fracsumloss), 
                           tf.reduce_mean(fracrangeloss)],
                    'loss, fracloss (sort pred), fracsumloss, fracrangeloss ')
    return loss
global_loss_list['fraction_loss_with_penalties_sort_pred']=fraction_loss_with_penalties_sort_pred






####### for the 'pre'clustering tests

def clusterloss(truth, pred):
    from keras import losses
    import keras.backend as K
    
    #truth: posx, posy, Efull, 3x ID
    #pred:  3x linear, 3x softmax 
    
    true_pos = truth[:,:,0:2]
    true_E   = truth[:,:,2:3]
    
    #true_E = tf.Print(true_E,[true_E],'true_E ')
    
    true_ID  = truth[:,:,3:6]
    
    n_vertex = tf.cast(tf.count_nonzero(true_E, axis=1), dtype='float32')
    
    pred_pos = pred[:,:,0:2]
    pred_E   = pred[:,:,2:3]
    pred_ID  = pred[:,:,3:6]
    
    pred_confidence = pred[:,:,6]
    
    pos_loss = tf.reduce_mean((true_pos-pred_pos)**2,axis=-1) * 24 * 24 / 4.
    #norm to about 1 being one minimum distance, only 
    
    E_loss  = tf.reduce_mean((true_E - pred_E)**2,axis=-1) / (50.*50.)
    
    ID_loss = 0 #losses.categorical_crossentropy(true_ID, pred_ID)
    
    loss_per_vertex = 1. * pos_loss + 1. * E_loss + 0.01 * ID_loss
    
    # add the confidence somewhere.
    loss_per_vertex /= ((1. - pred_confidence) + K.epsilon())
    loss_per_vertex = tf.reduce_sum(loss_per_vertex, axis=-1) / (n_vertex+K.epsilon())
    conf_sum = tf.reduce_sum((1. - pred_confidence),axis=-1) / (n_vertex+K.epsilon())
    
    loss = loss_per_vertex + 2. * conf_sum
    
    
    return tf.reduce_mean(loss)
    
    
global_loss_list['clusterloss']=clusterloss


def dummyloss(truth, pred):
    return tf.reduce_mean((tf.reduce_sum(pred,axis=-1)-tf.reduce_sum(truth,axis=-1))**2)
    

def clusterloss_clustercoords(truth, pred):
    from betaLosses import beta_clusterloss_clustercoords
    return beta_clusterloss_clustercoords(truth, pred)

global_loss_list['clusterloss_clustercoords']=clusterloss_clustercoords


def pixel_clustercoords(truth, pred):
    from betaLosses import beta_pixel_clustercoords
    return beta_pixel_clustercoords(truth, pred)
    
global_loss_list['pixel_clustercoords']=pixel_clustercoords


