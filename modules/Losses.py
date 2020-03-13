
'''
Define custom losses here and add them to the global_loss_list dict (important!)
Keep this file to functions with a few lines adding loss components
Do the actual implementation of the loss components in Loss_implementations.py

'''
global_loss_list = {}

import tensorflow as tf

##### prototype for merging loss function
from betaLosses import obj_cond_loss_rowsplits , obj_cond_loss_truth , pre_training_loss, null_loss
global_loss_list['obj_cond_loss_truth'] = obj_cond_loss_truth
global_loss_list['obj_cond_loss_rowsplits'] = obj_cond_loss_rowsplits
global_loss_list['pre_training_loss'] = pre_training_loss
global_loss_list['null_loss'] = null_loss




######

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
