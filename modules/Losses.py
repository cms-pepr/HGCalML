
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf

def weighted_frac_loss( truth, pred, usesqrt):
    
    print('pred',pred.shape)
    print('truth',truth.shape)
    
    sigfrac = truth[:,:,1:]
    weight = truth[:,:,0]
    weight = tf.expand_dims(weight,axis=2)
    
    print('weight',weight.shape)
    print('sigfrac',sigfrac.shape)
    
    if usesqrt:
        weight = tf.sqrt(tf.abs(weight))
    
    p_sigfrac = pred
    
    diffs = sigfrac - p_sigfrac
    diffsw = diffs * diffs * weight
    
    sumsig = tf.reduce_sum(sigfrac*weight, axis=1)
    
    sumsig = tf.Print(sumsig,[sumsig],'sumsig')
    
    sumdiffs = tf.reduce_sum(diffsw, axis=1)
    
    #this loss is a bit different from the paper one
    return  tf.reduce_mean(sumdiffs)#tf.sign(sumsig)*sumdiffs/sumsig) #the sign masks out the zeros
    
    

    

def fraction_loss( truth, pred):
    return weighted_frac_loss(truth, pred, True)
global_loss_list['fraction_loss']=fraction_loss
    

def fraction_loss_lin( truth, pred):
    return weighted_frac_loss(truth, pred, False)
global_loss_list['fraction_loss_lin']=fraction_loss_lin

