
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras
import keras.backend as K

def weighted_frac_loss( truth, pred, usesqrt, weightfactor=1.):
    #pred can contain more stuff, just take the first entries
    # the LAST entry of truth is energy
    sigfrac = truth[:,:,0:-1]
    #sigfrac = tf.Print(sigfrac,[sigfrac],'sigfrac',summarize=300)
    e_weight = truth[:,:,-1]
    e_weight = tf.expand_dims(e_weight, axis=2)
    #e_weight = tf.Print(e_weight,[e_weight],'e_weight',summarize=300)
    weight=e_weight
    if usesqrt:
        weight = tf.sqrt(tf.abs(weight)+K.epsilon())
    if weightfactor<=0:
        weight = tf.zeros_like(weight)+1.
    
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    
    #sigfrac = tf.Print(sigfrac,[sigfrac[0,0,:]],'sigfrac')
    #p_sigfrac = tf.Print(p_sigfrac,[p_sigfrac[0,0,:]],'p_sigfrac')
    
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
    return  tf.reduce_mean(sumdiffs/sumsig)#tf.sign(sumsig)*sumdiffs/sumsig) #the sign masks out the zeros
    
    

def fraction_loss_noweight( truth, pred):
    return weighted_frac_loss(truth, pred, True,-1.)
global_loss_list['fraction_loss_noweight']=fraction_loss_noweight
    

def fraction_loss( truth, pred):
    return weighted_frac_loss(truth, pred, True)
global_loss_list['fraction_loss']=fraction_loss
    

def fraction_loss_lin( truth, pred):
    return weighted_frac_loss(truth, pred, False)
global_loss_list['fraction_loss_lin']=fraction_loss_lin

