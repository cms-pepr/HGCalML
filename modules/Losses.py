
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras
import keras.backend as K

def weighted_frac_loss( truth, pred, usesqrt):
    #pred can contain more stuff, just take the first entries
    
    sigfrac = truth[:,:,1:]
    e_weight = truth[:,:,0:1]
    weight=e_weight
    if usesqrt:
        weight = tf.sqrt(tf.abs(weight)+K.epsilon())
    
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    print(p_sigfrac.shape)
    diffs = sigfrac - p_sigfrac
    diffsw = diffs * diffs * weight
    
    diffs  = tf.where(e_weight>0, diffs,  tf.zeros_like(diffs))
    diffsw = tf.where(e_weight>0, diffsw, tf.zeros_like(diffsw))
    
    sumsig = tf.reduce_sum(sigfrac*weight, axis=1) + K.epsilon()
    
    #sumsig = tf.Print(sumsig,[sumsig],'sumsig')
    
    sumdiffs = tf.reduce_sum(diffsw, axis=1)
    
    sumdiffs = tf.Print(sumdiffs,[sumdiffs],'sumdiffs')
    
    #this loss is a bit different from the paper one
    return  tf.reduce_mean(sumdiffs/sumsig)#tf.sign(sumsig)*sumdiffs/sumsig) #the sign masks out the zeros
    
    

    

def fraction_loss( truth, pred):
    return weighted_frac_loss(truth, pred, True)
global_loss_list['fraction_loss']=fraction_loss
    

def fraction_loss_lin( truth, pred):
    return weighted_frac_loss(truth, pred, False)
global_loss_list['fraction_loss_lin']=fraction_loss_lin

