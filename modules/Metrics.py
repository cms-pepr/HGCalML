
# Define custom metrics here and add them to the global_metrics_list dict (important!)
global_metrics_list = {}

import tensorflow as tf
import keras

def get_n_above_threshold(betas, threshold):
    #returns B x 1
    betas = tf.where(betas<threshold, tf.zeros_like(betas),betas)
    flattened = tf.reshape(betas,(tf.shape(betas)[0],-1))
    return tf.expand_dims(tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    

def pixel_over_threshold_accuracy(truth,pred):
    from betaLosses import create_pixel_loss_dict, mean_nvert_with_nactive
    d = create_pixel_loss_dict(truth,pred)
    
    maxprob = tf.where(
        tf.equal(tf.reduce_max(d['p_ID'], axis=-1, keepdims=True), d['p_ID']), 
        tf.zeros_like(d['p_ID']) + 1., 
        tf.zeros_like(d['p_ID'])
    )

    acc = tf.reduce_sum(maxprob*d['t_mask']*d['t_ID'],axis=-1)#B x V
    acc = tf.expand_dims(acc,axis=2)#Bx V x 1

    #so only above threshold
    mult = tf.where(d['p_beta']> 0.5,tf.zeros_like(d['p_beta'])+1.,tf.zeros_like(d['p_beta']))
    acc = acc * mult
    nabove = get_n_above_threshold(d['p_beta'], 0.5)
    
    acc = mean_nvert_with_nactive(acc, nabove) #B x 1
    acc = tf.reduce_mean(acc) #[]
    acc = tf.Print(acc,[acc, tf.reduce_mean(nabove)],'acc , nabove ')
    return acc 
    
    
    
global_metrics_list['pixel_over_threshold_accuracy'] = pixel_over_threshold_accuracy