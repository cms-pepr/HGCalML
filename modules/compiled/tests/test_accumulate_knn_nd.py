#import setGPU #enable GPU

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing_tools import Benchmarker

from accknn_op import AccumulateKnnNd


def tf_impl(coords,features,indices):
    
    neighbour_space = tf.gather_nd(coords, indices[...,tf.newaxis])
    neighbour_feat_uw = tf.gather_nd(features, indices[...,tf.newaxis])
    no_weight_mean = tf.reduce_mean(neighbour_feat_uw,axis=1)
    
    distances = neighbour_space- tf.expand_dims(coords, axis=1) # V x N x C
    distances_exp = tf.expand_dims(distances, axis=2) # V x N x 1 x C
    weights = tf.math.exp(-10.*distances_exp**2)
    
    neighbour_feat_uw = tf.expand_dims(neighbour_feat_uw, axis=3) # V x N x F x 1
    
    neighbour_feat = neighbour_feat_uw * weights
    mean = tf.reduce_mean(neighbour_feat, axis=1)
    max = tf.reduce_max(neighbour_feat, axis=1)
    
    #moments etc
    featsum = tf.squeeze(tf.reduce_sum(neighbour_feat_uw, axis=1),axis=2)  # V x F 
    
    out = tf.concat([mean,max],axis=-2)
    
    #add moments
    m_mean =  tf.reduce_sum(neighbour_feat_uw * distances_exp, axis=1) / tf.expand_dims(featsum, axis=2) # V x F x C
    m_mean = tf.where(tf.expand_dims(featsum, axis=2) ==0, tf.zeros_like(m_mean), m_mean)
    out = tf.concat([out,m_mean],axis=-2)
    #end moments
    var = tf.reduce_sum(neighbour_feat_uw * (distances_exp - tf.expand_dims(m_mean,axis=1))**2, axis=1) / tf.expand_dims(featsum, axis=2) # V x F x C
    var = tf.where(tf.expand_dims(featsum, axis=2) ==0, tf.zeros_like(var), var)
    out = tf.concat([out,var],axis=-2)
    
    out = tf.reshape(out, [out.shape[0], out.shape[1]*out.shape[2] ])
    out = tf.concat([out,featsum],axis=-1)
    
    return out


def custom_impl(coords, features, indices):
    out, midx, featsum = AccumulateKnnNd(n_moments=2, coords=coords,  features=features, indices=indices)
    #print('midx',midx)
    out = tf.reshape(out, [out.shape[0], out.shape[1]*out.shape[2] ])
    out = tf.concat([out,featsum],axis=-1)
    return out


    
bm = Benchmarker(tf_impl, custom_impl,"GravNet_ND")
bm.debugout=False
print('checking TF versus custom for same results')
for i in range(0):
    print('nvert',5+10*i, 'nfeat',32+2*i, 'nneigh',2+10*i)
    bm.difference( nvert = 5+10*i, nfeat = 32+2*i, nneigh = 2+10*i, ncoords = 4, onlyForward=False)   
    
    

bm.debugout=True
#bm.difference( nvert = 5, nfeat = 2, nneigh = 2, ncoords = 2, onlyForward=False)  
#bm.numerical_gradient(nvert = 5, nfeat = 2, nneigh = 2, ncoords = 2)
bm.debugout=False
#exit()

v100=True
nvert  = [int(i*100/2+150) for i in range(10)] 
nneigh = [int(25*i)+25 for i in range(0,4)] 
nfeat  = [int(32*i)+32 for i in range(0,4)] 


bm.run_extended_difference(nvert,nneigh,nfeat)

#exit()
print('checking TF versus custom for performance')
d_nfeat = 100
d_nneigh = 100
d_nvert = 10000
if v100:
    d_nvert = 25000
vertmulti = 1000
    
nvert  = [int(i*4*vertmulti+1000) for i in range(20)] 
nneigh = [int(25*i)+25 for i in range(0,4)] 
nfeat  = [int(32*i)+32 for i in range(0,4)] 

bm.run_extended_benchmark(nvert,nneigh,nfeat,d_nvert,d_nneigh,d_nfeat)
bm.run_extended_benchmark(nvert,nneigh,nfeat,d_nvert,d_nneigh,gradient=True)







