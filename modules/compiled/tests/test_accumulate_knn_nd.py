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
    
    distances = neighbour_space-neighbour_space[:,0:1,:] # V x N x C
    distances = tf.expand_dims(distances, axis=2) # V x N x 1 x C
    weights = tf.math.exp(-1.*distances**2)
    
    neighbour_feat_uw = tf.expand_dims(neighbour_feat_uw, axis=3) # V x N x F x 1
    
    neighbour_feat = neighbour_feat_uw * weights
    mean = tf.reduce_mean(neighbour_feat, axis=1)
    max = tf.reduce_max(neighbour_feat, axis=1)
    
    out = tf.concat([mean,max],axis=-2)
    out = tf.reshape(out, [coords.shape[0], 2*features.shape[1]*coords.shape[1]])
    
    return out


def custom_impl(coords, features, indices):
    out, midx = AccumulateKnnNd(n_moments=0, coords=coords,  features=features, indices=indices)
    #print('midx',midx)
    return tf.reshape(out, [coords.shape[0], 2*features.shape[1]*coords.shape[1]])

    
    
bm = Benchmarker(tf_impl, custom_impl,"GravNet_ND")
bm.debugout=False
print('checking TF versus custom for same results')
for i in range(30):
    print('nvert',5+10*i, 'nfeat',32+i, 'nneigh',2+i)
    bm.difference( nvert = 5+10*i, nfeat = 32+i, nneigh = 2+10*i, ncoords = 8, onlyForward=True)   

#exit()
print('checking TF versus custom for performance')
v100=True
vertmulti = 1000
if v100:
    vertmulti = 8000
nvert  = [int(i*vertmulti/2+1000) for i in range(30)] 
nneigh = [int(25*i)+25 for i in range(0,10)] 
nfeat  = [int(32*i)+32 for i in range(0,10)] 

d_nfeat = 100
d_nneigh = 100
d_nvert = 10000
if v100:
    d_nvert = 25000

bm.run_extended_benchmark(nvert,nneigh,nfeat,d_nvert,d_nneigh,d_nfeat)
bm.run_extended_benchmark(nvert,nneigh,nfeat,d_nvert,d_nneigh,gradient=True)







