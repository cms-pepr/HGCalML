#import setGPU #enable GPU

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test
from rknn_op import *
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing_tools import Benchmarker

from accknn_op import AccumulateKnn

def custom_impl(coords, features, indices):
    meanmax, _ = AccumulateKnn(n_moments=0, coords=coords,  features=features, indices=indices)
    return meanmax

def tf_impl(coords,features,indices):
    neighbour_space = tf.gather_nd(coords, indices[...,tf.newaxis])
    neighbour_feat_uw = tf.gather_nd(features, indices[...,tf.newaxis])
    no_weight_mean = tf.reduce_mean(neighbour_feat_uw,axis=1)
    
    distances = tf.reduce_sum((neighbour_space-neighbour_space[:,0:1,:])**2, axis=-1, keepdims=True)
    
    weights = tf.math.exp(-10.*distances)
    neighbour_feat = neighbour_feat_uw * weights
    mean = tf.reduce_mean(neighbour_feat, axis=1)
    
    max = tf.reduce_max(neighbour_feat, axis=1)
    return tf.concat([mean,max],axis=-1)



    
    
bm = Benchmarker(tf_impl, custom_impl,"GravNet_default")
bm.difference(nvert = 15, nfeat = 10, nneigh = 4, ncoords = 4)    

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







