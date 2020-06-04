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

def custom_impl(distances, features, indices):
    #print(distances)
    meanmax, mmidx = AccumulateKnn(n_moments=0, distances=distances,  features=features, indices=indices)
    #print('mmidx',mmidx)
    return meanmax#[:,features.shape[1]:] #just mean for now

def tf_impl(distances,features,indices):

    neighbour_feat_uw = tf.gather_nd(features, indices[...,tf.newaxis])
    
    #print('minmax indxs',tf.reduce_min(indices), tf.reduce_max(indices))
    weights = tf.math.exp(-1. * distances)[...,tf.newaxis]
    #print('weights',weights)
    #print('maxweight', tf.reduce_max(weights), 'min distance', tf.reduce_min(distances))
    
    neighbour_feat = neighbour_feat_uw * weights
    #print('neighbour_feat',neighbour_feat)
    mean = tf.reduce_mean(neighbour_feat, axis=1)
    
    max = tf.reduce_max(neighbour_feat, axis=1)
    return tf.concat([mean,max],axis=-1)


usecpu=False
    
    
bm = Benchmarker(tf_impl, custom_impl,"GravNet_default", use_distances_direct=True, 
                 tfoncpu=usecpu, customoncpu=usecpu)
bm.debugout=True
bm.difference(nvert = 5, nfeat = 3, nneigh = 2, ncoords = 4)    
bm.debugout=False

#exit()

nvert  = [int(i*100/2+150) for i in range(5)] 
nneigh = [int(25*i)+25 for i in range(0,4)] 
nfeat  = [int(32*i)+32 for i in range(0,4)] 



#bm.run_extended_difference(nvert,nneigh,nfeat)

#exit()
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







