import setGPU #enable GPU

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test
from rknn_op import *
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from accknn_op import AccumulateKnn


def tf_impl(coords,feats,indices):
    neighbour_space = tf.gather_nd(coords, indices[...,tf.newaxis])
    neighbour_feat_uw = tf.gather_nd(feats, indices[...,tf.newaxis])
    no_weight_mean = tf.reduce_mean(neighbour_feat_uw,axis=1)
    
    distances = tf.reduce_sum((neighbour_space-neighbour_space[:,0:1,:])**2, axis=-1, keepdims=True)
    
    weights = tf.math.exp(-1.*distances)
    neighbour_feat = neighbour_feat_uw * weights
    mean = tf.reduce_mean(neighbour_feat, axis=1)
    
    max = tf.reduce_max(neighbour_feat, axis=1)
    return tf.concat([mean,max],axis=-1)

def benchmark(nvert = 30000, nfeat = 64, nneigh = 128, ncoords = 4, dogradient=False):
    
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    feats  = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
    row_splits = tf.constant( [0, nvert] ,dtype='int32')
    
    indices, _ = rknn_op.RaggedKnn(num_neighbors=nneigh, row_splits=row_splits, data=coords, add_splits=False)
    
    if not dogradient:
        #each gets one dry run to compile
        meanmax, _ = AccumulateKnn(n_moments=0, coords=coords,  features=feats, indices=indices)
        t0 = time.time()
        for i in range(0,50):
            meanmax, _ = AccumulateKnn(n_moments=0, coords=coords,  features=feats, indices=indices)
        
        op_time= (time.time() - t0)/50.
        print('op_time',op_time)
        
        tf_time=0
        try:
            meanmax = tf_impl(coords=coords,  feats=feats, indices=indices)
            t0 = time.time()
            for i in range(0,50):
                meanmax = tf_impl(coords=coords,  feats=feats, indices=indices)
            tf_time= (time.time() - t0)/50.
        except:
            pass
            
        print('tf_time',tf_time)
            
        return op_time, tf_time
    
    else:
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
            t_newop.watch(coords)
            t_newop.watch(feats)
            meanmax, _ = AccumulateKnn(n_moments=0, coords=coords,  features=feats, indices=indices)
            
        #once to get it compiled in case needed
        feat_grad = t_newop.gradient(meanmax, feats)
        coord_grad = t_newop.gradient(meanmax, coords)
        
        t0 = time.time()
        for i in range(5) :
            feat_grad = t_newop.gradient(meanmax, feats)
            coord_grad = t_newop.gradient(meanmax, coords)
        op_time= (time.time() - t0)/5.
        
        tf_time=0
        try:
            with tf.GradientTape(persistent=True) as t_tfop:
                t_tfop.watch(coords)
                t_tfop.watch(feats)
                meanmax = tf_impl(coords=coords,  feats=feats, indices=indices)
        
            feat_grad = t_tfop.gradient(meanmax, feats)
            coord_grad = t_tfop.gradient(meanmax, coords)
            t0 = time.time()
            for i in range(5) :
                feat_grad = t_tfop.gradient(meanmax, feats)
                coord_grad = t_tfop.gradient(meanmax, coords)
            tf_time= (time.time() - t0)/5.
        except:
            pass
        return op_time, tf_time
        
    
    
    
    #tf.test.Benchmark()

#class AccumulateKNNTest(test.TestCase):
    
def test_uniform_test():#self):
    
    debugout=False
    
    nvert = 200
    nfeat = 32
    nneigh = 24
    ncoords = 4
    
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    feats  = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
    row_splits = tf.constant( [0, nvert] ,dtype='int32')
    
    indices, _ = rknn_op.RaggedKnn(num_neighbors=nneigh, row_splits=row_splits, data=coords, add_splits=False)
    
    ### direct implementation
    
    op_time = 0
    t0 = time.time()
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        t_newop.watch(feats)
        meanmax, _ = AccumulateKnn(n_moments=0, coords=coords,  features=feats, indices=indices)
        t1 = time.time()
        op_time= t1 - t0
        
    print('op time',op_time)
        
    feat_grad = t_newop.gradient(meanmax, feats)
    coord_grad = t_newop.gradient(meanmax, coords)
    
    if debugout:
        print('coords',coords,'\n')
        print('feats',feats,'\n')
        print('meanmax',meanmax,'\n')
        print('indices',indices)
    
        ### tf op implementation
        print('TFTFTF')
        
    tf_feat_grad = None
    tf_coord_grad = None
    t0 = time.time()
    with tf.GradientTape(persistent=True) as t_tfop:
        t_tfop.watch(coords)
        t_tfop.watch(feats)
        neighbour_space = tf.gather_nd(coords, indices[...,tf.newaxis])
        neighbour_feat_uw = tf.gather_nd(feats, indices[...,tf.newaxis])
        no_weight_mean = tf.reduce_mean(neighbour_feat_uw,axis=1)
        
        distances = tf.reduce_sum((neighbour_space-neighbour_space[:,0:1,:])**2, axis=-1, keepdims=True)
        
        if debugout:
            print('distances',distances)
        weights = tf.math.exp(-1.*distances)
        if debugout:
            print('weights',weights)
        
        neighbour_feat = neighbour_feat_uw * weights
        
        if debugout:
            print('weighted neighbour_feat',neighbour_feat)
        
        #print('weights',weights)
        
        mean = tf.reduce_mean(neighbour_feat, axis=1)
        
        max = tf.reduce_max(neighbour_feat, axis=1)
        tf_meanmax = tf.concat([mean,max],axis=-1)
    
    tf_time= time.time() - t0
        
    
    # print('indices',indices)
    #print('mean coord grad',t_tfop.gradient(mean, coords))   
    tf_feat_grad = t_tfop.gradient(tf_meanmax, feats)
    tf_coord_grad = t_tfop.gradient(tf_meanmax, coords)
    
    if debugout:
        print('TF mean',mean)
    if debugout:
        print('TF max',max)
    
    difference = meanmax - tf_meanmax
    max_difference = tf.reduce_max(tf.abs(difference)).numpy()
    
    print('max difference',max_difference)
    print('op time',op_time)
    print('tf time',tf_time)
    assert max_difference < 1e-3
    
    ## gradients
    #print('tf_feat_grad',tf_feat_grad)
    #print('tf_coord_grad',tf_coord_grad)
    
    #print('feat_grad',feat_grad)
    #print('coord_grad',coord_grad)
    
    feat_grad_diff = feat_grad - tf_feat_grad
    coord_grad_diff = coord_grad - tf_coord_grad
    
    #print('feat_grad_diff',feat_grad_diff)
    #print('coord_grad_diff',coord_grad_diff)
    
    #print('relative feat_grad_diff',feat_grad_diff/tf_feat_grad)
    #print('relative coord_grad_diff',coord_grad_diff/tf_coord_grad)
    
    maxfeatgraddiff = tf.reduce_max(tf.abs(feat_grad_diff))
    maxcoordgraddiff = tf.reduce_max(tf.abs(coord_grad_diff))
    
    print('max feature grad diff', maxfeatgraddiff)
    print('max coordinate grad diff', maxcoordgraddiff)
    
    assert maxfeatgraddiff < 1e-3
    assert maxcoordgraddiff < 1e-3
    
    #t_newop.gradient(meanmax, feats)
    #t_newop.gradient(meanmax, coords)
    
def run_extended_benchmark(gradient=False):
    #nvert  = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,10000]
    nvert  = [int(i*1000/2+1000) for i in range(30)]
    nneigh = [int(25*i)+25 for i in range(0,10)]
    nfeat  = [int(32*i)+32 for i in range(0,10)]
    
    d_nfeat = 100
    d_nneigh = 100
    d_nvert = 10000
    
    tf_times = []
    op_times = []
    tfx = []
    
    for nv in nvert:
        print('nvert benchmark, nvert:',nv)
        opt,tft = benchmark(nv,d_nfeat,d_nneigh,4, dogradient=gradient)
        if tft:
            tf_times.append(tft)
            tfx.append(nv)
        op_times.append(opt)
    
    plt.plot(nvert,op_times,color='green',label="custom",marker='o')
    plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
    plt.xlabel("# vertices")
    plt.ylabel("s")
    #plt.yscale('log')
    plt.legend()
    if gradient:
        plt.savefig("benchmark_grad_nvert.pdf")
    else:
        plt.savefig("benchmark_nvert.pdf")
    plt.close()
    
    tf_times=[]
    op_times=[]
    tfx=[]
    for nn in nneigh:
        print('nneigh benchmark, nn:',nn)
        opt,tft = benchmark(d_nvert,d_nfeat,nn,4,dogradient=gradient)
        if tft:
            tf_times.append(tft)
            tfx.append(nn)
        op_times.append(opt)
        
    plt.plot(nneigh,op_times,color='green',label="custom",marker='o')
    plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
    plt.xlabel("# neighbours")
    plt.ylabel("s")
    plt.legend()
    if gradient:
        plt.savefig("benchmark_grad_nneigh.pdf")
    else:
        plt.savefig("benchmark_nneigh.pdf")
    plt.close()
    
    tf_times=[]
    op_times=[]
    tfx=[]
    for nf in nfeat:
        print('nfeat benchmark, nfeat:',nf)
        opt,tft = benchmark(d_nvert,nf,d_nneigh,4,dogradient=gradient)
        if tft:
            tf_times.append(tft)
            tfx.append(nf)
        op_times.append(opt)
        
    plt.plot(nfeat,op_times,color='green',label="custom",marker='o')
    plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
    plt.xlabel("# features")
    plt.ylabel("s")
    plt.legend()
    if gradient:
        plt.savefig("benchmark_grad_nfeat.pdf")
    else:
        plt.savefig("benchmark_nfeat.pdf")
    plt.close()
    
    

test_uniform_test()    
run_extended_benchmark()
run_extended_benchmark(gradient=True)
#exit()
#benchmark()
#exit()