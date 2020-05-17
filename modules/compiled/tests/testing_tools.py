
import tensorflow as tf
from rknn_op import *
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Benchmarker(object):
    def __init__(self, tf_implementation, custom_implementation, name):
        self.tfimp=tf_implementation
        self.customimpl=custom_implementation
        self.name = name
        self.debugout=False

    def benchmark(self, nvert = 30000, nfeat = 64, nneigh = 128, ncoords = 4, dogradient=False,do_tf=True):
        
        coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
        feats  = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
        row_splits = tf.constant( [0, nvert] ,dtype='int32')
        
        indices, _ = rknn_op.RaggedKnn(num_neighbors=nneigh, row_splits=row_splits, data=coords, add_splits=False)
        
        if not dogradient:
            #each gets one dry run to compile
            meanmax = self.customimpl(coords=coords,  features=feats, indices=indices)
            t0 = time.time()
            for i in range(0,50):
                meanmax = self.customimpl( coords=coords,  features=feats, indices=indices)
            
            op_time= (time.time() - t0)/50.
            print('op_time',op_time)
            
            tf_time=0
            if do_tf:
                try:
                    meanmax = self.tfimp(coords=coords,  features=feats, indices=indices)
                    t0 = time.time()
                    for i in range(0,50):
                        meanmax = self.tfimp(coords=coords,  features=feats, indices=indices)
                    tf_time= (time.time() - t0)/50.
                except:
                    pass
                    
                print('tf_time',tf_time)
                    
            return op_time, tf_time
        
        else:
            with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
                t_newop.watch(coords)
                t_newop.watch(feats)
                meanmax  = self.customimpl( coords=coords,  features=feats, indices=indices)
                
            #once to get it compiled in case needed
            feat_grad = t_newop.gradient(meanmax, feats)
            coord_grad = t_newop.gradient(meanmax, coords)
            
            t0 = time.time()
            for i in range(5) :
                feat_grad = t_newop.gradient(meanmax, feats)
                coord_grad = t_newop.gradient(meanmax, coords)
            op_time= (time.time() - t0)/5.
            
            tf_time=0
            
            if do_tf:
                try:
                    with tf.GradientTape(persistent=True) as t_tfop:
                        t_tfop.watch(coords)
                        t_tfop.watch(feats)
                        meanmax = self.tfimp(coords=coords,  features=feats, indices=indices)
                
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
        
    def difference(self, nvert = 300, nfeat = 64, nneigh = 32, ncoords = 4, onlyForward=False):
        
        coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
        feats  = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
        row_splits = tf.constant( [0, nvert] ,dtype='int32')
        
        indices, _ = rknn_op.RaggedKnn(num_neighbors=nneigh, row_splits=row_splits, data=coords, add_splits=False)
        
        op_time = 0
        t0 = time.time()
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
            t_newop.watch(coords)
            t_newop.watch(feats)
            meanmax = self.customimpl( coords=coords,  features=feats, indices=indices)
            t1 = time.time()
            op_time= t1 - t0
            
        print('op time',op_time)
            
        feat_grad = t_newop.gradient(meanmax, feats)
        coord_grad = t_newop.gradient(meanmax, coords)
        
        if self.debugout:
            print('coords',coords,'\n')
            print('feats',feats,'\n')
            print('custom output',meanmax,'\n')
            print('indices',indices)
        
            ### tf op implementation
            print('TFTFTF')
            
        tf_feat_grad = None
        tf_coord_grad = None
        t0 = time.time()
        with tf.GradientTape(persistent=True) as t_tfop:
            t_tfop.watch(coords)
            t_tfop.watch(feats)
            tf_meanmax = self.tfimp(coords=coords,  features=feats, indices=indices)
        
        tf_time= time.time() - t0
        
        if self.debugout:
            print('TF output',tf_meanmax,'\n')
            
        
        tf_feat_grad = t_tfop.gradient(tf_meanmax, feats)
        tf_coord_grad = t_tfop.gradient(tf_meanmax, coords)
        
        
        difference = meanmax - tf_meanmax
        max_rel_difference = tf.reduce_max(tf.abs(difference/(tf_meanmax+1e-3))).numpy()
        max_difference = tf.reduce_max(tf.abs(difference)).numpy()
        
        print('max rel difference',max_rel_difference)
        print('max difference',max_difference)
        print('op time',op_time)
        print('tf time',tf_time)
        assert max_difference < 1e-2
        
        if onlyForward:
            return
        
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
        
        maxrelfeatgraddiff = tf.reduce_max(tf.abs(feat_grad_diff/(tf_feat_grad+1e-3)))
        maxrelcoordgraddiff = tf.reduce_max(tf.abs(coord_grad_diff/(tf_coord_grad+1e-3)))
        
        maxfeatgraddiff = tf.reduce_max(tf.abs(feat_grad_diff/(tf_feat_grad+1e-3)))
        maxcoordgraddiff = tf.reduce_max(tf.abs(coord_grad_diff/(tf_coord_grad+1e-3)))
        
        print('\nmax relative feature grad diff', maxrelfeatgraddiff)
        print('max relative coordinate grad diff', maxrelcoordgraddiff)
        
        if maxrelfeatgraddiff > 1e-2:
            print('Feature gradient off:')
            if self.debugout:
                print('custom ',feat_grad)
                print('TF ',tf_feat_grad)
            print('max rel diff',maxrelfeatgraddiff)
            print('max diff',maxfeatgraddiff)
            
        if maxrelcoordgraddiff > 1e-2:
            print('Coordinate gradient off:')
            if self.debugout:
                print('custom ',coord_grad)
                print('TF ',tf_coord_grad)
            print('max rel diff',maxrelcoordgraddiff)
            print('max diff',maxcoordgraddiff)
        
        
        assert maxfeatgraddiff < 1e-2
        assert maxcoordgraddiff < 1e-2
    
        
    def run_extended_benchmark(self,
        nvert,
        nneigh,
        nfeat,
        d_nvert = 10000,
        d_nneigh = 100,
        d_nfeat = 100,
        
        gradient=False,
        tf_thresholds = {'nvert': 55000,
                         'nneigh': 210,
                         'nfeat': 200}):
        
        tf_times = []
        op_times = []
        tfx = []
        
        for nv in nvert:
            print('nvert self.benchmark, nvert:',nv, "do tf",tf_thresholds['nvert']>nv)
            opt,tft = self.benchmark(nv,d_nfeat,d_nneigh,4, dogradient=gradient,do_tf=tf_thresholds['nvert']>nv)
            if tft:
                tf_times.append(tft)
                tfx.append(nv)
            op_times.append(opt)
        
        plt.plot(nvert,op_times,color='green',label="custom",marker='o')
        plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
        plt.xlabel("# vertices")
        plt.ylabel("time")
        #plt.yscale('log')
        plt.legend()
        if gradient:
            plt.savefig(self.name+"benchmark_grad_nvert.pdf")
        else:
            plt.savefig(self.name+"benchmark_nvert.pdf")
        plt.close()
        
        tf_times=[]
        op_times=[]
        tfx=[]
        for nn in nneigh:
            print('nneigh self.benchmark, nn:',nn)
            opt,tft = self.benchmark(d_nvert,d_nfeat,nn,4,
                                     dogradient=gradient,do_tf=tf_thresholds['nneigh']>nn)
            if tft:
                tf_times.append(tft)
                tfx.append(nn)
            op_times.append(opt)
            
        plt.plot(nneigh,op_times,color='green',label="custom",marker='o')
        plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
        plt.xlabel("# neighbours")
        plt.ylabel("time")
        plt.legend()
        if gradient:
            plt.savefig(self.name+"benchmark_grad_nneigh.pdf")
        else:
            plt.savefig(self.name+"benchmark_nneigh.pdf")
        plt.close()
        
        tf_times=[]
        op_times=[]
        tfx=[]
        for nf in nfeat:
            print('nfeat self.benchmark, nfeat:',nf)
            opt,tft = self.benchmark(d_nvert,nf,d_nneigh,4,
                                     dogradient=gradient,do_tf=tf_thresholds['nfeat']>nf)
            if tft:
                tf_times.append(tft)
                tfx.append(nf)
            op_times.append(opt)
            
        plt.plot(nfeat,op_times,color='green',label="custom",marker='o')
        plt.plot(tfx,tf_times,color='orange',label="TF",marker='o')
        plt.xlabel("# features")
        plt.ylabel("time")
        plt.legend()
        if gradient:
            plt.savefig(self.name+"benchmark_grad_nfeat.pdf")
        else:
            plt.savefig(self.name+"benchmark_nfeat.pdf")
        plt.close()
            
        
        