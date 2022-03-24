
import tensorflow as tf
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from select_knn_op import SelectKnn
import os

def makeIndices(nvert,nneigh):
    all = []
    for i in range(nvert):
        a = np.array([],dtype='int32')
        while len(a) < nneigh-1:
            a = np.random.choice(nvert, nneigh-1, replace=False)
            a = a[a != i]
        a = np.concatenate([np.array([i],dtype='int32'),a],axis=-1)
        a = np.expand_dims(a, axis=0)
        all.append(a)
        
    
    return np.concatenate(all,axis=0)


class Benchmarker(object):
    def __init__(self, tf_implementation, custom_implementation, name, use_distances_direct,tfoncpu,customoncpu,mean_and_max):
        self.tfimp=tf_implementation
        self.customimpl=custom_implementation
        self.name = name
        self.debugout=False
        self.use_distances_direct=use_distances_direct
        self.tfoncpu=tfoncpu
        self.customoncpu=customoncpu
        self.mean_and_max=mean_and_max

    def benchmark(self, nvert = 30000, nfeat = 64, nneigh = 128, ncoords = 4, dogradient=False,do_tf=True):
        
        coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
        feats  = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
        row_splits = tf.constant( [0, nvert] ,dtype='int32')
        
        indices, distances = SelectKnn(K=nneigh, coords=coords, row_splits=row_splits)
        if self.use_distances_direct:
            coords = distances
        tf_failed = False
        if not dogradient:
            #each gets one dry run to compile
            meanmax = self.customimpl(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
            t0 = time.time()
            for i in range(0,50):
                meanmax = self.customimpl(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
            
            op_time= (time.time() - t0)/50.
            print('op_time',op_time)
            
            tf_time=0
            if do_tf:
                
                try:
                    meanmax = self.tfimp(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
                    t0 = time.time()
                    for i in range(0,50):
                        meanmax = self.tfimp(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
                    tf_time= (time.time() - t0)/50.
                except:
                    tf_failed=True
                    
                print('tf_time',tf_time)
                    
            return op_time, tf_time
        
        else:
            with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
                t_newop.watch(coords)
                t_newop.watch(feats)
                meanmax  = self.customimpl(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
                
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
                        meanmax = self.tfimp(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
                
                    feat_grad = t_tfop.gradient(meanmax, feats)
                    coord_grad = t_tfop.gradient(meanmax, coords)
                    t0 = time.time()
                    for i in range(5) :
                        feat_grad = t_tfop.gradient(meanmax, feats)
                        coord_grad = t_tfop.gradient(meanmax, coords)
                    tf_time= (time.time() - t0)/5.
                except:
                    tf_failed=True
            return op_time, tf_time
        
    
    def difference(self, nvert = 300, nfeat = 64, nneigh = 32, ncoords = 4, onlyForward=False, assert_error=True):
        
        coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
        feats  = np.random.rand(nvert,nfeat) 
        #to make the max unambiguous 
        frange = np.arange(nvert)
        np.random.shuffle(frange)
        toadd = np.expand_dims(frange, axis=1)
        feats = tf.constant(feats+toadd,  dtype='float32') 
        
        row_splits = tf.constant( [0, nvert] ,dtype='int32')
        #print('building indices')
        with tf.device("/cpu:0"):
            indices, distances = SelectKnn(K=nneigh, coords=coords, row_splits=row_splits)
            #indices = indices[:,1:]
            #distances = distances[:,1:]
        #print('process custom op')
        if self.use_distances_direct:
            coords = distances
        op_time = 0
        
        tfdevstring = "/gpu:0"
        if self.customoncpu:
            tfdevstring = "/cpu:0"
        tfdev = tf.device(tfdevstring)
        t0 = time.time()
        with tfdev:
            t0 = time.time()
            with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
                t_newop.watch(coords)
                t_newop.watch(feats)
                meanmax = self.customimpl( coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
                t1 = time.time()
                op_time= t1 - t0
            
        #print('op time',op_time)
        with tfdev:
            coord_grad = t_newop.gradient(meanmax, coords)
            feat_grad = t_newop.gradient(meanmax, feats)
        
        if self.debugout:
            print('coords',coords,'\n')
            print('feats',feats,'\n')
            print('custom output',meanmax,'\n')
            print('indices',indices)
        
            ### tf op implementation
            print('TFTFTF')
            
        tf_feat_grad = None
        tf_coord_grad = None
        #print('process TF op')
        tfdevstring = "/gpu:0"
        if self.tfoncpu:
            tfdevstring = "/cpu:0"
        tfdev = tf.device(tfdevstring)
        t0 = time.time()
        with tfdev:
            with tf.GradientTape(persistent=True) as t_tfop:
                t_tfop.watch(coords)
                t_tfop.watch(feats)
                tf_meanmax = self.tfimp(coords,  features=feats, indices=indices, mean_and_max=self.mean_and_max)
        
        tf_time= time.time() - t0
        
        
            
        if self.debugout:
            print('TF output',tf_meanmax,'\n')
            
        with tfdev:
            tf_feat_grad = t_tfop.gradient(tf_meanmax, feats)
            tf_coord_grad = t_tfop.gradient(tf_meanmax, coords)
        
        with tf.device("/cpu:0"):
            difference = meanmax - tf_meanmax
            max_rel_difference = tf.reduce_max(tf.abs(difference/(tf.abs(tf_meanmax)+1e-3))).numpy()
            max_difference = tf.reduce_max(tf.abs(difference)).numpy()
            
            #print('max rel difference',max_rel_difference)
            #print('max difference',max_difference)
            #print('op time',op_time)
            #print('tf time',tf_time)
            if assert_error:
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
            
            
            maxfeatgraddiff = tf.reduce_max(tf.abs(feat_grad_diff))
            maxcoordgraddiff = tf.reduce_max(tf.abs(coord_grad_diff))
            
            rel_feat_grad_diff = (feat_grad_diff)/(tf.abs(tf_feat_grad)+1e-2)
            rel_coord_grad_diff = coord_grad_diff/(tf.abs(tf_coord_grad)+1e-2)
            
            
            maxrelfeatgraddiff = tf.reduce_max(tf.abs(rel_feat_grad_diff))
            maxrelcoordgraddiff = tf.reduce_max(tf.abs(rel_coord_grad_diff))
            
            #print('\nmax relative feature grad diff', maxrelfeatgraddiff)
            #print('max relative coordinate grad diff', maxrelcoordgraddiff)
            
            def check_indices():
                idx_ok=True
                for i in tf.range(indices.shape[0]):
                    y,idx,c = tf.unique_with_counts(indices[i])
                    if (c.numpy() > 1).any() or (indices[i].numpy() >= indices.shape[0]).any() :
                        idx_ok=False
                        print("indices not unique", indices[i])
                if idx_ok:
                    print('indices ok')
            
            
            if self.debugout:
                print('custom feature grad ',feat_grad)
                print('TF feature grad',tf_feat_grad)
                print('difference',feat_grad_diff)
                
                print('custom coord grad',coord_grad)
                print('TF coord grad',tf_coord_grad)
                print('Difference',coord_grad_diff)
                    
            if maxrelfeatgraddiff > 1e-2:
                print('Feature gradient off:')
                print('max rel diff',maxrelfeatgraddiff)
                print('max diff',maxfeatgraddiff)
                print('min,max feat', tf.reduce_min(feats), tf.reduce_max(feats))
                print('min,max coords', tf.reduce_min(coords), tf.reduce_max(coords))
                check_indices()
                
            if maxrelcoordgraddiff > 1e-2:
                print('Coordinate gradient off:')
                print('max rel diff',maxrelcoordgraddiff)
                print('max diff',maxcoordgraddiff)
                print('min,max feat', tf.reduce_min(feats), tf.reduce_max(feats))
                print('min,max coords', tf.reduce_min(coords), tf.reduce_max(coords))
                check_indices()
                
            if maxfeatgraddiff > 1e-2:
                print('Feature gradient off:')
                print('max rel diff',maxrelfeatgraddiff)
                print('max diff',maxfeatgraddiff)
                print('min,max feat', tf.reduce_min(feats), tf.reduce_max(feats))
                print('min,max coords', tf.reduce_min(coords), tf.reduce_max(coords))
                check_indices()
                
            if maxcoordgraddiff > 1e-2:
                print('Coordinate gradient off:')
                print('max rel diff',maxrelcoordgraddiff)
                print('max diff',maxcoordgraddiff)
                print('min,max feat', tf.reduce_min(feats), tf.reduce_max(feats))
                print('min,max coords', tf.reduce_min(coords), tf.reduce_max(coords))
                check_indices()
            
            if assert_error:
                assert maxrelfeatgraddiff < 5e-2
                assert maxrelcoordgraddiff < 5e-2
                
            reldifference = tf.reshape(difference/(tf.abs(tf_meanmax)+1e-4),[-1])
            difference = tf.reshape(difference,[-1])
            rel_feat_grad_diff = tf.reshape(rel_feat_grad_diff,[-1])
            rel_coord_grad_diff = tf.reshape(rel_coord_grad_diff,[-1])
            feat_grad_diff = tf.reshape(feat_grad_diff,[-1])
            coord_grad_diff = tf.reshape(coord_grad_diff,[-1])
                
            return difference,reldifference,rel_feat_grad_diff,rel_coord_grad_diff,feat_grad_diff,coord_grad_diff
        
    def run_extended_difference(self,
                                nvert,
                                nneigh,
                                nfeat,
                                addstring=""):
        
        diff = []
        reldiff = []
        relcoordgraddiff = []
        relfeatgraddiff = []
        coordgraddiff = []
        featgraddiff = []
        for nv in nvert:
            for nn in nneigh:
                for nf in nfeat:
                    print('nv:',nv, 'nf:',nf, 'nn:' ,nn)
                    for blub in range(5): #run a few times
                        d,dr,fr,cr,f,c = self.difference(nv,nf,nn, ncoords = 4, onlyForward=False, assert_error=False)
                        #print('>>> max feat diff',tf.reduce_max(tf.abs(f)))
                        diff.append(d)
                        reldiff.append(dr)
                        coordgraddiff.append(c)
                        featgraddiff.append(f)
                        relcoordgraddiff.append(cr)
                        relfeatgraddiff.append(fr)
        
        def conc_and_reshape(intensor):
            x = tf.concat(intensor,axis=0)
            x = tf.reshape(x, [-1])
            return x.numpy()
        
        diff = conc_and_reshape(diff)
        reldiff = conc_and_reshape(reldiff)
        coordgraddiff = conc_and_reshape(coordgraddiff)
        featgraddiff = conc_and_reshape(featgraddiff)
        #print('total >>> max feat diff',tf.reduce_max(tf.abs(featgraddiff)))
        relcoordgraddiff = conc_and_reshape(relcoordgraddiff)
        relfeatgraddiff = conc_and_reshape(relfeatgraddiff)
        
        nbins=101
        
        print('plotting...')
        plt.close()
        plt.hist(diff, bins=nbins)
        plt.xlabel("Output Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"output_diff.pdf")
        plt.close()
        plt.hist(reldiff, bins=nbins)
        plt.xlabel("Relative Output Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"rel_output_diff.pdf")
        plt.close()
        plt.hist(coordgraddiff, bins=nbins)
        plt.xlabel("Coordinate Gradient Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"coord_grad_diff.pdf")
        plt.close()
        plt.hist(featgraddiff, bins=nbins)
        plt.xlabel("Feature Gradient Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"feat_grad_diff.pdf")
        plt.close()
        plt.hist(relcoordgraddiff, bins=nbins)
        plt.xlabel("Relative Coordinate Gradient Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"rel_coord_grad_diff.pdf")
        plt.close()
        plt.hist(relfeatgraddiff, bins=nbins)
        plt.xlabel("Relative Feature Gradient Difference")
        plt.yscale('log')
        plt.savefig(self.name+addstring+"rel_feat_grad_diff.pdf")
        plt.close()
        
    
        
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
            
        
        