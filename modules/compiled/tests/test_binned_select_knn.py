
import tensorflow as tf
import numpy as np
from binned_select_knn_op import BinnedSelectKnn
from select_knn_op import SelectKnn
import time



def createData(nvert,ncoords):
    coords = np.random.rand(nvert,ncoords)*1.04
    coords[:,0] *= 1.1
    coords = tf.constant( coords ,dtype='float32')
    row_splits = tf.constant( [0,  nvert//2, nvert] ,dtype='int32')
    return coords, row_splits


for nvert in [1000,50000,200000,500000,1000000]:
    
    coords, row_splits = createData(nvert,3)
    
    idx, dist = SelectKnn(4, coords, row_splits)
    print(idx)
    print(dist)
    
    
    idx, dist = BinnedSelectKnn(4, coords, row_splits)
    
    print(idx)
    print(dist)
    
    #exit()
    
    from slicing_knn_op import SlicingKnn
    
    for test_neighbours in [32,64,256]:
        
        nbins = None
        
        idx, dist = BinnedSelectKnn(test_neighbours, coords, row_splits, n_bins=nbins)
        start = time.time()
        for _ in range(5):
            idx, dist = BinnedSelectKnn(test_neighbours, coords, row_splits, n_bins=nbins)
        end =  time.time()
        print('took',(end-start)/5.,'s','for',nvert,'points on',nbins,'and nn',test_neighbours )
        
        idx, dist = SlicingKnn(test_neighbours, coords, row_splits, features_to_bin_on=[0,1], bin_width=(0.02,0.02))
        start = time.time()
        for _ in range(5):
            idx, dist = SlicingKnn(test_neighbours, coords, row_splits, features_to_bin_on=[0,1], bin_width=(0.02,0.02))
        end =  time.time()
        print('took',(end-start)/5.,'s','for',nvert,'and nn',test_neighbours,'points on SlicingKnn')
    
    continue
    idx, dist = SelectKnn(test_neighbours, coords, row_splits)
    start = time.time()
    for _ in range(5):
        idx, dist = SelectKnn(test_neighbours, coords, row_splits)
    end =  time.time()
    print('took',(end-start)/5.,'s','for',nvert,'points on SelectKnn')
    
    
    
    


