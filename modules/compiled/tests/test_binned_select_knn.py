
import tensorflow as tf
import numpy as np
from binned_select_knn_op import BinnedSelectKnn
from select_knn_op import SelectKnn
import time



def createData(nvert,ncoords):
    coords = np.random.rand(nvert,ncoords)*1.04
    coords[:,0] *= 2.
    coords = tf.constant( coords ,dtype='float32')
    row_splits = tf.constant( [0,  nvert//2, nvert] ,dtype='int32')
    return coords, row_splits

for nvert in [300000]:
    coords, row_splits = createData(nvert,3)
    binwidth = tf.constant([0.1])
    
    idx, dist = SelectKnn(4, coords, row_splits)
    print(idx)
    print(dist)
    
    idx, dist = BinnedSelectKnn(4, coords, row_splits, bin_width=binwidth)
    
    print(idx)
    print(dist)
    from slicing_knn_op import SlicingKnn
    
    for bw in [0.01,0.05,0.1,0.2,0.4]:
        start = time.time()
        for _ in range(5):
            idx, dist = BinnedSelectKnn(64, coords, row_splits, bin_width=tf.constant([bw]))
        end =  time.time()
        print('took',(end-start)/5.,'s','for',nvert,'points and bin width',bw)
        start = time.time()
        for _ in range(5):
            idx, dist = SlicingKnn(64, coords, row_splits, features_to_bin_on=[0,1], bin_width=(bw,bw))
        end =  time.time()
        print('took',(end-start)/5.,'s','for',nvert,'points on SlicingKnn and bin width',bw)
    

    start = time.time()
    for _ in range(5):
        idx, dist = SelectKnn(64, coords, row_splits)
    end =  time.time()
    print('took',(end-start)/5.,'s','for',nvert,'points on SelectKnn')
    
    
    
    


