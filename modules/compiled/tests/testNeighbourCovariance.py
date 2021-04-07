

from neighbour_covariance_op import NeighbourCovariance

import numpy as np
import tensorflow as tf
from select_knn_op import SelectKnn
import time

n_vert=300000
n_coords=3
n_feats=2
n_neigh=64

coords = tf.random.uniform((n_vert,n_coords), dtype='float32',seed=2)
feats = tf.random.uniform((n_vert,n_feats), dtype='float32',seed=2)
row_splits = tf.constant( [0, n_vert] ,dtype='int32')

nidx,_ = SelectKnn(n_neigh, coords,  row_splits)


print('launching op')
cov,means = NeighbourCovariance(coords, feats, nidx)

t0=time.time()
for _ in range(20):
    cov,means = NeighbourCovariance(coords, feats, nidx)

print((time.time()-t0)/20)
print(cov.shape,cov)
print(means.shape)