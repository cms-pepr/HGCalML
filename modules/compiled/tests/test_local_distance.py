

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test
import time
from local_distance_op import LocalDistance
from select_knn_op import SelectKnn

def createData(n_vert, n_coords):
    
    coords = tf.random.uniform((n_vert,n_coords), dtype='float32',seed=2)
    row_splits = tf.constant([0,n_vert],dtype='int32')
    
    
    return coords, row_splits #row splits just for other ops, no need to test

n_neigh=5
max_radius=-1

coords,  row_splits  = createData(10, 3)
neigh,dst = SelectKnn(n_neigh, coords,  row_splits, max_radius = max_radius, tf_compatible=True)
dst=None
dst2=None
with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
    t_newop.watch(coords)
    dst2 = LocalDistance(coords, neigh)
    coord_grad = t_newop.gradient(dst2, coords)
    

with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_compareop:
    t_compareop.watch(coords)
    _,dst = SelectKnn(n_neigh, coords,  row_splits, max_radius = max_radius, tf_compatible=True)
    
    coord_grad_cmp = t_compareop.gradient(dst, coords)
    

print('dst2',dst2)
print('diff ',dst2-dst)
print('grad',coord_grad)
print('diff',coord_grad-coord_grad_cmp)