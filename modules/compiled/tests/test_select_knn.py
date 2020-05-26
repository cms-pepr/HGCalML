import tensorflow as tf
import numpy as np

import time
from select_knn_op import SelectKnn
from rknn_op import *

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()
    
    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 2 and len(shape_B) == 2
    assert shape_A[0] == shape_B[0]# and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[1, 0]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=1), axis=1)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=1), axis=0)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


def createData(nvert,ncoords):
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    row_splits = tf.constant( [0,  nvert] ,dtype='int32')
    return coords, row_splits


def custom_impl(K, coords, row_splits):
    return SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-.2, tf_compatible=False)

def tf_implt(K, coords, row_splits):
    
    out_indices=[]
    for i in range(row_splits.shape[0]-1):
    
        distance_matrix = euclidean_squared(coords[row_splits[i]:row_splits[i+1]], coords[row_splits[i]:row_splits[i+1]])
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, K+1)
        ranked_indices += row_splits[i]
        out_indices.append(ranked_indices)
    
    return tf.concat(out_indices,axis=0)[:,1:]




coords, row_splits = createData(20, 4)

K = 10
print('launching custom')
x = custom_impl(K, coords, row_splits)
print('taking time')
t0 = time.time()
for i in range(20):
    #print(i)
    x = custom_impl(K, coords, row_splits)
c_time = (time.time()-t0)/20.

print('x',x)

print('custom time',c_time)

print('launching TF')
tf_x = tf_implt(K = K, coords=coords,  row_splits=row_splits)
t0 = time.time()
for i in range(20):
    tf_x = tf_implt(K = K, coords=coords,  row_splits=row_splits)
tf_time = (time.time()-t0)/20.
print('tfx',tf_x)

print('tf time',tf_time)



diff = tf.abs(tf_x - x)
maxdiff = tf.reduce_max(diff).numpy()

if maxdiff > 0 :
    #permutations lead to number that can be diveded by 2
    rest = tf.reduce_sum(diff, axis=-1)%2
    maxdiff = tf.reduce_sum(rest).numpy()
if maxdiff > 0 :
    print('diff',diff)
    print('rest',rest)
    print('maxdiff',maxdiff)
    
print('min', tf.reduce_min(x), 'max', tf.reduce_max(x))



indices, _ = rknn_op.RaggedKnn(num_neighbors=int(K), row_splits=row_splits, data=coords, add_splits=False)

t0 = time.time()
for i in range(20):
    indices, _ = rknn_op.RaggedKnn(num_neighbors=int(K), row_splits=row_splits, data=coords, add_splits=False)
rk_time = (time.time()-t0)/20.

diff = tf.abs(tf_x - indices)
maxdiff = tf.reduce_max(diff).numpy()

print('max idx',tf.reduce_max(indices))
print('min idx',tf.reduce_min(indices))

print('rknn time', rk_time )




