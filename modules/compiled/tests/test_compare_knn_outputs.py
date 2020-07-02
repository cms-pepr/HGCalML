from DeepJetCore.training.gpuTools import DJCSetGPUs
DJCSetGPUs()

import tensorflow as tf
import numpy as np

import time
from compare_knn_outputs_op import CompareKnnOutputs
from select_knn_op import SelectKnn
from rknn_op import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(True)

def createData(nvert,ncoords):
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    row_splits = tf.constant( [0,  nvert] ,dtype='int32')
    return coords, row_splits

def euclidean_squared(A, B):
    A = tf.expand_dims(A, axis = 1) #V x 1 x C
    B = tf.expand_dims(B, axis = 0) #1 x V x C
    return tf.reduce_sum((A-B)**2, axis=-1)

def selectNeighbours_TF(K, coords, row_splits, return_distances=False):

    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)

        out_indices=[]
        out_dst=[]
        for i in range(row_splits.shape[0]-1):

            distance_matrix = euclidean_squared(coords[row_splits[i]:row_splits[i+1]], coords[row_splits[i]:row_splits[i+1]])
            ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, K)
            ranked_indices += row_splits[i]
            out_indices.append(ranked_indices)
            out_dst.append(ranked_distances)

        if return_distances:

           idcs=tf.concat(out_indices,axis=0)[...,tf.newaxis]

           distances = tf.reduce_sum(
               (coords[:, tf.newaxis, :] - tf.gather_nd(coords,idcs)) ** 2,
               axis=-1)

    if return_distances:
        return (idcs, distances), t_newop
    return tf.concat(out_indices,axis=0), t_newop

def selectNeighbours_CUDA(K, coords, row_splits, return_distances=False):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop

def calculateIndecies(nVertices, nNeighbours):
    coords, row_splits = createData(nVertices, 4)

    #  print("***COORDS***")
    #  print(coords)

    ind_custom, _ = selectNeighbours_CUDA(nNeighbours, coords, row_splits, return_distances=False)

    ind_tf, _ = selectNeighbours_TF(nNeighbours, coords, row_splits, return_distances=False)

    return ind_tf, ind_custom


#****** MAIN ******
N_VERTICIES = 100
N_NEIGHBOURS = 20

ind_tf, ind_custom = calculateIndecies(N_VERTICIES,N_NEIGHBOURS)
ind_custom = ind_custom[0]

print("***INDECIES, CUDA IMPL:***")
print(ind_custom)
print("***INDECIES, TF IMPL:***")
print(ind_tf)

outTensor=compareTensors(ind_tf, ind_custom)
print("***COMPARISON TENSOR***")
print(outTensor)

#  with tf.device('/CPU:0'):
#      outTensor=compareTensors(ind_tf, ind_custom)
#      print("***COMPARISON TENSOR***")
#      print(outTensor)
#
#  with tf.device('/GPU:3'):
#      outTensor=compareTensors(ind_tf, ind_custom)
#      print("***COMPARISON TENSOR***")
#      print(outTensor)

    #  dummyT1 = tf.constant(2*N_NEIGHBOURS*np.random.rand(N_VERTICIES,N_NEIGHBOURS) ,dtype='int32')
    #  dummyT2 = tf.constant(2*N_NEIGHBOURS*np.random.rand(N_VERTICIES,N_NEIGHBOURS) ,dtype='int32')
    #  print("***DUMMY1***")
    #  print(dummyT1)
    #  print("***DUMMY2***")
    #  print(dummyT2)
    #
    #  outTensor=compareTensors(dummyT1, dummyT2)
    #  print("***COMPARISON TENSOR***")
    #  print(outTensor)



