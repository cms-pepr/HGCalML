from DeepJetCore.training.gpuTools import DJCSetGPUs
DJCSetGPUs()

import tensorflow as tf
import numpy as np

import time
from compare_knn_outputs_op import CompareKnnOutputs
from select_knn_op import SelectKnn
from new_knn_op import NewKnn
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


def selectNeighbours_NewKnnCPU(K, coords, row_splits, return_distances=False):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = NewKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True, n_bins_x = 3, n_bins_y = 3)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop

def calculateIndecies(nVertices, nNeighbours, nDims = 4):
    coords, row_splits = createData(nVertices, nDims)

    #  coords = tf.constant([[0.,0.,0.], [3.,4.,100.], [11.,11.,0.], [100.,100.,0.]])

    print("***COORDS***")
    print(coords)
    #  print("***ROW_SPLITS***")
    #  print(row_splits)

    ind_custom, _ = selectNeighbours_CUDA(nNeighbours, coords, row_splits, return_distances=False)

    ind_newKnn, _ =  selectNeighbours_NewKnnCPU(nNeighbours, coords, row_splits, return_distances=False)

    ind_tf, _ = selectNeighbours_TF(nNeighbours, coords, row_splits, return_distances=True)

    return ind_tf, ind_custom, ind_newKnn


#****** MAIN ******
N_VERTICIES = 50
N_NEIGHBOURS = 10
N_DIMS = 4

with tf.device('/CPU:0'):
    ind_tf, ind_custom, ind_newKnn = calculateIndecies(N_VERTICIES,N_NEIGHBOURS, N_DIMS)

print("***DISTANCES, TF IMPL:***")
print(ind_tf[1])
print("***DISTANCES, NEW_KNN IMPL:***")
print(ind_newKnn[1])

ind_custom = ind_custom[0]
ind_newKnn = ind_newKnn[0]
ind_tf =tf.squeeze(ind_tf[0],axis=2)

print("***INDECIES, CUDA IMPL:***")
print(ind_custom)
print("***INDECIES, TF IMPL:***")
print(ind_tf)
print("***INDECIES, NEW_KNN IMPL:***")
print(ind_newKnn)

outTensor=compareTensors(ind_tf, ind_custom)
print("***COMPARISON TENSOR: TF vs CUDA***")
print(outTensor)

outTensor=compareTensors(ind_tf, ind_newKnn)
print("***COMPARISON TENSOR: TF vs. NEW_KNN ***")
print(outTensor)

#  tmp = outTensor.numpy()
#  if (np.sum(tmp)>0):
#      print ("***MISTAKE FOUND!!!***")

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



