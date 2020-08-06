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


def selectNeighbours_NewKnnCPU(K, coords, row_splits, return_distances, _n_bins_x, _n_bins_y):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = NewKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True, n_bins_x = _n_bins_x, n_bins_y = _n_bins_y)
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
#  N_VERTICIES = 500
#  N_NEIGHBOURS = 30
#  N_DIMS = 4
#
#  with tf.device('/CPU:0'):
#      ind_tf, ind_custom, ind_newKnn = calculateIndecies(N_VERTICIES,N_NEIGHBOURS, N_DIMS)
#
#  print("***DISTANCES, TF IMPL:***")
#  print(ind_tf[1])
#  print("***DISTANCES, NEW_KNN IMPL:***")
#  print(ind_newKnn[1])
#
#  ind_custom = ind_custom[0]
#  ind_newKnn = ind_newKnn[0]
#  ind_tf =tf.squeeze(ind_tf[0],axis=2)
#
#  print("***INDECIES, CUDA IMPL:***")
#  print(ind_custom)
#  print("***INDECIES, TF IMPL:***")
#  print(ind_tf)
#  print("***INDECIES, NEW_KNN IMPL:***")
#  print(ind_newKnn)
#
#  outTensor=compareTensors(ind_tf, ind_custom)
#  print("***COMPARISON TENSOR: TF vs CUDA***")
#  print(outTensor)
#
#  outTensor=compareTensors(ind_tf, ind_newKnn)
#  print("***COMPARISON TENSOR: TF vs. NEW_KNN ***")
#  print(outTensor)





    #  reldiff = []
    #  for it in range(1):
    #      coords, row_splits = createData(4000, 4)
    #      K = 200
    #
    #
    #      for i in range(1,5):
    #          x_c, gt_c = selectNeighbours_NewKnnCPU(K, coords, row_splits, True,i,i)
    #      #  d_c = x_c[1]
    #      #  x_c = x_c[0]
    #
    #      #print('custom ids',x_c)
    #      #  custom_grad=gt_c.gradient(d_c, coords)
    #      #print('custom grad', custom_grad)
    #
    #      x_tf, gt_tf = selectNeighbours_TF(K, coords, row_splits, return_distances=True)
    #      #  d_tf = x_tf[1]
    #      #  x_tf = x_tf[0]
    #      #print(d_tf)
    #      #  tfgrad = gt_tf.gradient(d_tf, coords)
    #      #print('tf ids',x_tf)
    #      #print('tf grad',tfgrad)
    #      #  reldiff.append((custom_grad-tfgrad)/(tfgrad+1e-3))
    #      #  print('max diff',tf.reduce_max(tf.abs(custom_grad-tfgrad)), 'max rel grad diff', tf.reduce_max(  tf.abs(custom_grad-tfgrad)/tf.abs(tfgrad+1e-3) ))
    #      #  print('mean abs grad',tf.reduce_mean(tf.abs(tfgrad)))
    #
    #  #  print('plot')
    #  #  plt.hist(np.reshape(tf.concat(reldiff,axis=0).numpy(),[-1]))
    #  #  plt.xlabel("Rel Output Difference")
    #  #  plt.yscale('log')
    #  #  plt.savefig("select_grad_output_diff.pdf")
    #  #  plt.close()


    #exit()

    #  for i in range(2,3):
    #  print('launching custom, i: %f' % (i))

n_bins_x = 2
n_bins_y = 2
N_ITERS = 2

N_VERT = 100000
K = 500

coords, row_splits = createData(N_VERT, 4)

#  with tf.device('/CPU:0'):
#      print('launching CPU kernel')
#      x = selectNeighbours_NewKnnCPU(K, coords, row_splits, True,n_bins_x,n_bins_y)
#      print('taking time')
#      t0 = time.time()
#      for i in range(N_ITERS):
#          #print(i)
#          x,g = selectNeighbours_NewKnnCPU(K, coords, row_splits, True, n_bins_x,n_bins_y)
#          #  gg=g.gradient(x[1], coords)
#          #print(gg)
#      c_time = (time.time()-t0)/N_ITERS
#      print('custom time',c_time)


print('launching NEW CUDA kernel')
print("n_bins_x: ",n_bins_x)
x = selectNeighbours_NewKnnCPU(K, coords, row_splits, True,n_bins_x,n_bins_y)

#  print('x',x)
print('taking time')
t0 = time.time()
for i in range(N_ITERS):
    x,g = selectNeighbours_NewKnnCPU(K, coords, row_splits, True, n_bins_x,n_bins_y)
new_cuda_time = (time.time()-t0)/N_ITERS
print('NEW CUDA time',new_cuda_time)





print('launching OLD CUDA kernel')
print("n_bins_x: ",n_bins_x)
x = selectNeighbours_CUDA(K, coords, row_splits, return_distances=False)

#  print('x',x)
print('taking time')
t0 = time.time()
for i in range(N_ITERS):
    x,g = selectNeighbours_CUDA(K, coords, row_splits, return_distances=False)
current_cuda_time = (time.time()-t0)/N_ITERS
print('OLD CUDA time',current_cuda_time)

print('\n\n**********************************')
print('N_VERT: ',N_VERT)
print('K: ',K,"\n")
print('OLD: ',round(current_cuda_time,3),'; NEW(',n_bins_x,'x',n_bins_y,'): ', round(new_cuda_time,3))
print('**********************************')

exit()
