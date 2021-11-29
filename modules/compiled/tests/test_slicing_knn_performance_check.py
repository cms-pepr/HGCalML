import tensorflow as tf
import numpy as np
import time
import sys
from compare_knn_outputs_op import CompareKnnOutputs
from select_knn_op import SelectKnn
from slicing_knn_op import SlicingKnn
import unittest

from DeepJetCore.training.gpuTools import DJCSetGPUs
DJCSetGPUs()

gpus = tf.config.list_physical_devices('GPU')
#  if gpus:
#    try:
#      # Currently, memory growth needs to be the same across GPUs
#      for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)
#      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#      # Memory growth must be set before GPUs have been initialized
#      print(e)



tf.debugging.set_log_device_placement(False)

def createData(nvert,ncoords,seed):
    np.random.seed(seed)
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    row_splits = tf.constant( [0, int(nvert/2),  nvert] ,dtype='int32')
    return coords, row_splits

def selectNeighbours_CUDA(K, coords, row_splits):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1.0, tf_compatible=True)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop


if __name__ == '__main__':

    N_VERTICIES = 200000
    N_NEIGHBOURS = 50
    N_DIMS = 10
    seed = 12345
    coords, row_splits = createData(N_VERTICIES, N_DIMS, seed)
    row_splits = tf.constant( [0, int(N_VERTICIES/2), N_VERTICIES] ,dtype='int32')

    #  print(coords)

    import time
    out_select_knn, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits)
    start_time = time.time()
    for _ in range(0,10):
        out_select_knn, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits)

    print("SelectKnn --- %s seconds ---" % (time.time() - start_time))
    ind_select_knn = out_select_knn[0]
    dist_select_knn = out_select_knn[1]

    ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(4,4))
    start_time = time.time()
    for _ in range(0,10):
        ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(4,4))
    print("SlicingKnn --- %s seconds ---" % (time.time() - start_time))


    outTensor=compareTensors(ind_slice_knn, ind_select_knn)
    n_mistakes = np.sum(outTensor[0].numpy())
    if n_mistakes>=1:

        print("recall: ",1.0-n_mistakes/(N_VERTICIES*N_NEIGHBOURS))
