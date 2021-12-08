import tensorflow as tf
import numpy as np
import time
import sys
import os
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
    return coords

def createBlobData(nvert,ncoords,seed,n_blobs):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=nvert, centers=n_blobs, n_features=ncoords,
                      random_state=seed)
    coords = tf.constant( X ,dtype='float32')
    return coords

def selectNeighbours_CUDA(K, coords, row_splits,radius):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=radius, tf_compatible=True)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop


if __name__ == '__main__':

    # CONFIGURATIONS
    N_VERTICIES = 200000
    #  N_VERTICIES = 200
    N_NEIGHBOURS = 50
    N_DIMS = 4
    seed = 12345
    n_bins_x = 10
    n_bins_y = n_bins_x
    KnnKernelType = "SlicingKnn"
    N_ITERS = 10
    N_BLOBS = 50
    radius = -1.0

    #  log_file = "kNN_timing_v2.csv"
    #  ROW_SPLITS = (0, int(N_VERTICIES/2), N_VERTICIES)

    log_file = "kNN_timing_blobs.csv"
    ROW_SPLITS = (0, N_VERTICIES)

    #  coords = createData(N_VERTICIES, N_DIMS, seed)
    #  coords = 26*coords

    coords = createBlobData(N_VERTICIES, N_DIMS, seed, N_BLOBS)

    row_splits = tf.constant(ROW_SPLITS ,dtype='int32')

    r_coords = tf.transpose(coords) # since tf.map_fn apply fn to each element unstacked on axis 0
    r_max = tf.map_fn(tf.math.reduce_max, r_coords, fn_output_signature=tf.float32)
    r_min = tf.map_fn(tf.math.reduce_min, r_coords, fn_output_signature=tf.float32)
    print("***COORD MAX***")
    print(r_max)
    print("***COORD MIN***")
    print(r_min)

    # TESTING
    #  if True

    # SelectKnn
    KnnKernelType = "SelectKnn"
    for tmp_val in (-1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1, 0.05, 0.01):
        radius = 12*tmp_val

    # SlicingKnn
    #  KnnKernelType = "SlicingKnn"
    #  for tmp_val in (3,5,10,20,30,40,50,75,100,200,300,400):
    #      n_bins_x = tmp_val
    #      n_bins_y = tmp_val

        # SelectKnn results to calculate recall
        ind_exact, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits,-1.0)
        ind_exact = ind_exact[0]
        exec_time = 0.0
        recall = -1.0

        if KnnKernelType == "SelectKnn":
            # Measure performance of SelectKnn
            start_time = time.time()
            for _ in range(0,N_ITERS):
                out_select_knn, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits,radius)
            exec_time = (time.time() - start_time) / N_ITERS
            print("SelectKnn --- %f seconds ---" % exec_time)
            n_bins_x = -1
            n_bins_y = -1

            ind_select_knn = out_select_knn[0]
            outTensor=compareTensors(ind_exact, ind_select_knn)
            n_mistakes = np.sum(outTensor[0].numpy())
            recall = 1.0-n_mistakes/(N_VERTICIES*N_NEIGHBOURS)
            print("recall: ",recall)

        elif KnnKernelType == "SlicingKnn":
            radius = -1.0

            ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(n_bins_x,n_bins_y))
            start_time = time.time()
            for _ in range(0,N_ITERS):
                ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(n_bins_x,n_bins_y))
            exec_time = (time.time() - start_time) / N_ITERS
            print("SlicingKnn --- %f seconds ---" % exec_time)

            outTensor=compareTensors(ind_slice_knn, ind_exact)
            n_mistakes = np.sum(outTensor[0].numpy())
            recall = 1.0-n_mistakes/(N_VERTICIES*N_NEIGHBOURS)
            print("recall: ",recall)

        else:
            print("No kNN kernel of type: %s" % KnnKernelType)
            sys.exit(0)

        if not os.path.exists(log_file):
            with open(log_file,"w") as f:
                f.write("kernelType,seed,n_events,n_dims,row_splits,K,n_iters,n_bins_x,n_bins_y,radius,n_blobs,time_per_iter,recall\n")

        with open(log_file,"a+") as f:
            f.write("%s,%d,%d,%d,%s,%d,%d,%d,%d,%f,%d,%f,%f\n" % (KnnKernelType,seed,N_VERTICIES,N_DIMS,"-".join([str(x) for x in ROW_SPLITS]),N_NEIGHBOURS,N_ITERS,n_bins_x,n_bins_y,radius,N_BLOBS,exec_time,recall))
