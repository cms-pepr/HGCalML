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
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop

class TestKnn(unittest.TestCase):

    def setUp(self):
        self.N_VERTICIES = 200000
        self.N_NEIGHBOURS = 30
        self.N_DIMS = 4
        self.seed = 12345
        self.coords, self.row_splits = createData(self.N_VERTICIES, self.N_DIMS, self.seed)

        out_old_cuda, _ = selectNeighbours_CUDA(self.N_NEIGHBOURS, self.coords, self.row_splits)
        self.ind_old_cuda = out_old_cuda[0]
        self.dist_old_cuda = out_old_cuda[1]

    def test_first_coordinate_index_out_of_range(self):
        with self.assertRaises(Exception) as context:
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (self.N_DIMS,1), n_bins=(8,8))
        self.assertTrue('Value error' in str(context.exception))

    def test_second_coordinate_index_out_of_range(self):
        with self.assertRaises(Exception) as context:
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,self.N_DIMS), n_bins=(8,8))
        self.assertTrue('Value error' in str(context.exception))

    def test_identical_coordinate_indices(self):
        with self.assertRaises(Exception) as context:
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,0), n_bins=(8,8))
        self.assertTrue('Value error' in str(context.exception))

    def test_specify_n_bins_and_bin_width(self):
        with self.assertRaises(Exception) as context:
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), n_bins=(8,8), bin_width=(0.13,0.13))
        self.assertTrue('Specify either' in str(context.exception))

    def test_specify_no_n_bins_and_no_bin_width(self):
        with self.assertRaises(Exception) as context:
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1))
        self.assertTrue('Specify either' in str(context.exception))

    def test_compare_indices_old_cuda_vs_knn(self):
        ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), n_bins=(8,8))
        outTensor=compareTensors(self.ind_old_cuda, ind_slice_knn)
        self.assertEqual(np.sum(outTensor[0].numpy()),0)

    def test_compare_indices_old_cuda_vs_knn_specify_bin_width(self):
        ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), bin_width=(0.13,0.13))
        outTensor=compareTensors(self.ind_old_cuda, ind_slice_knn)
        self.assertEqual(np.sum(outTensor[0].numpy()),0)


if __name__ == '__main__':
    unittest.main()

    #  N_VERTICIES = 200000
    #  N_NEIGHBOURS = 50
    #  N_DIMS = 10
    #  seed = 12345
    #  coords, row_splits = createData(N_VERTICIES, N_DIMS, seed)
    #  row_splits = tf.constant( [0, int(N_VERTICIES/2), N_VERTICIES] ,dtype='int32')
    #
    #  #  print(coords)
    #
    #  import time
    #  out_old_cuda, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits)
    #  start_time = time.time()
    #  for _ in range(0,10):
    #      out_old_cuda, _ = selectNeighbours_CUDA(N_NEIGHBOURS, coords, row_splits)
    #
    #  print("OLD --- %s seconds ---" % (time.time() - start_time))
    #  ind_old_cuda = out_old_cuda[0]
    #  dist_old_cuda = out_old_cuda[1]
    #
    #  ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(4,4))
    #  start_time = time.time()
    #  for _ in range(0,10):
    #      ind_slice_knn, dist_slice_knn = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(4,4))
    #  print("NEW --- %s seconds ---" % (time.time() - start_time))
    #
    #
    #  outTensor=compareTensors(ind_slice_knn, ind_old_cuda)
    #  n_mistakes = np.sum(outTensor[0].numpy())
    #  if n_mistakes>=1:
    #      #  print("outTensor:")
    #      #  print(outTensor)
    #      #
    #      #  print("ind_old_cuda:")
    #      #  print(ind_old_cuda)
    #      #
    #      #  print("ind_slice_knn:")
    #      #  print(ind_slice_knn)
    #
    #      #  print("dist_old_cuda:")
    #      #  print(dist_old_cuda)
    #      #
    #      #  print("dist_slice_knn:")
    #      #  print(dist_slice_knn)
    #
    #      print("recall: ",1.0-n_mistakes/(N_VERTICIES*N_NEIGHBOURS))
    #
    #  #
    #  #  coords, row_splits = createData(N_VERTICIES, N_DIMS, seed+1)
    #  #  _, _ = SlicingKnn(K = N_NEIGHBOURS, coords=coords, row_splits=row_splits, features_to_bin_on = (0,1), n_bins=(8,8))
    #  #
    #  #  print("ind_slice_knn:")
    #  #  print(ind_slice_knn)
