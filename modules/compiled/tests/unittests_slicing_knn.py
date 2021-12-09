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
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



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

        out_selectKnn_cuda, _ = selectNeighbours_CUDA(self.N_NEIGHBOURS, self.coords, self.row_splits)
        self.ind_selectKnn_cuda = out_selectKnn_cuda[0]
        self.dist_selectKnn_cuda = out_selectKnn_cuda[1]

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

    def test_compare_indices_selectKnn_cuda_vs_knn(self):
        ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), n_bins=(8,8))
        outTensor=compareTensors(self.ind_selectKnn_cuda, ind_slice_knn)
        self.assertEqual(np.sum(outTensor[0].numpy()),0)

    def test_compare_indices_selectKnn_cuda_vs_knn_specify_bin_width(self):
        ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), bin_width=(0.13,0.13))
        outTensor=compareTensors(self.ind_selectKnn_cuda, ind_slice_knn)
        self.assertEqual(np.sum(outTensor[0].numpy()),0)

    def test_cpu_specialized_slicingknn_kernel(self):
        with tf.device('/cpu:0'):
            ind_slice_knn, self.dist_slice_knn = SlicingKnn(K = self.N_NEIGHBOURS, coords=self.coords, row_splits=self.row_splits, features_to_bin_on = (0,1), n_bins=(8,8))
            outTensor=compareTensors(self.ind_selectKnn_cuda, ind_slice_knn)
            self.assertEqual(np.sum(outTensor[0].numpy()),0)


if __name__ == '__main__':
    unittest.main()
