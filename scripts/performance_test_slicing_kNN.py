# ********** CONFIG **********
SEED=1234

# if False - use currently used CUDA kernel
use_slicing_method = True
#  use_slicing_method = False

#  N_ITERS = 10
N_ITERS = 20

#  x_axis = "n_vert"
#  N_VERTS = [10000, 30000, 50000, 100000, 200000]
#  KS = [50, 100, 200]
#  N_DIMS = [4]
#  N_BINS = [5]

#  x_axis = "n_dims"
#  N_VERTS = [50000, 100000, 200000]
N_VERTS = [300000]
N_SPLITS = 1
RANDOM_SPLITS = False
#  N_VERTS = [400000]
#  N_VERTS = [100000]
KS = [50]
#  N_DIMS = [2,4,8,16,32,64]
N_DIMS = [4]
#  N_DIMS = [8,16]
#  N_BINS = [5, 10, 15, 20]
N_BINS = [15]
#  N_BINS = [15]

#  x_axis = "n_bins"
#  N_VERTS = [100000]
#  KS = [100]
#  N_DIMS = [4]
#  N_BINS = [3,4,5,6,7,8,9,10]


machine = "1080"

#  machine = "T4"
#  gpus_to_use = "1" # "comma-separated string"

#  machine = "V100"
#  gpus_to_use = "2" # "comma-separated string"

allow_growing_memory = False
memory_limit= -666 # Mb
# ********** CONFIG end **********


# construct name of out txt-file
out_textfile = "./"
out_textfile += "test_flat_rndm_1Sep2021_Slicing.csv"

from DeepJetCore.training.gpuTools import DJCSetGPUs
if machine=="1080":
    DJCSetGPUs()
else:
    DJCSetGPUs(gpus_to_use)

import tensorflow as tf
import numpy as np
import os

import time
from compare_knn_outputs_op import CompareKnnOutputs
from select_knn_op import SelectKnn
from slicing_knn_op import SlicingKnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(False)


# configure GPUs settings for execution
if (memory_limit>0 or allow_growing_memory):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if allow_growing_memory:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if memory_limit>0:
                    tf.config.experimental.set_virtual_device_configuration(
                            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def createData(nvert,ncoords,seed,n_splits=1,random_splits=False):
    np.random.seed(seed)
    np_coords = np.random.rand(nvert,ncoords)
    coords = tf.constant( np_coords ,dtype='float32')
    tmp_arr = []
    if random_splits:
        rand_arr = [np.random.rand() for _ in range(n_splits)]
        rand_arr /= np.sum(rand_arr)
        rand_arr = (nvert*rand_arr).astype(int)
        tmp_arr  = [0]
        for iEl in rand_arr:
            tmp_arr.append(tmp_arr[-1]+iEl)
        tmp_arr[-1] = nvert
    else:
        tmp_arr = [int(i*nvert/n_splits) for i in range(n_splits+1)]
    row_splits = tf.constant(tmp_arr, dtype='int32')
    return coords, row_splits, np_coords

def selectNeighbours_CUDA(K, coords, row_splits, return_distances=False):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True)
    return out, t_newop

def calculateIndecies_newKernel(K_, coords_, row_splits, features_to_bin_on, n_bins):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SlicingKnn(K = K, coords=coords, row_splits=row_splits, features_to_bin_on = features_to_bin_on, n_bins=n_bins)
    return out

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop

def calculate_recall(a,b):
    a = tf.dtypes.cast(a, tf.int32)
    b = tf.dtypes.cast(b, tf.int32)
    c = tf.sets.intersection(a,b)
    return c.values.shape[0]/(a.shape[0]*a.shape[1])

def getRowSplitsString(row_splits):
    outStr = ""
    for iEl in row_splits.numpy():
        outStr = outStr + str(iEl) + "_"
    outStr = outStr[:-1]
    return outStr

### MAIN ###
if (use_slicing_method == False):
    N_BINS = [1]

for N_DIM in N_DIMS:
    for N_VERT in N_VERTS:
        coords, row_splits, np_coords = createData(N_VERT, N_DIM, SEED+N_DIM+N_VERT+66*N_SPLITS,N_SPLITS,RANDOM_SPLITS)
        N_VERTICIES = N_VERT
        for N_BIN in N_BINS:
            N_BINS_X = N_BIN
            N_BINS_Y = N_BIN

            for K in KS:
                print('launching Slicing CUDA kernel')
                print("n_bins_x: ",N_BIN)
                #  print("*** First 3 coords: ***")
                #  print(coords[0:3,:])
                recall = -1.0
                if use_slicing_method:
                    ref_indc, _ = selectNeighbours_CUDA(K, coords, row_splits, True)
                    slice_indc = calculateIndecies_newKernel(K,coords, row_splits, (0,1), (N_BINS_X, N_BINS_Y))
                    recall = calculate_recall(ref_indc[0],slice_indc[0])
                    #  recall = -666
                    print()
                    print("*** Neighbours for the first 3 vtx: ***")
                    print(slice_indc[0][0:3,:])
                    print()
                else:
                    x,_ = selectNeighbours_CUDA(K, coords, row_splits, True)
                    print()
                    print("*** Neighbours for the first 3 vtx: ***")
                    print(x[0][0:3,:])
                    print()

                print('taking time')
                t0 = time.time()
                for i in range(N_ITERS):
                    if use_slicing_method:
                        _ = calculateIndecies_newKernel(K,coords, row_splits, (0,1), (N_BINS_X,N_BINS_Y))
                    else:
                        x,g = selectNeighbours_CUDA(K, coords, row_splits, True)
                exec_time = (time.time()-t0)/N_ITERS

                print('\n\n**********************************')
                print('*** Exec time: %f seconds' % (exec_time))
                if (use_slicing_method):
                    print('*** Recall: %f' % (recall))
                print('*** N_VERT = %d' %(N_VERT))
                print('*** K = %d' %(K))
                print('*** N_BIN = %d' %(N_BIN))
                print('*** N_DIM = %d' %(N_DIM))
                print('*** use_slicing_method = %d' %(use_slicing_method))
                print('*** N_ITERS = %d' %(N_ITERS))
                print('*** row_splits = %s' %(getRowSplitsString(row_splits)))
                print('**********************************')


                from datetime import datetime
                now = datetime.now() # current date and time
                date_time = now.strftime("%m/%d/%Y,%H:%M:%S")

                if not os.path.isfile(out_textfile):
                    f= open(out_textfile,"a+")
                    f.write("date,time,machine,n_vertices,K,n_bins,n_dims,exec_time,recall,tag,row_splits,n_iters\n")
                    f.close()

                f= open(out_textfile,"a+")
                if (use_slicing_method):
                    f.write('%s,"%s",%d,%d,%d,%d,%.3f,%.2f,"%s",%s,%d\n' % (date_time, machine, N_VERT, K, N_BIN*N_BIN, N_DIM, exec_time, recall, "Slicing", getRowSplitsString(row_splits),N_ITERS))
                else:
                    f.write('%s,"%s",%d,%d,%d,%d,%.3f,%.2f,"%s,%s,%d"\n' % (date_time, machine, N_VERT, K, -1, N_DIM, exec_time, recall, "old_cuda", getRowSplitsString(row_splits),N_ITERS))


                f.close()
exit()
