
import tensorflow as tf
import numpy as np
from bin_by_coordinates_op import BinByCoordinates
import time

def createData(nvert,ncoords):
    coords = np.random.rand(nvert,ncoords)*1.04
    coords[:,0] *= 2.
    coords = tf.constant( coords ,dtype='float32')
    row_splits = tf.constant( [0,  nvert//2, nvert] ,dtype='int32')
    return coords, row_splits

coordinates, row_splits = createData(10,3)
binwidth = tf.constant([0.3])


BinByCoordinates(coordinates, row_splits, binwidth)
start = time.time()
for _ in range(20):
    binning, fbinning, nperbin, nb = BinByCoordinates(coordinates, row_splits, binwidth)

ntime = time.time()-start
print(nb, ntime/20.)

print(nperbin)

if coordinates.shape[1]==2:
    coordinates = np.concatenate([coordinates, np.zeros_like(coordinates[:,0:1])],axis=-1)
np.save("coords.npy",coordinates)
np.save("binning.npy",fbinning)


#exit()


print(nb)

sorting = tf.argsort(fbinning)

scoords = tf.gather_nd( coordinates, sorting[...,tf.newaxis])
sbinning = tf.gather_nd( fbinning, sorting[...,tf.newaxis])


bin_boundaries = nperbin
# add a leading zero to n_per_bin
bin_boundaries = tf.concat([row_splits[0:1], bin_boundaries],axis=0)
# make it row split like
bin_boundaries = tf.cumsum(bin_boundaries)
print(bin_boundaries)

from binned_select_knn_op import _BinnedSelectKnn

idx,dist = _BinnedSelectKnn(5, scoords,  sbinning, bin_boundaries=bin_boundaries, n_bins=nb, bin_width=binwidth )

from index_replacer_op import IndexReplacer

print('pre-resort')
print(idx)
print(dist)


idx = IndexReplacer(idx,sorting)
dist = tf.scatter_nd(sorting[...,tf.newaxis], dist, dist.shape)
idx = tf.scatter_nd(sorting[...,tf.newaxis], idx, idx.shape)

print('post-resort')
print(idx)
print(dist)

from select_knn_op import SelectKnn
idx, dist = SelectKnn(5, coordinates, row_splits) #scoords should not change row splits

print('SelectKnn')
print(idx)
print(dist)

#sort back









