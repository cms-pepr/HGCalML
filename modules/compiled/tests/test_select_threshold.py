
import tensorflow as tf
from select_threshold_op import SelectThreshold
import numpy as np
import time

# SelectThreshold(x, pl, rowsplits, threshold=0.5)


nvert=10000
nfeat=128

xs = tf.constant(np.random.rand(nvert) ,dtype='float32')
xs = tf.reshape(xs, [-1,1])
rs = tf.constant([0,int(nvert/4),int(nvert/2),nvert],dtype='int32')
pl = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')

print(xs, pl, rs)

newfeat, newrs, scatter_idxs = SelectThreshold(xs,pl,rs,threshold=0.5)

bef = time.time()
for  _ in range(20):
    newfeat, newrs, scatter_idxs = SelectThreshold(xs,pl,rs,threshold=0.5)
totaltime = time.time() - bef



print('output')
print(newfeat, rs, scatter_idxs)

print('scattered back')
print(tf.scatter_nd(scatter_idxs, newfeat ,shape=pl.shape))

print('total time', totaltime)