#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from select_mod_knn_op import SelectModKnn as SelectKnn

#extent = 4

granularity = 100
psize = 3


x,y = np.meshgrid(np.linspace(0,4,granularity),np.linspace(0,4,granularity))

#make it position vectors
pos = np.concatenate( [np.expand_dims(x,axis=2),np.expand_dims(y,axis=2)],axis=-1 )
pos = np.reshape(pos, [-1,2])
cols = pos[...,0]*0.


transform = np.expand_dims(np.random.rand(2,2),axis=0)
tpos = np.expand_dims(pos, axis=1)
tpos = tpos*transform
tpos = np.sum(tpos, axis=-1)

radius_in_trsf = (np.max(tpos[...,0])-np.min(tpos[...,0]))/20.

cols = np.where((tpos[...,0]-tpos[granularity//2:granularity//2+1,0])**2 + 
                (tpos[...,1]-tpos[granularity//2:granularity//2+1,1])**2 < radius_in_trsf, 1, cols)

selpos = pos#[np.max(pos,axis=-1)<extent]
selcols = cols#[np.max(pos,axis=-1)<extent]
selcols[granularity//2] = 2

plt.scatter(selpos[...,0],selpos[...,1],c=selcols,s=psize)
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()

coord_mod=tf.constant(transform,dtype='float32')
coord_mod = tf.tile(coord_mod, [len(pos),1,1])
print(coord_mod.shape)


nidx, dist = SelectKnn(K=granularity*granularity, 
          coords=tf.constant(pos,dtype='float32'), 
          coord_mod=coord_mod, 
          row_splits=tf.constant([0, len(pos)],dtype='int32'), 
          max_radius=-1)

knncols = np.where(dist[granularity//2] < radius_in_trsf, 3, np.zeros_like(cols))
print('knncols',knncols.shape)
print(dist[granularity//2])
knncols[nidx[granularity//2]] = knncols #get indices right again

plt.scatter(selpos[...,0],selpos[...,1],c=knncols,s=psize)
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()


plt.scatter(tpos[...,0],tpos[...,1],c=cols,s=psize)
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()
