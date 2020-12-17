import tensorflow as tf
import numpy as np
import time
from select_knn_op import SelectKnn
from local_cluster_op import LocalCluster as lc
import matplotlib.pyplot as plt
from ragged_plotting_tools import make_cluster_coordinates_plot


radius=1.0
nvert=10000
K=300

def createData(nvert,ncoords=2):
    coords = 1.*tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    hier = tf.constant( np.random.rand(nvert) ,dtype='float32')
    row_splits = tf.constant( [0, nvert//2,  nvert] ,dtype='int32')
    sel,_ = SelectKnn(K = K, coords=coords,  row_splits=row_splits, tf_compatible=False,max_radius=radius)
    return sel, row_splits, hier, coords


neighbour_idxs, row_splits, hier, coords = createData(nvert)
#print(neighbour_idxs,'\n', row_splits, '\n',hier)
hierarchy_idxs = []
for i in range(len(row_splits.numpy())-1):
    a = tf.argsort(hier[row_splits[i]:row_splits[i+1]],axis=0)
    #print('a',a)
    hierarchy_idxs.append(a+row_splits[i])
    
hierarchy_idxs  = tf.concat(hierarchy_idxs, axis=0)
#print('hierarchy_idxs\n',hierarchy_idxs)
    
global_idxs = tf.range(nvert,dtype='int32')


rs,sel,gidx = lc(neighbour_idxs, hierarchy_idxs, global_idxs, row_splits)

#print('rs',rs.numpy())
#print('sel',sel.numpy())
#print('gidx',gidx.numpy())


def makecolours(asso):
    uasso = np.unique(asso)
    cols = asso.copy()
    for i in range(len(uasso)):
        cols[asso == uasso[i]] = i
    return np.array(cols,dtype='float')


for i in range(len(row_splits)-1):
    truthHitAssignementIdx = np.array(gidx[row_splits[i]:row_splits[i+1]].numpy())
    truthHitAssignementIdx = makecolours(truthHitAssignementIdx)+1.
    #predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
    #predBeta = np.ones_like(predBeta,dtype='float')-1e-2
    predCCoords = coords[row_splits[i]:row_splits[i+1]].numpy()
    
    print(truthHitAssignementIdx.shape)
    print(predCCoords.shape)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    plt.scatter(predCCoords[:,0],predCCoords[:,1],c=truthHitAssignementIdx,cmap='jet')
    plt.show()
    plt.close(fig)