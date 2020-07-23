import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import time
from ragged_plotting_tools import make_cluster_coordinates_plot
import matplotlib.pyplot as plt

from latent_space_grid_op import LatentSpaceGrid

def createData(nvert,ncoords,nfeat):
    c = np.random.rand(nvert,ncoords)
    c[:,0]*=2
    x = tf.constant( c ,dtype='float32')
    rs = tf.constant( [0,int(nvert/2),nvert] ,dtype='int32')
    feat = tf.constant( np.random.rand(nvert,nfeat) ,dtype='float32')
    return x, rs, feat

x,rs,feat = createData(2000,2,3)
'''
 .Attr("size: float")
    .Attr("min_cells: int") //same for all dimensions
    .Input("coords: float32")
    .Input("min_coord: float32") //per dimension
    .Input("max_coord: float32")
    .Input("row_splits: int32")
    .Output("select_idxs: float32")
    .Output("pseudo_rowsplits: int32");
'''



idxs,psrs,ncc,backgather = LatentSpaceGrid(size=.1, 
                                       min_cells=1,
                                       coords = x,
                                       row_splits=rs )
t0 = time.time()
for i in range(10):
    idxs,psrs,ncc,backgather = LatentSpaceGrid(size=.4, 
                                       min_cells=4,
                                       coords = x,
                                       row_splits=rs )
tot_time = time.time()-t0
print('op time', tot_time/20)


#exit()

resorted = tf.gather_nd(feat, tf.expand_dims(idxs,axis=1))
pseudo_ragged = tf.RaggedTensor.from_row_splits(
      values=resorted,
      row_splits=psrs)

print(pseudo_ragged)

#pseudo_ragged = tf.concat([pseudo_ragged, tf.zeros([pseudo_ragged.shape[0], resorted.shape[1]], dtype='float32')],axis=1)


pseudo_ragged = tf.reduce_max(pseudo_ragged, axis=1)
pseudo_ragged = tf.where(tf.abs(pseudo_ragged)>1e10,0,pseudo_ragged)

print(pseudo_ragged)

back = tf.gather_nd(pseudo_ragged, tf.expand_dims(backgather,axis=1))



#some visualisation

row_splits=rs

for i in range(len(row_splits)-1):
        
    colors = backgather[row_splits[i]:row_splits[i+1]].numpy()
    #colors = back[row_splits[i]:row_splits[i+1]].numpy()
    
    truthHitAssignementIdx = np.array(colors,dtype='float')
    predBeta = np.ones_like(truthHitAssignementIdx)
    predCCoords = x[row_splits[i]:row_splits[i+1]].numpy()
    
    print('back',colors)
    print('back maxmin',np.max(colors), np.min(colors))
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    make_cluster_coordinates_plot(plt, ax, 
                                  truthHitAssignementIdx, #[ V ] or [ V x 1 ]
                                  predBeta,               #[ V ] or [ V x 1 ]
                                  predCCoords,            #[ V x 2 ]
                                  identified_coords=None,
                                  beta_threshold=0.1, 
                                  distance_threshold=1.,
                                  cmap=None
                                )
    plt.show()
    #plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
    #fig.clear()
    #plt.close(fig)
        #exit()













