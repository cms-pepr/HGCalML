from condensate_op import BuildCondensates
import tensorflow as tf
from ragged_plotting_tools import make_cluster_coordinates_plot
import matplotlib.pyplot as plt
import numpy as np
import time


n_vert=300000
n_ccoords=2
n_feat=3

radius=0.5

betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.5 , maxval=0.99,seed=1)
features = tf.random.uniform((n_vert,n_feat), dtype='float32',seed=1)
ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,int(n_vert/3),int(n_vert*2/3),n_vert], dtype='int32')

summed_features, asso_idx = BuildCondensates(ccoords=ccoords, betas=betas, features=features, row_splits=row_splits, radius=radius, min_beta=0.1)
t0 = time.time()
for _ in range(20):
    summed_features, asso_idx = BuildCondensates(ccoords=ccoords, betas=betas, features=features, row_splits=row_splits, radius=radius, min_beta=0.1)
totaltime = (time.time()-t0)/20.


#print('betas',betas)
#print('features',features)
#print('ccoords',ccoords)
#print('summed_features',summed_features)
#print('asso_idx',asso_idx)
#print('n condensates', tf.unique(asso_idx))

print('op time', totaltime)


for i in range(len(row_splits)-1):
    
    truthHitAssignementIdx = np.array(asso_idx[row_splits[i]:row_splits[i+1]].numpy(),dtype='float')
    predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
    predCCoords = ccoords[row_splits[i]:row_splits[i+1]].numpy()
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    make_cluster_coordinates_plot(plt, ax, 
                                  truthHitAssignementIdx, #[ V ] or [ V x 1 ]
                                  predBeta,               #[ V ] or [ V x 1 ]
                                  predCCoords,            #[ V x 2 ]
                                  identified_coords=None,
                                  beta_threshold=0.1, 
                                  distance_threshold=radius,
                                  cmap=None
                                )
    plt.show()
    #plt.savefig("plot_"+str(i)+".pdf")
    fig.clear()
    plt.close(fig)
    #exit()
    
    
    