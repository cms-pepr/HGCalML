from condensate_op import BuildCondensates
import tensorflow as tf
from ragged_plotting_tools import make_cluster_coordinates_plot
import matplotlib.pyplot as plt
import numpy as np
import time

print('starting test')
n_vert=400000
n_ccoords=2
n_feat=3
soft=True
radius=0.7

betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.02 , maxval=0.99,seed=2)


ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,n_vert//2,n_vert], dtype='int32')

print('first call')
asso_idx, is_cpoint = BuildCondensates(ccoords=ccoords, betas=betas,  row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)

#print(ccoords)
#print(asso_idx)
#print(is_cpoint)
#exit()

print('starting taking time')
t0 = time.time()
for _ in range(0):
    asso_idx, is_cpoint = BuildCondensates(ccoords=ccoords, betas=betas,  row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)
totaltime = (time.time()-t0)/100.



print('op time', totaltime)

#exit()
#exit()
#print('betas',betas)
#print('ccoords',ccoords)
#print('summed_features',summed_features)
#print('asso_idx',asso_idx)
#print('n condensates', tf.unique(asso_idx))

def makecolours(asso):
    uasso = np.unique(asso)
    cols = asso.copy()
    for i in range(len(uasso)):
        cols[asso == uasso[i]] = i
    return np.array(cols,dtype='float')


for radius in [0.6, 1.3]:
    asso_idx, is_cpoint  = BuildCondensates(ccoords=ccoords, betas=betas, row_splits=row_splits, 
                                            radius=radius, min_beta=0.1, soft=soft)
    print('refs', np.unique(asso_idx))
    for i in range(len(row_splits)-1):
        
        truthHitAssignementIdx = np.array(asso_idx[row_splits[i]:row_splits[i+1]].numpy())
        
        truthHitAssignementIdx = makecolours(truthHitAssignementIdx)+1.
        
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
        #plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
        fig.clear()
        plt.close(fig)
        #exit()
    
    
    