from condensate_op import BuildCondensates
import tensorflow as tf
from ragged_plotting_tools import make_cluster_coordinates_plot
import matplotlib.pyplot as plt
import numpy as np
import time

print('starting test')
n_vert=4000
n_ccoords=2
n_feat=3
soft=False
radius=0.7

betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.01 , maxval=0.1+1e-3,seed=2)


ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,n_vert//2,n_vert], dtype='int32')

print('first call')
asso_idx, is_cpoint,n = BuildCondensates(ccoords=ccoords, betas=betas,  row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)

#print(ccoords)
#print(asso_idx)
#print(is_cpoint)
#exit()

print('starting taking time')
t0 = time.time()
for _ in range(0):
    asso_idx, is_cpoint,n = BuildCondensates(ccoords=ccoords, betas=betas,  row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)
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
    asso_idx, is_cpoint,n  = BuildCondensates(ccoords=ccoords, betas=betas, row_splits=row_splits, 
                                            radius=radius, min_beta=0.1, soft=soft)
    print('refs', np.unique(asso_idx))
    print('N',n)
    for i in range(len(row_splits)-1):
        
        truthHitAssignementIdx = np.array(asso_idx[row_splits[i]:row_splits[i+1]].numpy())
        ncond = n[i:i+1]
        print('N condensates', ncond.numpy())
        truthHitAssignementIdx = makecolours(truthHitAssignementIdx)+1.
        predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
        #predBeta = np.ones_like(predBeta,dtype='float')-1e-2
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
                                      cmap=None,
                                      noalpha=True
                                    )
        plt.show()
        #plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
        fig.clear()
        plt.close(fig)
        #exit()
    
    
    