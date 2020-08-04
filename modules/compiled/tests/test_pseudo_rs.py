
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from condensate_op import BuildCondensates
from ragged_plotting_tools import make_cluster_coordinates_plot
from pseudo_rs_op import CreatePseudoRS

idxs = tf.constant([3,-1,3,2,-1,-30,-30,-30,6], dtype='int32')
data = tf.expand_dims(tf.constant([3.1,-1.1,3.2,2.1,-1.2,-30.1,-30.2,-30.3,6.1], dtype='float32'),axis=1) #tf.constant([3,3,3,2,-1,-1,6], dtype='int32')

sids, psrs, sdata = CreatePseudoRS(idxs,data)
print('data',data)
print('sids',sids)
print('psrs',psrs)


print('sorted data', sdata)
backs = tf.scatter_nd(sids, sdata ,shape=tf.shape(sdata))
print('re-unsorted data', backs)


print('starting full test')
n_vert=10000
n_ccoords=2
n_feat=3
soft=True
radius=0.7

betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.01 , maxval=0.3+1e-4,seed=2)


ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,n_vert//2,n_vert], dtype='int32')

print('first call')
asso_idx, is_cpoint,n = BuildCondensates(ccoords=ccoords, betas=betas,  row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)

print(tf.unique(asso_idx)[0])


sids, psrs, sccoords = CreatePseudoRS(asso_idx,ccoords)
#print('data',sdata)
#print('sids',sids)
print('psrs',psrs)
sbeta = tf.gather_nd(betas,sids)
sasso_idx = tf.gather_nd(asso_idx,sids)

print('ccoords',ccoords.shape)
print('sccoords',sccoords.shape)
print('betas',betas.shape)
print('sbetas',betas.shape)


backs = tf.scatter_nd(sids, sccoords ,shape=tf.shape(sccoords))
assert np.all(backs == ccoords)

def makecolours(asso):
    uasso = np.unique(asso)
    cols = asso.copy()
    for i in range(len(uasso)):
        cols[asso == uasso[i]] = i
    cols = np.array(cols,dtype='float') / float(len(uasso))
    return plt.get_cmap('prism')(cols)[:,:-1]

colors = makecolours(asso_idx.numpy())+1
scolors = tf.gather_nd(colors,sids).numpy()

for i in range(len(row_splits)-1):
    truthHitAssignementIdx = colors[row_splits[i]:row_splits[i+1]]

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
                                  noalpha=True,
                                  direct_color=True
                                )
    plt.show()
    #plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
    fig.clear()
    plt.close(fig)


for i in range(len(psrs)-1): 
    
    truthHitAssignementIdx = scolors[psrs[i]:psrs[i+1]]

    predBeta = sbeta[psrs[i]:psrs[i+1]].numpy()
    predCCoords = sccoords[psrs[i]:psrs[i+1]].numpy()
    
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
                                  noalpha=True,
                                  direct_color=True
                                )
    plt.show()
    #plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
    fig.clear()
    plt.close(fig)
    
    

