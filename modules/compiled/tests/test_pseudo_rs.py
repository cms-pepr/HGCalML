
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from condensate_op import BuildCondensates
from ragged_plotting_tools import make_cluster_coordinates_plot
from pseudo_rs_op import CreatePseudoRS
from LayersRagged import VertexScatterer, CondensateToPseudoRS, RaggedSumAndScatter


idxs = tf.constant([3,-1,3,2,-1,-30,-30,-30,6], dtype='int32')
data = tf.expand_dims(tf.constant([3.1,-1.1,3.2,2.1,-1.2,-30.1,-30.2,-30.3,6.1], dtype='float32'),axis=1) #tf.constant([3,3,3,2,-1,-1,6], dtype='int32')

sids, psrs, sdata,_ = CreatePseudoRS(idxs,data)
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
soft=False
radius=0.7

betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.01 , maxval=0.3+1e-4,seed=2)


ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,n_vert//2,n_vert], dtype='int32')





data = ccoords
sccoords, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=radius,  soft=soft, threshold=0.1)([data, ccoords, betas, row_splits])
#sids, psrs, sccoords = CreatePseudoRS(asso_idx,ccoords)
#print('data',sdata)
#print('sids',sids)
print('psrs',psrs)
sbeta = tf.gather_nd(betas,sids)

print('ccoords',ccoords.shape)
print('sccoords',sccoords.shape)
print('betas',betas.shape)
print('sbetas',betas.shape)

xsames = RaggedSumAndScatter()([sccoords, psrs, belongs_to_prs])
xsames = VertexScatterer()([xsames, sids, xsames])


backs = VertexScatterer()([sccoords, sids, sccoords]) #same shape
#tf.scatter_nd(sids, sccoords ,shape=tf.shape(sccoords))

def meancoords(ca,cb):
    ca = tf.expand_dims(ca, axis=2)
    cb = tf.expand_dims(cb, axis=2)
    m = tf.concat([ca,cb], axis=-1)
    return tf.reduce_mean(m,axis=2).numpy()
               

assert np.all(backs == ccoords)

def makecolours(asso):
    uasso = np.unique(asso)
    cols = asso.copy()
    for i in range(len(uasso)):
        cols[asso == uasso[i]] = i
    cols = np.array(cols,dtype='float') / float(len(uasso))
    return plt.get_cmap('viridis')(cols)

colors = makecolours(asso_idx.numpy())+1
scolors = tf.gather_nd(colors,sids).numpy()

for i in range(len(row_splits)-1):
    truthHitAssignementIdx = colors[row_splits[i]:row_splits[i+1]]

    predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
    #predBeta = np.ones_like(predBeta,dtype='float')-1e-2
    #predCCoords = tf.reduce_mean(tf.concat([xsames[row_splits[i]:row_splits[i+1]], .numpy()
    predCCoords = meancoords(xsames[row_splits[i]:row_splits[i+1]],ccoords[row_splits[i]:row_splits[i+1]])
    
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
    
    

