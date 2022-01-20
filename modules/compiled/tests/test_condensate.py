from condensate_op import BuildCondensates
import tensorflow as tf
import matplotlib
from ragged_plotting_tools import make_cluster_coordinates_plot
import matplotlib.pyplot as plt
import numpy as np
import time

print('starting test')
n_vert=50000
n_ccoords=2
n_feat=3
soft=True
min_beta=0.02
radius=0.7

#betas = tf.random.uniform((n_vert,1), dtype='float32',minval=0.01 , maxval=0.1+1e-3,seed=2)

#dist = None #tf.random.uniform((n_vert,1), dtype='float32',minval=0.5 , maxval=1.5,seed=2)
ccoords = 3.*tf.random.uniform((n_vert,n_ccoords), dtype='float32',seed=1)
row_splits = tf.constant([0,n_vert//2,n_vert], dtype='int32')


def makebetas(ccoords_in,row_splits, ncenters=5):
    #overlay some gaussians plus random
    allbetas=[]
    alldistmod=[]
    for i in range(len(row_splits)-1):
        ccoords = ccoords_in[row_splits[i]:row_splits[i+1]]
        centers_x = tf.random.uniform((1,ncenters), dtype='float32',minval=tf.reduce_min(ccoords[:,0]) , 
                                      maxval=tf.reduce_max(ccoords[:,0]),seed=i+2)
        centers_y = tf.random.uniform((1,ncenters), dtype='float32',minval=tf.reduce_min(ccoords[:,1]) , 
                                      maxval=tf.reduce_max(ccoords[:,1]),seed=i+2)
        centers = tf.concat([centers_x,centers_y],axis=0)# 2 x 5
        
        distmod = tf.random.uniform(centers[0:1,:].shape, dtype='float32',minval=0.1 , maxval=.6,seed=2)
        #ccoords: V x 2
        distances = (tf.expand_dims(ccoords,axis=2)-tf.expand_dims(centers,axis=0))**2
        distances = tf.reduce_sum(distances,axis=1)#V x 5
        beta = tf.reduce_sum(tf.exp(-distances/(2.*distmod**2)),axis=1,keepdims=True) / ncenters#V x 1, max 1
        distmod = tf.reduce_sum(distmod*tf.exp(-distances/(2.*distmod)),axis=1,keepdims=True) / beta  / ncenters
        beta *= tf.random.uniform(beta.shape, dtype='float32',minval=0.9 , maxval=1.,seed=2)
        beta /= tf.reduce_max(beta)
        allbetas.append(beta)
        alldistmod.append(distmod)
    return tf.concat(allbetas,axis=0),tf.concat(alldistmod,axis=0)
    
    
betas,dist = makebetas(ccoords,row_splits,7)
#dist=None  

print('first call')
asso_idx, is_cpoint,n = BuildCondensates(ccoords=ccoords, betas=betas,  dist=dist,row_splits=row_splits, radius=radius, min_beta=0.1, soft=soft)

#print(ccoords)
#print(asso_idx)
#print(is_cpoint)
#exit()

print('starting taking time')
t0 = time.time()
for _ in range(20):
    asso_idx, is_cpoint,n = BuildCondensates(ccoords=ccoords, 
                                             betas=betas,  
                                             dist=dist,row_splits=row_splits, 
                                             radius=radius, min_beta=0.1, soft=soft)
totaltime = (time.time()-t0)/20.



print('op time', totaltime)

#exit()
#exit()
#print('betas',betas)
#print('ccoords',ccoords)
#print('summed_features',summed_features)
#print('asso_idx',asso_idx)
#print('n condensates', tf.unique(asso_idx))

def makecolours(ta,rdst=None):
    unta = np.unique(ta)
    unta = unta[unta>-0.1]
    if rdst is None:
        np.random.shuffle(unta)
    else:
        rdst.shuffle(unta)
    out = ta.copy()
    for i in range(len(unta)):
        out[ta ==unta[i]]=i
    out[out<0]=-1
    return np.array(out,dtype='float32')
    

#exit()
for radius in [1.]:#, 0.5, 0.8]:
    
    feats = tf.ones_like(betas)
    asso_idx, is_cpoint,n,sumfeat  = BuildCondensates(ccoords=ccoords, betas=betas, 
                                              dist = dist,
                                              tosum=feats,
                                              row_splits=row_splits, 
                                            radius=radius, min_beta=min_beta, soft=soft)
    
    print(sumfeat[is_cpoint>0], tf.reduce_sum(sumfeat[is_cpoint>0]))
    u,_,c = tf.unique_with_counts(asso_idx)
    argu = tf.argsort(u)
    u = np.array(u)
    c = np.array(c)
    argu = np.expand_dims(argu,axis=1)
    
    print(c[argu][u[argu]>=0],tf.reduce_sum(feats*betas))#,c[argu])
    
    
    print('refs', np.unique(asso_idx))
    print('N',n)
    for i in range(len(row_splits)-1):
        
        truthHitAssignementIdx = np.array(asso_idx[row_splits[i]:row_splits[i+1]].numpy())
        isc = np.array(is_cpoint[row_splits[i]:row_splits[i+1]].numpy())
        ncond = n[i:i+1]
        print('N condensates', ncond.numpy())
        truthHitAssignementIdx = makecolours(truthHitAssignementIdx)
        
        predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
        #predBeta = np.ones_like(predBeta,dtype='float')-1e-2
        predCCoords = ccoords[row_splits[i]:row_splits[i+1]].numpy()
        
        fig = plt.figure(figsize=(8,5))
        
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        make_cluster_coordinates_plot(plt, ax, 
                                      truthHitAssignementIdx, #[ V ] or [ V x 1 ]
                                      predBeta,               #[ V ] or [ V x 1 ]
                                      predCCoords,            #[ V x 2 ]
                                      identified_coords=None,
                                      beta_threshold=min_beta, 
                                      distance_threshold=radius,
                                      cmap=None,
                                      beta_plot_threshold=0.,
                                      noalpha=True
                                    )
        ax.scatter( predCCoords[isc>0][:,0], predCCoords[isc>0][:,1],
                     matplotlib.rcParams['lines.markersize'] ** 2,
                       c='#000000',  # rgba_cols[identified],
                       marker='X')
        
        ax2.scatter(predCCoords[:,0], predCCoords[:,1],
                    c=predBeta)
        #plt.show()
        plt.savefig("plot_"+str(i)+"_rad_"+str(radius)+".pdf")
        #fig.clear()
        plt.close(fig)
        #break
        #exit()
    
    
    