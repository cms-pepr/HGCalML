import tensorflow as tf
import numpy as np
import time
from select_knn_op import SelectKnn
from local_cluster_op import LocalCluster as lc
import matplotlib.pyplot as plt
from ragged_plotting_tools import make_cluster_coordinates_plot
from LayersRagged import VertexScatterer
from sklearn.utils import shuffle
import os

def makecolours(asso):
    uasso = np.unique(asso)
    uasso = shuffle(uasso)
    cols = asso.copy()
    for i in range(len(uasso)):
        cols[asso == uasso[i]] = i
    return np.array(cols,dtype='float')

def plots(gidx, row_splits, glcoords, vidx, outdir, onlyFirst=False):
    os.system('mkdir -p '+outdir)
    for i in range(len(row_splits)-1):
        truthHitAssignementIdx = np.array(gidx[row_splits[i]:row_splits[i+1]])
        truthHitAssignementIdx = makecolours(truthHitAssignementIdx)+1.
        #predBeta = betas[row_splits[i]:row_splits[i+1]].numpy()
        #predBeta = np.ones_like(predBeta,dtype='float')-1e-2
        predCCoords = glcoords[row_splits[i]:row_splits[i+1]]
        
        print(truthHitAssignementIdx.shape)
        print(predCCoords.shape)
        
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        
        plt.scatter(predCCoords[:,0],predCCoords[:,1],c=truthHitAssignementIdx,cmap='jet')
        #plt.show()
        plt.savefig(outdir+'/0000'+str(vidx)+'.png')
        plt.close(fig)
        #if onlyFirst:
        #    break
        

radius=0.1
nvert=10000
K=1000
outdir='lc_cpu'

def applyClustering(K, incoords, hier, row_splits):
    
    hierarchy_idxs=[]
    for i in range(len(row_splits.numpy())-1):
        a = tf.argsort(hier[row_splits[i]:row_splits[i+1]],axis=0)
        hierarchy_idxs.append(a+row_splits[i])
    hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
    print('hierarchy_idxs',hierarchy_idxs.shape)
    print('incoords',incoords.shape)
    assert incoords.shape[0] == hierarchy_idxs.shape[0]
    neighs,_ = SelectKnn(K = K, coords=incoords,  row_splits=row_splits, tf_compatible=False,max_radius=radius)
    rs,sel,gscatidx = lc(neighs, hierarchy_idxs, row_splits)
    
    return rs,sel, gscatidx

def createData(nvert,ncoords=2, ncentres=4):
    
    
    coords = [] #tf.constant( np.random.rand(nvert) ,dtype='float32')
    hier=[]
    truth=[]
    row_splits = tf.constant( [0, nvert//2,  nvert] ,dtype='int32')
    for nvrs in [nvert//2,nvert-nvert//2]:
        for nv,i in zip([nvrs//3, nvrs//3, nvrs-2*(nvrs//3)], [0,1,2]):
            mean=None
            scale = 1.1 - 0.2*np.random.rand(2)
            
            if i==0:
                mean = np.array([0., 1.2])*scale
            if i==1:
                mean = np.array([1.2, 0.])*scale
            if i==2:
                mean = np.array([0., -1.5])*scale
            cov = np.array([[0.2, 0],
                   [0, 0.2]])*scale
            trgaus = np.array(np.random.multivariate_normal(mean, cov, nv))
            
            
            print(trgaus.shape)
            trgaus = np.array(trgaus)
            coords.append(trgaus[:,0:2])
            diff = np.expand_dims(np.array(mean), axis=0) - trgaus[:,0:2]
            hier.append(np.sqrt( diff[:,0]*diff[:,0] + diff[:,1]*diff[:,1] ))
            truth.append(np.zeros_like(diff[:,0])+i)
            
    coords = np.array(np.concatenate(coords, axis=0),dtype='float32')
    hier = 5.*np.array(np.concatenate(hier,axis=0),dtype='float32')
    #hier *= hier*(1.1 - 0.2*np.random.
    truth = np.array(np.concatenate(truth,axis=0),dtype='float32')
    glidxs = np.arange(nvert,dtype='int32')
    print(coords.shape)
    return row_splits, hier, coords, truth, glidxs


gl_row_splits, gl_hier, gl_coords, gl_truth, global_idxs = createData(nvert)

for a in [gl_row_splits, gl_hier, gl_coords, gl_truth, global_idxs]:
    print(a.shape)
#for i in range(len(row_splits)-1):
#    predCCoords = coords[row_splits[i]:row_splits[i+1]]
#    plt.scatter(predCCoords[:,0],predCCoords[:,1],s=hier[row_splits[i]:row_splits[i+1]],cmap='jet',
#                c=truth[row_splits[i]:row_splits[i+1]])
#    plt.show()

#print(neighbour_idxs,'\n', row_splits, '\n',hier)

hier = gl_hier
coords = gl_coords
row_splits = gl_row_splits
gidx = global_idxs
scatters=[]
for i in range(0):

    row_splits,sel,backscatter = applyClustering(K, coords, hier, row_splits)
    scatters.append(backscatter)
    print('coords pre sel ',coords.shape)
    print('sel',sel.shape)
    hier = tf.gather_nd(hier,sel)
    coords = tf.gather_nd(coords,sel)
    #this gives back the original indices after backgather
    gidx = tf.gather_nd(gidx,sel)
    coords /= 2. #shift stuff closer together
    print('row_splits',row_splits.numpy())
    print('coords',coords.shape)
    print('hier',hier.shape)
    print('backscatter',backscatter.shape)
    
    sel_gidx = gidx
    for k in range(len(scatters)):
        l = len(scatters) - k - 1
        print('scatters[l]',scatters[l].shape)
        sel_gidx = tf.gather_nd(sel_gidx, scatters[l] )
        print('scattered to', sel_gidx.shape)
    
    print(sel_gidx.numpy())
    plots(sel_gidx.numpy(), gl_row_splits, gl_coords, i, outdir,onlyFirst=True)
    if len(sel) < 4:
        break


#timing test
vs=[]
ts=[]
sortt=[]
K=10
radius=0.05

def getHierIdx(hier, row_splits):
    hierarchy_idxs=[]
    for i in range(len(row_splits.numpy())-1):
        a = tf.argsort(hier[row_splits[i]:row_splits[i+1]],axis=0)
        hierarchy_idxs.append(a+row_splits[i])
    return tf.concat(hierarchy_idxs,axis=0)
    
for v in [10000,50000,100000,400000]:
    row_splits, hier, coords, truth, glidxs = createData(v)
    neighs,_ = SelectKnn(K = K, coords=coords,  row_splits=row_splits, tf_compatible=False,max_radius=radius)
    
    #warmup
    hierarchy_idxs=getHierIdx(hier, row_splits)
    start = time.time()
    hierarchy_idxs=getHierIdx(hier, row_splits)
    sortt.append(time.time()-start)
    #warm up
    rs,sel,gscatidx = lc(neighs, hierarchy_idxs, row_splits)
    
    start = time.time()
    for _ in range(10):
        rs,sel,gscatidx = lc(neighs, hierarchy_idxs, row_splits)
    totime = (time.time()-start)/10.
    print(totime,'s for',v,'vertices')
    vs.append(v)
    ts.append(totime)
    
plt.plot(vs,ts)
plt.xlabel("# vertices")
plt.ylabel("t LC [s]")
plt.show()

plt.plot(vs,sortt)
plt.xlabel("# vertices")
plt.ylabel("t sort [s]")
plt.show()

#print('rs',rs.numpy())
#print('sel',sel.numpy())
#print('gidx',gidx.numpy())



        