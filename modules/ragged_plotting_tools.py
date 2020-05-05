
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from plotting_tools import base_plotter, plotter_3d
from index_dicts import create_index_dict
from numba import jit
import math

'''
Everything here assumes non flattened format:

B x V x F

'''
# tools before making the ccoords plot working on all events
@jit(nopython=True)        
def c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold):
    

    for e in range(len(betasel)):
        selected = []
        for si in range(len(sorting[e])):
            i = sorting[e][si]
            use=True
            for s in selected:
                distance = math.sqrt( (s[0]-ccoords[e][i][0])**2 +  (s[1]-ccoords[e][i][1])**2 )
                if distance  < distance_threshold:
                    use=False
                    break
            if not use:
                betasel[e][i] = False
                continue
            else:
                selected.append(ccoords[e][i])
             
    return betasel
    
def collectoverthresholds(data, 
                          beta_threshold, distance_threshold):
    
    betas   = np.reshape(data['predBeta'], [data['predBeta'].shape[0], -1])
    ccoords = np.reshape(data['predCCoords'], [data['predCCoords'].shape[0], -1, data['predCCoords'].shape[2]])
    
    sorting = np.argsort(-betas, axis=1)
    
    betasel = betas > beta_threshold
    
    bsel =  c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold)
    
    
    return np.reshape(bsel , [data['predBeta'].shape[0], data['predBeta'].shape[1], data['predBeta'].shape[2]])

#alredy selected for one event here!


def selectEvent(rs, feat, truth, event):
    rs = np.array(rs , dtype='int')
    rs = rs[:rs[-1]]

    feat = feat[rs[event]:rs[event+1],...]
  
    return feat, truth[rs[event]:rs[event+1],...]

def createRandomizedColors(basemap,seed=0):
    cmap = plt.get_cmap(basemap)
    vals = np.linspace(0,1,256)
    np.random.seed(seed)
    np.random.shuffle(vals)
    return plt.cm.colors.ListedColormap(cmap(vals))


def make_cluster_coordinates_plot(plt, ax, 
                                  truthHitAssignementIdx, #[ V ] or [ V x 1 ]
                                  predBeta,               #[ V ] or [ V x 1 ]
                                  predCCoords,            #[ V x 2 ]
                                  beta_threshold=0.2, distance_threshold=0.8,
                                  cmap=None
                                ):
    
    #data = create_index_dict(truth,pred,usetf=False)
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(predBeta.shape)>1:
        predBeta = np.array(predBeta[:,0])
    
    if np.max(predBeta)>1.:
        raise ValueError("make_cluster_coordinates_plot: at least one beta value is above 1. Check your model!")
    
    ax.set_aspect(aspect=1.)
    #print(truthHitAssignementIdx)
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.98,0.98,0.98]
    #print(rgbcolor)
    #print(rgbcolor.shape)
    alphas = predBeta
    alphas[alphas<0.01] = 0.01
    alphas = np.expand_dims(alphas, axis=1)
    
    rgba_cols = np.concatenate([rgbcolor,alphas],axis=-1)
    rgb_cols = np.concatenate([rgbcolor,np.zeros_like(alphas+1.)],axis=-1)
    
    sorting = np.reshape(np.argsort(alphas, axis=0), [-1])
    
    ax.scatter(predCCoords[:,0][sorting],
              predCCoords[:,1][sorting],
              s=.25*matplotlib.rcParams['lines.markersize'] ** 2,
              c=rgba_cols[sorting])
    
    
    if beta_threshold < 0. or beta_threshold > 1 or distance_threshold<0:
        return
    
    data = {'predBeta': np.expand_dims(np.expand_dims(predBeta,axis=-1),axis=0),
            'predCCoords': np.expand_dims(predCCoords,axis=0)}
    
    
    #run the inference part
    identified = collectoverthresholds(data,beta_threshold,distance_threshold)[0,:,0] #V
    
    
    ax.scatter(predCCoords[:,0][identified],
              predCCoords[:,1][identified],
              s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
              c='#000000',#rgba_cols[identified],
              marker='+')
    
    return identified
    

def make_original_truth_shower_plot(plt, ax,
                                    truthHitAssignementIdx,                      
                                    recHitEnergy, 
                                    recHitX,
                                    recHitY,
                                    recHitZ,
                                    cmap=None):
    
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(recHitEnergy.shape)>1:
        recHitEnergy = np.array(recHitEnergy[:,0])
    if len(recHitX.shape)>1:
        recHitX = np.array(recHitX[:,0])
    if len(recHitY.shape)>1:
        recHitY = np.array(recHitY[:,0])
    if len(recHitZ.shape)>1:
        recHitZ = np.array(recHitZ[:,0])
        
        
    pl = plotter_3d(output_file="/tmp/plot")#will be ignored
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    rgbcolor[truthHitAssignementIdx<0]=[0.92,0.92,0.92]
    
    pl.set_data(x = recHitX , y=recHitY   , z=recHitZ, e=recHitEnergy , c =rgbcolor)
    pl.marker_scale=2.
    pl.plot3d(ax=ax)
    
    
def make_eta_phi_projection_truth_plot(plt, ax,
                                    truthHitAssignementIdx,                      
                                    recHitEnergy, 
                                    recHitEta,
                                    recHitPhi,
                                    predEta,
                                    predPhi,
                                    truthEta,
                                    truthPhi,
                                    truthEnergy,
                                    predBeta,               #[ V ] or [ V x 1 ]
                                    predCCoords,            #[ V x 2 ]
                                    beta_threshold=0.2, distance_threshold=0.8,
                                    cmap=None,
                                    identified=None):
    
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(recHitEnergy.shape)>1:
        recHitEnergy = np.array(recHitEnergy[:,0])
    if len(recHitEta.shape)>1:
        recHitEta = np.array(recHitEta[:,0])
    if len(recHitPhi.shape)>1:
        recHitPhi = np.array(recHitPhi[:,0])
    if len(predEta.shape)>1:
        predEta = np.array(predEta[:,0])
    if len(predPhi.shape)>1:
        predPhi = np.array(predPhi[:,0])
    if len(truthEta.shape)>1:
        truthEta = np.array(truthEta[:,0])
    if len(truthPhi.shape)>1:
        truthPhi = np.array(truthPhi[:,0])
    if len(truthEnergy.shape)>1:
        truthEnergy = np.array(truthEnergy[:,0])
        
        
    if len(truthHitAssignementIdx.shape)>1:
        truthHitAssignementIdx = np.array(truthHitAssignementIdx[:,0])
    if len(predBeta.shape)>1:
        predBeta = np.array(predBeta[:,0])
        
    

    ax.set_aspect(aspect=1.)
    #print(truthHitAssignementIdx)
    if cmap is None:
        rgbcolor = plt.get_cmap('prism')((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]
    else:
        rgbcolor = cmap((truthHitAssignementIdx+1.)/(np.max(truthHitAssignementIdx)+1.))[:,:-1]

    size_scaling = np.log(recHitEnergy+1)+0.1
    size_scaling /=  np.max(size_scaling)

    ax.scatter(recHitPhi,
              recHitEta,
              s=.25*size_scaling,
              c=rgbcolor)
     
    _, truth_idxs = np.unique(truthHitAssignementIdx,return_index=True)
    
    truth_size_scaling=np.log(truthEnergy[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0] +1.)+0.1
    truth_size_scaling /=  np.max(truth_size_scaling)
    
    ax.scatter(truthPhi[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
              truthEta[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
              s=100.*truth_size_scaling,
              c=rgbcolor[truth_idxs][truthHitAssignementIdx[truth_idxs] >= 0],
              marker='x')
    
    
    if beta_threshold < 0. or beta_threshold > 1 or distance_threshold<0:
        return
    
    data = {'predBeta': np.expand_dims(np.expand_dims(predBeta,axis=-1),axis=0),
            'predCCoords': np.expand_dims(predCCoords,axis=0)}
       
    #run the inference part
    if identified is None:
        identified = collectoverthresholds(data,beta_threshold,distance_threshold)[0,:,0] #V
    
    
    ax.scatter(predPhi[identified],
              predEta[identified],
              s=2.*matplotlib.rcParams['lines.markersize'] ** 2,
              c='#000000',#rgba_cols[identified],
              marker='+')
    
    
        