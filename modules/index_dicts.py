

import tensorflow as tf
import numpy as np


def create_feature_dict(feat):
    '''
    recHitEnergy,
    recHitEta   ,
    recHitRelPhi,
    recHitTheta ,
    recHitR   ,
    recHitX     ,
    recHitY     ,
    recHitZ     ,
    recHitTime  
    '''
    outdict={}
    outdict['recHitEnergy']  = feat[:,0:1] 
    outdict['recHitEta']     = feat[:,1:2] 
    outdict['recHitRelPhi']  = feat[:,2:3] 
    outdict['recHitTheta']   = feat[:,3:4] 
    outdict['recHitR']       = feat[:,4:5] 
    outdict['recHitX']       = feat[:,5:6] 
    outdict['recHitY']       = feat[:,6:7] 
    outdict['recHitZ']       = feat[:,7:8] 
    outdict['recHitTime']    = feat[:,8:9] 
    return outdict
    

def create_truth_dict(truth, usetf=False):
    outdict={}
    outdict['truthHitAssignementIdx']    =  truth[:,0:1]
    if usetf:
        outdict['truthIsNoise']              =  tf.where(truth[:,0:1] < -0.1, 
                                                     tf.zeros_like(truth[:,0:1])+1, 
                                                     tf.zeros_like(truth[:,0:1]))
        
    else:
        outdict['truthIsNoise']              =  np.where(truth[:,0:1] < -0.1, 
                                                     np.zeros_like(truth[:,0:1])+1, 
                                                     np.zeros_like(truth[:,0:1]))
    outdict['truthNoNoise'] = 1. - outdict['truthIsNoise']
    outdict['truthHitAssignedEnergies']  =  truth[:,1:2]
    outdict['truthHitAssignedEtas']      =  truth[:,8:9]
    outdict['truthHitAssignedPhis']      =  truth[:,9:10]

    # New
    outdict['truthRechitsSum']      =  truth[:,16:17]
    outdict['truthRealEnergy']      =  truth[:,15:16]
    outdict['truthIsSpectator']      =  truth[:,14:15]

    return outdict



def create_index_dict(truth, pred, usetf=True):
    '''
    input features as
    B x V x F
    with F = colours
    
    truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    
    np.array(truthHitAssignementIdx, dtype='float32')   , # 1
            truthHitAssignedEnergies ,
            truthHitAssignedEtas     ,
            truthHitAssignedPhis 
            
    
    '''
    xyzt=True
    
    outdict={}
    #make it all lists
    outdict['truthHitAssignementIdx']    =  truth[:,0:1]
    if usetf:
        outdict['truthIsNoise']              =  tf.where(truth[:,0:1] < -0.1, 
                                                     tf.zeros_like(truth[:,0:1])+1, 
                                                     tf.zeros_like(truth[:,0:1]))
        
    else:
        outdict['truthIsNoise']              =  np.where(truth[:,0:1] < -0.1, 
                                                     np.zeros_like(truth[:,0:1])+1, 
                                                     np.zeros_like(truth[:,0:1]))
    outdict['truthNoNoise'] = 1. - outdict['truthIsNoise']
    outdict['truthHitAssignedEnergies']  =  truth[:,1:2]
    outdict['truthHitAssignedEtas']      =  truth[:,8:9]
    outdict['truthHitAssignedPhis']      =  truth[:,9:10]
    
    outdict['truthHitAssignedX']      =  truth[:,2:3]
    outdict['truthHitAssignedY']      =  truth[:,3:4]
    outdict['truthHitAssignedZ']      =  truth[:,4:5]
    outdict['truthHitAssignedT']      =  truth[:,10:11]

    # New
    outdict['truthRechitsSum']      =  truth[:,16:17]
    outdict['truthRealEnergy']      =  truth[:,15:16]
    outdict['truthIsSpectator']      =  truth[:,14:15]

    outdict['predBeta']       = pred[:,0:1]
    outdict['predEnergy']     = pred[:,1:2]
    
    outdict['predEta']        = pred[:,2:3]
    outdict['predPhi']        = pred[:,3:4]
    outdict['predCCoords']    = pred[:,4:6]
    
    outdict['predAdditional'] = pred[:,6:]
    
    if xyzt:
        outdict['predX']        = pred[:,2:3]
        outdict['predY']        = pred[:,3:4]
        outdict['predT']        = pred[:,4:5]
        outdict['predCCoords']    = pred[:,5:7]
        outdict['predAdditional'] = pred[:,7:]
        

    

    return outdict




feature_length=9
pred_length=6


def split_feat_pred(pred):
    '''
    returns features, prediction
    '''
    return pred[...,:feature_length], pred[...,feature_length:]
    
    
    

