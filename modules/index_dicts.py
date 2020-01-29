

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
    outdict['recHitEnergy']  = feat[:,0] 
    outdict['recHitEta']     = feat[:,1] 
    outdict['recHitRelPhi']  = feat[:,2] 
    outdict['recHitTheta']   = feat[:,3] 
    outdict['recHitR']       = feat[:,4] 
    outdict['recHitX']       = feat[:,5] 
    outdict['recHitY']       = feat[:,6] 
    outdict['recHitZ']       = feat[:,7] 
    outdict['recHitTime']    = feat[:,8] 
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
    outdict={}
    #make it all lists
    outdict['truthHitAssignementIdx']    =  truth[:,0]
    if usetf:
        outdict['truthIsNoise']              =  tf.where(truth[:,0] < -0.1, 
                                                     tf.zeros_like(truth[:,0])+1, 
                                                     tf.zeros_like(truth[:,0]))
    else:
        outdict['truthIsNoise']              =  np.where(truth[:,0] < -0.1, 
                                                     np.zeros_like(truth[:,0])+1, 
                                                     np.zeros_like(truth[:,0]))
    outdict['truthHitAssignedEnergies']  =  truth[:,1]
    outdict['truthHitAssignedEtas']      =  truth[:,8]
    outdict['truthHitAssignedPhis']      =  truth[:,9]
    
    outdict['predBeta']       = pred[:,0]
    outdict['predEnergy']     = pred[:,1]
    outdict['predEta']        = pred[:,2]
    outdict['predPhi']        = pred[:,3]
    outdict['predCCoords']    = pred[:,4:]

    return outdict
