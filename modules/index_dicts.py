

import tensorflow as tf
import numpy as np



feature_length=9
pred_length=6
n_classes=6

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
    
    
    outdict['truthClasses']      =  truth[:,19:19+n_classes]

    return outdict



def create_index_dict(truth, pred, usetf=True, n_ccoords=2):
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


    outdict['ticlHitAssignementIdx']      =  truth[:,17:18]
    outdict['ticlHitAssignedEnergies']      =  truth[:,18:19]

    outdict['truthClasses']      =  truth[:,19:19+n_classes]
    #(None, 9) (None, 1) (None, 1) (None, 3) (None, 2)
    #print(raw_inputs.shape, beta.shape, energy.shape, xyt.shape, ccoords.shape)
    outdict['predBeta']       = pred[:,0:1]
    outdict['predEnergy']     = pred[:,1:2]
    
    
    outdict['predXY']        = pred[:,2:4]
    
    outdict['predX']        = pred[:,2:3]
    outdict['predY']        = pred[:,3:4]
    outdict['predT']        = pred[:,4:5]
    outdict['predCCoords']    = pred[:,5:5+n_ccoords]
    outdict['predClasses'] = pred[:,5+n_ccoords:5+n_ccoords+n_classes]
        
        

    return outdict




def split_feat_pred(pred):
    '''
    returns features, prediction
    '''
    return pred[...,:feature_length], pred[...,feature_length:]
    
    
    



def create_ragged_cal_feature_dict(feat, truth):
    outdict = {}

    outdict['recHitEnergy'] = feat[:, 0]
    outdict['recHitEta'] = feat[:, 1]
    outdict['recHitRelPhi'] = feat[:, 2]
    outdict['recHitTheta'] = feat[:, 3]
    outdict['recHitR'] = feat[:, 4]
    outdict['recHitX'] = feat[:, 5]
    outdict['recHitY'] = feat[:, 6]
    outdict['recHitZ'] = feat[:, 7]
    outdict['recHitTime'] = feat[:, 8]

    outdict['truthHitAssignementIdx'] = truth[:, 0]
    outdict['truthIsNoise'] = truth[:, 1]
    outdict['truthNoNoise'] = truth[:, 2]
    outdict['truthHitAssignedEnergies'] = truth[:, 3]
    outdict['truthHitAssignedX'] = truth[:, 4]
    outdict['truthHitAssignedY'] = truth[:, 5]
    outdict['truthHitAssignedZ'] = truth[:, 6]
    outdict['truthHitAssignedT'] = truth[:, 7]
    outdict['truthHitAssignedEtas'] = truth[:, 8]
    outdict['truthHitAssignedPhis'] = truth[:, 9]
    outdict['truthRechitsSum'] = truth[:, 10]
    outdict['truthRealEnergy'] = truth[:, 11]
    outdict['truthIsSpectator'] = truth[:, 12]
    outdict['truthClasses'] = truth[:, 13]
    outdict['ticlHitAssignementIdx'] = truth[:, 14]
    outdict['ticlHitAssignedEnergies'] = truth[:, 15]

    return outdict
