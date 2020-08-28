
from __future__ import print_function
import tensorflow as tf
import keras
import keras.backend as K
from index_dicts import create_index_dict, split_feat_pred, create_feature_dict
import time
from Loss_tools import huber
#factorise a bit


#
#
#
#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)
#
#

def create_loss_dict(truth, pred):
    return create_index_dict(truth, pred)

def killNan(a):
    return a


def printAsRagged(msg, S, C , row_splits):
    print(msg, tf.RaggedTensor.from_row_splits(values=tf.RaggedTensor.from_row_splits(values =   S, row_splits=C), 
                                                     row_splits=row_splits))
    


def get_one_over_sigma(beta, beta_min=1e-2):
    return (( 1. / (1. - beta + K.epsilon()) - 1.) + beta_min)
    



###### keras trick

pre_training_loss_counter=0
def pre_training_loss(truth, pred):
    feat,pred = split_feat_pred(pred)
    d = create_loss_dict(truth, pred)
    feat = create_feature_dict(feat)
    
    truthxy = tf.concat([d['truthHitAssignedX'],d['truthHitAssignedY']],axis=-1)
    
    #print('>>>>> >>> > > > > >', truthxy.shape, d['predCCoords'].shape, d['predXY'].shape, )
    
    diff = 10.*d['predCCoords'] - truthxy
    diff = d['truthNoNoise']*(0.1*d['predBeta'])*diff**2
    
    beta_med = (d['predBeta']-0.5)**2
    
    posl = (d['predXY'] - truthxy)**2 / 1000.
    
    loss = tf.reduce_mean(diff) + tf.reduce_mean(beta_med) + tf.reduce_mean(posl)
    tf.print('pretrain loss',loss, 'posl**2', tf.reduce_mean(posl))
    return loss
    
    
def batch_beta_weighted_truth_mean(b_l_in,b_istruth,b_beta_scaling):
    
    
    t_l_in = tf.reduce_sum(b_beta_scaling*b_istruth*b_l_in)#  1
    t_den =  tf.reduce_sum(b_istruth*b_beta_scaling) # 1
    t_den = tf.where(t_den==0, 1e-6, t_den)
    return t_l_in/t_den


#this needs to be per ragged batch! same for spectators   
#but we're in eager so whatever 
def beta_weighted_truth_mean(l_in, d, row_splits, beta_scaling, is_not_spectator=None):#l_in  V x 1
    
    batch_size = row_splits.shape[0] - 1
    out = tf.constant(0., tf.float32)
    istruth = d['truthNoNoise']
    if is_not_spectator is not None:
        istruth *= is_not_spectator
    
    for b in tf.range(batch_size):
        b_beta_scaling = beta_scaling[row_splits[b]:row_splits[b + 1]]
        b_istruth = istruth[row_splits[b]:row_splits[b + 1]]
        b_l_in = l_in[row_splits[b]:row_splits[b + 1]]
        if tf.reduce_max(b_istruth) == 0:
            continue
        out += batch_beta_weighted_truth_mean(b_l_in,b_istruth,b_beta_scaling)
    
    out /= float(batch_size)+1e-5
    return out

def batch_spectator_penalty(isspect,beta):
    out = tf.reduce_sum(isspect * beta )
    out /= tf.reduce_sum(isspect)+1e-3
    return out

def spectator_penalty(d,row_splits):
    
    batch_size = row_splits.shape[0] - 1
    out = tf.constant(0., tf.float32)
    isspect = d['truthIsSpectator']
    beta = d['predBeta']
    
    for b in tf.range(batch_size):
        out += batch_spectator_penalty(isspect[row_splits[b]:row_splits[b + 1]],
                                       beta[row_splits[b]:row_splits[b + 1]])
    
    out /= float(batch_size)+1e-3
    return out
    
def null_loss(truth, pred):
    return 0*tf.reduce_mean(pred)+0*tf.reduce_mean(truth)

from LayersRagged import RaggedConstructTensor
ragged_constructor=RaggedConstructTensor()

class _obj_cond_config(object):
    def __init__(self):
        self.energy_loss_weight = 1.
        self.use_energy_weights = False
        self.q_min = 0.5
        self.no_beta_norm = False
        self.potential_scaling = 1.
        self.repulsion_scaling = 1.
        self.s_b = 1.
        self.position_loss_weight = 1.
        self.timing_loss_weight = 1.
        self.use_spectators=True
        self.log_energy=False
        self.beta_loss_scale=1.
        self.use_average_cc_pos=False
        self.payload_rel_threshold=0.9
        self.rel_energy_mse=False
        self.smooth_rep_loss=False
        self.pre_train=False
        self.huber_energy_scale = 2.


config = _obj_cond_config()

g_time = time.time()

def full_obj_cond_loss(truth, pred_in, rowsplits):
    
    
    start_time = time.time()
    
    if truth.shape[0] is None: 
        return tf.constant(0., tf.float32)
    
    from object_condensation import indiv_object_condensation_loss_2

    rowsplits = tf.cast(rowsplits, tf.int64)#just for first loss evaluation from stupid keras
    
    feat,pred = split_feat_pred(pred_in)
    d = create_loss_dict(truth, pred)
    feat = create_feature_dict(feat)
    #print('feat',feat.shape)
    
    d['predBeta'] = tf.clip_by_value(d['predBeta'],1e-6,1.-1e-6)

    truthIsSpectator = d['truthIsSpectator'][:, 0]
    
    classes, row_splits = d['truthHitAssignementIdx'][...,0], rowsplits[ : rowsplits[-1,0],0]
    
    energyweights = d['truthHitAssignedEnergies']
    energyweights = tf.math.log(0.1 * energyweights + 1.)
    
    if not config.use_energy_weights:
        energyweights *= 0.
    energyweights += 1.
    
    #just to mitigate the biased sample
    energyweights = tf.where(d['truthHitAssignedEnergies']>10.,energyweights+0.1, energyweights*(d['truthHitAssignedEnergies']/10.+0.1))
    
    #also using log now, scale back in evaluation #
    den_offset = 1.
    if config.log_energy:
        raise ValueError("loss config log_energy is not supported anymore. Please use the 'ExpMinusOne' layer within the model instead to scale the output.")
    
    energy_diff = (d['predEnergy'] - d['truthHitAssignedEnergies']) 
    
    scaled_true_energy = d['truthHitAssignedEnergies']
    if config.rel_energy_mse:
        scaled_true_energy *= scaled_true_energy
    sqrt_true_en = tf.sqrt(scaled_true_energy)
    energy_loss = energy_diff/(sqrt_true_en+den_offset)
    
    if config.huber_energy_scale>0:
        huber_scale = config.huber_energy_scale * sqrt_true_en
        energy_loss = huber(energy_loss, huber_scale)
    else:
        energy_loss = energy_loss**2
    
    pos_offs = None
    payload_loss = None
    
    xdiff = d['predX']+feat['recHitX']  -   d['truthHitAssignedX']
    ydiff = d['predY']+feat['recHitY']  -   d['truthHitAssignedY']
    pos_offs = tf.reduce_sum(tf.concat( [xdiff,  ydiff],axis=-1)**2, axis=-1, keepdims=True)
    
    tdiff = d['predT']  -   d['truthHitAssignedT']
    #print("d['truthHitAssignedT']", tf.reduce_mean(d['truthHitAssignedT']), tdiff)
    tdiff = (1e6 * tdiff)**2
    # self.timing_loss_weight
    
    payload_loss = energyweights * tf.concat([config.energy_loss_weight * energy_loss ,
                          config.position_loss_weight * pos_offs,
                          config.timing_loss_weight * tdiff], axis=-1)
    
    
    
    attractive_loss, rep_loss, noise_loss, min_beta_loss, payload_loss_full  = indiv_object_condensation_loss_2(d['predCCoords'], #
                                                                                             d['predBeta'][...,0],  #remove last 1 dim
                                                                                             classes, 
                                                                                             row_splits,
                                                                                             truthIsSpectator,
                                                                                             Q_MIN=config.q_min, 
                                                                                             S_B=config.s_b,
                                                                                             energyweights=energyweights[...,0],
                                                                                             no_beta_norm=config.no_beta_norm,
                                                                                             payload_loss=payload_loss,
                                                                                             ignore_spectators=not config.use_spectators,
                                                                                             use_average_cc_pos=config.use_average_cc_pos,
                                                                                             payload_rel_threshold=config.payload_rel_threshold,
                                                                                             smooth_rep_loss=config.smooth_rep_loss)
    
    attractive_loss *= config.potential_scaling
    rep_loss *= config.potential_scaling * config.repulsion_scaling
    min_beta_loss *= config.beta_loss_scale
    
    spectator_beta_penalty = 0.
    if config.use_spectators:
        spectator_beta_penalty =  0.1 * spectator_penalty(d,row_splits)
        spectator_beta_penalty = tf.where(tf.math.is_nan(spectator_beta_penalty),0,spectator_beta_penalty)
    
    #attractive_loss = tf.where(tf.math.is_nan(attractive_loss),0,attractive_loss)
    #rep_loss = tf.where(tf.math.is_nan(rep_loss),0,rep_loss)
    #min_beta_loss = tf.where(tf.math.is_nan(min_beta_loss),0,min_beta_loss)
    #noise_loss = tf.where(tf.math.is_nan(noise_loss),0,noise_loss)
    #payload_loss_full = tf.where(tf.math.is_nan(payload_loss_full),0,payload_loss_full)
    
    
    
    energy_loss = payload_loss_full[0]
    pos_loss = payload_loss_full[1]
    time_loss = payload_loss_full[2]
    #energy_loss *= 0.0000001
    
    # neglect energy loss almost fully
    loss = attractive_loss + rep_loss +  min_beta_loss +  noise_loss  + energy_loss + time_loss + pos_loss + spectator_beta_penalty
    
    loss = tf.debugging.check_numerics(loss,"loss has nan")

    
    if config.pre_train:
         preloss = pre_training_loss(truth,pred_in)
         loss /= 10.
         loss += preloss
         
    print('loss',loss.numpy(), 
          'attractive_loss',attractive_loss.numpy(),
          'rep_loss', rep_loss.numpy(), 
          'min_beta_loss', min_beta_loss.numpy(), 
          'noise_loss' , noise_loss.numpy(),
          'energy_loss', energy_loss.numpy(), 
          'pos_loss', pos_loss.numpy(), 
          'time_loss', time_loss.numpy(), 
          'spectator_beta_penalty', spectator_beta_penalty)
    
    
    print('time for this loss eval',int((time.time()-start_time)*1000),'ms')
    global g_time
    print('time for total batch',int((time.time()-g_time)*1000),'ms')
    g_time=time.time()
    
    return loss
    
    
subloss_passed_tensor=None
def obj_cond_loss_rowsplits(truth, pred):
    global subloss_passed_tensor
    #print('>>>>>>>>>>> nbatch',truth.shape[0])
    if subloss_passed_tensor is not None: #passed_tensor is actual truth
        temptensor=subloss_passed_tensor
        subloss_passed_tensor=None
        #print('calling min_beta_loss_rowsplits', temptensor)
        return full_obj_cond_loss(temptensor, pred, truth)
        
    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)


def obj_cond_loss_truth(truth, pred):
    global subloss_passed_tensor
    if subloss_passed_tensor is not None: #passed_tensor is rs from other function
        temptensor=subloss_passed_tensor
        subloss_passed_tensor=None
        #print('calling min_beta_loss_truth', temptensor)
        return full_obj_cond_loss(truth, pred, temptensor)

    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)


def pretrain_obj_cond_loss_rowsplits(truth, pred):
    global subloss_passed_tensor
    #print('>>>>>>>>>>> nbatch',truth.shape[0])
    if subloss_passed_tensor is not None: #passed_tensor is actual truth
        temptensor=subloss_passed_tensor
        subloss_passed_tensor=None
        #print('calling min_beta_loss_rowsplits', temptensor)
        return pre_training_loss(temptensor, pred)
        
    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)


def pretrain_obj_cond_loss_truth(truth, pred):
    global subloss_passed_tensor
    if subloss_passed_tensor is not None: #passed_tensor is rs from other function
        temptensor=subloss_passed_tensor
        subloss_passed_tensor=None
        #print('calling min_beta_loss_truth', temptensor)
        return pre_training_loss(truth, pred)

    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)




    
