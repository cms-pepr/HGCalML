
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL")
raise ImportError("MODULE",__name__,"will be removed")

'''
This file should contain the actual loss implementations.
Losses.py should only contain the dictionary of losses, or very simple one-line calls to
functions defined in this file here

All energy weighting should be done with sqrt(E) and must use the function 'energy_weighting'
which makes sure that: 
a) tf.sqrt() has a gradient (adding epsilon)
b) the energy remains 0 if the input was zero

'''

import tensorflow as tf
import keras
import keras.backend as K
from Loss_tools import create_loss_dict, energy_weighting, sortFractions

'''
    Just for reference, create_loss_dict  
    outputs:
    mask                 : B x V x 1
    t_sigfrac, p_sigfrac : B x V x Fracs
    r_energy             : B x V 
    t_energy             : B x V x Fracs
    t_sumenergy          : B x Fracs
    t_n_rechits          : B 
    r_eta                : B x V 
    r_phi                : B x V 
    t_issc               : B x V
    r_showers            : B 
    t_showers            : B
'''
    
    

def n_shower_loss(truth, pred):
    ldict = create_loss_dict(truth, pred)
    loss = (ldict['t_showers']-ldict['r_showers'])**2
    return loss


def good_frac_range_loss(truth, pred):
    ldict = create_loss_dict(truth, pred)
    t_sigfrac   = ldict['t_sigfrac']
    p_sigfrac   = ldict['p_sigfrac']
    rangeloss   =  tf.where(p_sigfrac < -0.5, (p_sigfrac + 0.5)**2, tf.zeros_like(p_sigfrac))
    rangeloss   += tf.where(p_sigfrac >  1.5, (p_sigfrac - 1.5)**2, tf.zeros_like(p_sigfrac))
    rangeloss   = tf.reduce_mean(tf.reduce_mean(rangeloss, axis=2), axis=1)
    return rangeloss



def good_frac_sum_loss(truth, pred):
    ldict = create_loss_dict(truth, pred)
    t_sigfrac   = ldict['t_sigfrac']
    p_sigfrac   = ldict['p_sigfrac']
    diffsq = energy_weighting(ldict['r_energy'], True) * (tf.reduce_sum(t_sigfrac, axis=-1) - tf.reduce_sum(p_sigfrac, axis=-1))**2
    diffsq = tf.reduce_mean(diffsq, axis=1)
    return diffsq
    
    
    
def _frac_loss(truth, pred, sort_pred):
    ldict = create_loss_dict(truth, pred)
    t_sigfrac   = ldict['t_sigfrac']
    r_energy    = ldict['r_energy']
    t_sigfrac   = ldict['t_sigfrac']
    p_sigfrac   = ldict['p_sigfrac']
    t_sumenergy = ldict['t_sumenergy']
    
    t_sigfrac   = sortFractions(t_sigfrac, 
                                tf.expand_dims(r_energy,axis=2), 
                                tf.expand_dims(ldict['r_eta'],axis=2))
    
    if sort_pred:
        p_sigfrac   = sortFractions(p_sigfrac, 
                                    tf.expand_dims(r_energy,axis=2), 
                                    tf.expand_dims(ldict['r_eta'],axis=2))
        
    r_energy =  energy_weighting(r_energy, True)
    t_sumenergy = tf.sqrt(ldict['t_sumenergy']+K.epsilon())
    
    diffsq   = tf.expand_dims(r_energy,axis=2)*(t_sigfrac - p_sigfrac)**2 #B x V x F
    
    loss = tf.reduce_mean(tf.reduce_mean(diffsq,axis=1), axis=1) # / (ldict['t_n_rechits']+K.epsilon())
    loss = tf.where(ldict['t_n_rechits']<1., tf.zeros_like(loss),loss)
    
    return loss

def frac_loss(truth, pred):
    return _frac_loss(truth, pred, sort_pred=False)
def frac_loss_sort_pred(truth, pred):
	return _frac_loss(truth, pred, sort_pred=True)


def modular_loss(D, MSE=False, ORTHO=False, ANGLE=False, NORM=False, COS=False):

    def loss_fn(truth, prediction):
        prediction= tf.reshape(prediction, shape=(-1, D, D))
        # truth = tf.reshape(truth, shape=(-1, D,D))
        # loss = tf.Variable(initial_value=0.0, name='loss')
        loss = 0.0
        # print("Truth: ", tf.reduce_mean(truth))
        # print("Prediction: ",tf.reduce_mean(prediction))
        
        if MSE: 
            # loss.assign_add(tf.reduce_mean((truth - prediction) * (truth - prediction)))
            loss += tf.reduce_mean((truth - prediction) * (truth - prediction))
            
        if ORTHO:
            prod = tf.matmul(prediction, prediction, transpose_a=False, transpose_b=True)
            offd = (tf.reduce_sum(prod, axis=[1,2]) - tf.linalg.trace(prod)) / D**2
            # loss.assign_add(tf.reduce_mean(offd))
            loss += tf.reduce_mean(tf.math.abs(offd))
            
        if ANGLE or NORM:
            if ANGLE:
                prod = tf.einsum('ijk,ijk->ik', truth, prediction)
            tn = tf.norm(truth, axis=1)
            pn = tf.norm(prediction, axis=1)
            # print("tn: ", tf.reduce_mean(tn))
            # print("tn: ", tn.shape)
            # print("pn: ", pn.shape)
            if NORM:
                # loss.assign_add(tf.reduce_mean(tf.math.abs(nt-np)))
                loss += tf.reduce_mean(tf.math.abs(tn-pn))
            if ANGLE:
                cos = tf.math.abs(prod / (tn * pn + 1e-5))
                # loss.assign_add(tf.reduce_mean(1 - cos))
                if COS:
                    loss += tf.reduce_mean(1 - cos)
                else:
                    sin = tf.math.abs(1 - tf.math.square(cos))
                    loss += tf.reduce_mean(sin)
        
        return loss
    
    return loss_fn    
