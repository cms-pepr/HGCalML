
import tensorflow as tf
import keras
import keras.backend as K

def sortFractions(fracs, energies, to_sort):
    '''
    
    fracs      : B x V x Fracs
    energies   : B x V x 1
    to_sort    : B x V x 1
        
    '''
    frac_energies   = energies*fracs
    frac_sumenergy  = tf.reduce_sum(frac_energies, axis=1)
    
    #frac_energies    : B x V x Fracs
    #frac_sumenergy   : B x Fracs
    
    weighted_to_sort = tf.reduce_sum(frac_energies * to_sort, axis=1)/(frac_sumenergy+K.epsilon())
    
    #set the zero entries to something big to make them appear at the end of the list
    weighted_to_sort = tf.where(tf.abs(weighted_to_sort)>0., weighted_to_sort, tf.zeros_like(weighted_to_sort)+500.)
    
    ranked_to_sort, ranked_indices = tf.nn.top_k(-weighted_to_sort, tf.shape(fracs)[2])
    
    #ranked_indices = tf.Print(ranked_indices,[weighted_to_sort, ranked_to_sort],'weighted_to_sort, ranked_to_sort ', summarize=200)
    
    ranked_indices = tf.expand_dims(ranked_indices, axis=2)
    
    batch_range = tf.range(0, tf.shape(fracs)[0])
    batch_range = tf.expand_dims(batch_range, axis=1)
    batch_range = tf.expand_dims(batch_range, axis=1)
        
    batch_indices = tf.tile(batch_range, [1, tf.shape(fracs)[2], 1]) # B x Fracs x 1
    indices = tf.concat([batch_indices, ranked_indices], axis=-1) # B x Fracs x 2
    
    identity_matrix = tf.eye(tf.shape(fracs)[2]) #Fracs x Fracs 1 matrix
    identity_matrix = tf.expand_dims(identity_matrix, axis=0) # 1 x F x F
    identity_matrix = tf.tile(identity_matrix, [tf.shape(fracs)[0],1,1])  # B x F x F
    sorted_identity_matrix = tf.gather_nd(identity_matrix, indices) # B x F x F
    
    # sorted_identity_matrix : B x Fm x Fm
    # predicted_fracs        : B x V  x Ff
    
    # B x Fm x Fm --> B x V x Fm x Fm
    sorted_identity_matrix = tf.expand_dims(sorted_identity_matrix, axis=1)
    sorted_identity_matrix = tf.tile(sorted_identity_matrix, [1,tf.shape(fracs)[1],1,1])
    # B x V x Fm x Fm
    
    # predicted_fracs   : B x V  x Ff --> B x V x Ff x 1
    sorted_predicted_fractions = tf.expand_dims(fracs, axis=3)
    
    out = tf.squeeze(tf.matmul(sorted_identity_matrix, sorted_predicted_fractions), axis=-1)
    
    #out = tf.Print(out, [fracs[0,0,:], out[0,0,:]], 'in / out ', summarize=300)
    
    return out




def deltaPhi(a, b):
    import math as m
    diff = a - b
    diff = tf.where(diff >=  m.pi, diff - 2.*m.pi, diff)
    diff = tf.where(diff <  -m.pi, diff + 2.*m.pi, diff)
    return diff

def deltaR2(a_eta, a_phi, b_eta, b_phi):
    
    deta = a_eta - b_eta
    dphi = deltaPhi(a_phi, b_phi)
    
    return deta**2 + dphi**2

def makeDR2Matrix(a_eta, a_phi, b_eta, b_phi):
    '''
    Assumes a_x, b_x to be of dimension B x M 
    '''
    dim   = tf.shape(a_eta)[1]
    batch = tf.shape(a_eta)[0]
    
    a_eta = tf.tile(tf.expand_dims(a_eta, axis=2), [1,1,dim]) # B x M x M_same
    a_eta = tf.reshape(a_eta, [batch,-1])
    
    b_eta = tf.tile(tf.expand_dims(b_eta, axis=1), [1,dim,1]) # B x M_same x M
    b_eta = tf.reshape(b_eta, [batch,-1])
    
    a_phi = tf.tile(tf.expand_dims(a_phi, axis=2), [1,1,dim]) # B x M x M_same
    a_phi = tf.reshape(a_phi, [batch,-1])
    
    b_phi = tf.tile(tf.expand_dims(b_phi, axis=1), [1,dim,1]) # B x M_same x M
    b_phi = tf.reshape(b_phi, [batch,-1])
    
    dR2 = deltaR2(a_eta, a_phi, b_eta, b_phi)
    dR2 = tf.reshape(dR2, [batch,dim,dim])

    return dR2


    
def weightedCenter(energies, fracs, var, isPhi=False):
    '''
    Inputs:
    energy: B x V x 1
    fracs:  B x V x F
    var:    B x V x 1
    '''
    frac_energies   = energies*fracs
    frac_sumenergy  = tf.reduce_sum(frac_energies, axis=1)
    weighted = None
    if isPhi:
        reference = var[:,0:1,:]
        weighted = tf.reduce_sum(frac_energies * deltaPhi(var,reference), axis=1)/(frac_sumenergy+K.epsilon())
        weighted += tf.squeeze(reference, axis=1)
    else:
        weighted = tf.reduce_sum(frac_energies * var, axis=1)/(frac_sumenergy+K.epsilon())
    return weighted







