
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL")
raise ImportError("MODULE",__name__,"will be removed")


import tensorflow as tf
import tensorflow.keras.backend as K



def huber(x, d):
    losssq  = x**2   
    absx = tf.abs(x)                
    losslin = d**2 + 2. * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)



def create_loss_dict(truth, pred):
    '''
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
    
    t_sigfrac = truth[:,:,0:-1]
    p_sigfrac = pred[:,:,0:tf.shape(truth)[2]-1]
    r_energy  = truth[:,:,-1]
    
    mask = tf.where(r_energy>0., tf.zeros_like(r_energy)+1,tf.zeros_like(r_energy))
    mask = tf.expand_dims(mask, axis=2)

    t_energy  = tf.expand_dims(r_energy, axis=2)*t_sigfrac
    t_sumenergy = tf.reduce_sum(t_energy, axis=1)
    
    t_n_rechits = tf.cast(tf.count_nonzero(r_energy, axis=1), dtype='float32')
    
    t_issc    = tf.reduce_sum(t_sigfrac,axis=-1)
    t_issc    = tf.where(t_issc>0., t_issc, tf.zeros_like(t_issc))
    t_issc    = tf.where(t_issc>1., tf.zeros_like(t_issc), t_issc)
    
    r_showers = pred[:,0,-3]
    t_showers = tf.cast(tf.count_nonzero(tf.reduce_sum(t_sigfrac, axis=1), axis=1), dtype='float32')
    
    
    return    {'mask'        : mask,
               't_sigfrac'   : t_sigfrac,
               'p_sigfrac'   : p_sigfrac,
               'r_energy'    : r_energy,
               't_energy'    : t_energy,
               't_sumenergy' : t_sumenergy,
               't_n_rechits' : t_n_rechits,
               'r_eta'       : tf.squeeze(pred[:,:,tf.shape(pred)[2]-2:tf.shape(pred)[2]-1],axis=-1),
               'r_phi'       : tf.squeeze(pred[:,:,tf.shape(pred)[2]-1:tf.shape(pred)[2]]  ,axis=-1),
               't_issc'      : t_issc,
               'r_showers'   : r_showers,
               't_showers'   : t_showers
               }
    




def energy_weighting(e, usesqrt, weightfactor=1.):
    e_in = e
    if usesqrt:
        e = tf.sqrt(tf.abs(e)+K.epsilon())
    if weightfactor<=0:
        e = tf.zeros_like(e)+1.
    return tf.where(e_in>0, e, tf.zeros_like(e))


def sortFractions(fracs, energies, to_sort):
    '''
    
    fracs      : B x V x Fracs
    energies   : B x V x 1
    to_sort    : B x V x 1
        
    '''
    frac_energies   = energies*tf.abs(fracs)
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

def makeDR2Matrix_SC_hits(sc_eta, sc_phi, hit_eta, hit_phi):
    '''
    Assumes sc_x to be of dimension B x N_SC 
    Assumes sc_x to be of dimension B x V x 1 
    
    Returns distances in B x V x N_SC
    '''
    dim    = tf.shape(sc_eta)[1]
    n_vert = tf.shape(hit_eta)[1]
    batch  = tf.shape(sc_eta)[0]
    
    sc_eta = tf.tile(tf.expand_dims(sc_eta, axis=1), [1,n_vert,1]) 
    sc_eta = tf.reshape(sc_eta, [batch,-1])
    sc_phi = tf.tile(tf.expand_dims(sc_phi, axis=1), [1,n_vert,1]) 
    sc_phi = tf.reshape(sc_phi, [batch,-1])
    
    hit_eta = tf.tile(hit_eta, [1,1,dim]) 
    hit_eta = tf.reshape(hit_eta, [batch,-1])
    hit_phi = tf.tile(hit_phi, [1,1,dim])
    hit_phi = tf.reshape(hit_phi, [batch,-1])
    
    dR2 = deltaR2(sc_eta, sc_phi, hit_eta, hit_phi)
    dR2 = tf.reshape(dR2, [batch,n_vert,dim])

    return dR2


def weightedCoordLoss(fracs, r_energy, coords):
    '''
    fracs:    B x V x F
    energies: B x V 
    coords:   B x V x C
    
    returns:  B x V
    '''
    from caloGraphNN import euclidean_squared
    
    mask = tf.where(r_energy>0, tf.zeros_like(r_energy)+1., tf.zeros_like(r_energy))
    #r_energy = tf.expand_dims(r_energy, axis=2)
     
    distances = euclidean_squared(coords, coords) # B x V x V
    fracdiff  = euclidean_squared(fracs, fracs)   # B x V x V
    #fracdiff =  tf.where(fracdiff<0.5, tf.zeros_like(fracdiff), fracdiff)
    #distances = tf.where(fracdiff<0.5, tf.zeros_like(distances), distances)
    
    diffsq = (distances - fracdiff)**2
    diffsq = tf.where(tf.logical_and(fracdiff>0.5, distances>0.5), tf.zeros_like(diffsq), diffsq)
    #if fracdiff large, distance should be large
    #fracdiff is max 1. distances are order 1
    weighted = mask * tf.reduce_sum(diffsq, axis = 2)
    return weighted
    
    
def weightedCenter(energies, fracs, var, isPhi=False):
    '''
    Inputs:
    energy: B x V x 1
    fracs:  B x V x F
    var:    B x V x 1
    
    output: B x F
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







