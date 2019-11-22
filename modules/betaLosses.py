

import tensorflow as tf
import keras.backend as K

#factorise a bit


#
#
#
#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)
#
#

def create_loss_dict(truth, pred):
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
            
    recHitEnergy,
            recHitEta   ,
            recHitRelPhi,
            recHitTheta ,
            recHitMag   ,
            recHitX     ,
            recHitY     ,
            recHitZ     ,
            recHitTime  
    
    '''
    outdict={}
    #make it all lists
    outdict['truthHitAssignementIdx']    =  truth[:,:,0]
    outdict['truthIsNoise']              =  tf.where(truth[:,:,0] < 0, 
                                                     tf.zeros_like(truth[:,:,0])+1, 
                                                     tf.zeros_like(truth[:,:,0]))
    outdict['truthHitAssignedEnergies']  =  truth[:,:,1]
    outdict['truthHitAssignedEtas']      =  truth[:,:,2]
    outdict['truthHitAssignedPhis']      =  truth[:,:,3]
    
    outdict['predBeta']       = pred[:,:,0]
    outdict['predEnergy']     = pred[:,:,1]
    outdict['predEta']        = pred[:,:,2]
    outdict['predPhi']        = pred[:,:,3]
    outdict['predCCoords']    = pred[:,:,4:]

    return outdict


def construct_ragged_matrices_indexing_tensors(row_splits):
    sub = row_splits[1:] - row_splits[:-1]
    a = sub**2
    b = tf.cumsum(a)
    b = tf.concat(([0], b), axis=0)
    ai = tf.ragged.row_splits_to_segment_ids(b)[..., tf.newaxis]
    b = tf.gather_nd(b, ai)
    c = tf.range(0, tf.reduce_sum(a)) - b
    vector_num_elements = tf.gather_nd(sub, ai)
    vector_range_within_batch_elements = c
    A = tf.cast(tf.math.floor(vector_range_within_batch_elements/vector_num_elements), tf.int64)[..., tf.newaxis]
    B = tf.math.floormod(vector_range_within_batch_elements,vector_num_elements)[..., tf.newaxis]


    C = tf.concat(([0], tf.cumsum(tf.gather_nd(sub, tf.ragged.row_splits_to_segment_ids(row_splits)[..., tf.newaxis]))), axis=0)

    '''
    A is like this:
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2
     3 3 3 3 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
     2 3 3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5 5]
     
    B is like this:
    [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 0 1 2 3 0 1 2 3
     0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4
     5 0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5]
     
    C is like this:
    [ 0  5 10 15 20 25 29 33 37 41 45 49 53 57 63 69 75 81 87 93]
    
    C is used for constructing tensors with two ragged dimensions
     
    '''

    return A,B, C




def get_one_over_sigma(beta, beta_min=1e-3):
    return 1. / (1. - beta + K.epsilon()) - 1. + beta_min
    

def get_arb_loss(ccoords, row_splits, beta, is_noise, beta_min=1e-3):
    #padded row splits
    row_splits = tf.reshape(x_row_splits, (-1,))
    batch_size_plus_1 = tf.cast(row_splits[-1], tf.int32)
    row_splits = tf.slice(row_splits, [0], batch_size_plus_1[..., tf.newaxis])
    
    
    sigma = 1. / get_one_over_sigma(beta, betamin)
    
    
    A,B,C = construct_ragged_matrices_indexing_tensors(row_splits)


    # Jan's losses
    # S is given (I am just setting it to all ones)
    d_square = tf.reduce_sum((tf.gather_nd(ccoords, A) - tf.gather_nd(ccoords, B))**2, axis=-1)


    N_minus_N_noise = tf.RaggedTensor.from_row_splits(values=(1-is_noise), row_splits=row_splits)
    N_minus_N_noise = tf.reduce_sum(N_minus_N_noise, axis=-1) # seems wrong? reduce sum, also axis?

    # This is given sorry it was easy to make a dummy version over here
    S = tf.ones_like(d_square)

    collected_sigma_i = tf.gather_nd(sigma, A)
    collected_sigma_j = tf.gather_nd(sigma, B)

    attractive_loss = (1-tf.gather_nd(is_noise, A))* (1-tf.gather_nd(is_noise, B))* (S*d_square)/(collected_sigma_i*collected_sigma_j)
    attractive_loss = tf.RaggedTensor.from_row_splits(values=attractive_loss, row_splits=C)
    attractive_loss = tf.RaggedTensor.from_row_splits(values=attractive_loss, row_splits=row_splits)
    attractive_loss = tf.reduce_sum(attractive_loss, axis=[1,2])
    # Normalize
    attractive_loss = attractive_loss / (N_minus_N_noise**2)
    # Mean over the batch dimension
    attractive_loss = tf.reduce_mean(attractive_loss)



    rep_loss = (1-tf.gather_nd(is_noise, A))* (1-tf.gather_nd(is_noise, B))* ((1-S))/(collected_sigma_i*collected_sigma_j*d_square + 1/0.0001)
    rep_loss = tf.RaggedTensor.from_row_splits(values=rep_loss, row_splits=C)
    rep_loss = tf.RaggedTensor.from_row_splits(values=rep_loss, row_splits=row_splits)
    rep_loss = tf.reduce_sum(rep_loss, axis=[1,2])
    # Normalize
    rep_loss = rep_loss / (N_minus_N_noise**2)
    # Mean over the batch dimension
    rep_loss = tf.reduce_mean(rep_loss)



    # It's a ragged tensor with two ragged axes
    min_beta_loss = tf.RaggedTensor.from_row_splits(values=(1.-S)*1e5 + (S*(1/collected_sigma_j)), row_splits=C)
    min_beta_loss = tf.RaggedTensor.from_row_splits(values=min_beta_loss, row_splits=row_splits)
    min_beta_loss = tf.reduce_min(min_beta_loss, axis=2)
    min_beta_loss = tf.reduce_sum(min_beta_loss, axis=1)
    # Normalize
    min_beta_loss = min_beta_loss / N_minus_N_noise
    # Mean over the batch dimension
    min_beta_loss = tf.reduce_mean(min_beta_loss)


    return attractive_loss, rep_loss, min_beta_loss


###### keras trick


def pre_training_loss(truth, pred):
    d = create_loss_dict(truth, pred)
    pos_loss = tf.reduce_mean((\
        (d['predEta'] - d['truthHitAssignedEtas'])**2 + \
        (d['predPhi'] - d['truthHitAssignedPhis'])**2 ))
    return pos_loss
    
    
    
def null_loss(truth, pred):
    return 0*tf.reduce_mean(pred)+0*tf.reduce_mean(truth)

def full_min_beta_loss(truth, pred, rowsplits):
    
    beta_min = 1e-3
    
    d = create_loss_dict(truth, pred)
    attractive_loss, rep_loss, min_beta_loss= get_arb_loss(d['predCCoords'], 
                                                           rowsplits, 
                                                           d['predBeta'], 
                                                           d['truthIsNoise'], 
                                                           beta_min=beta_min)
    
    onedivsigma = get_one_over_sigma(d['predBeta'],beta_min)
    noise_loss  = tf.reduce_mean((d['truthIsNoise']*d['predBeta'])**2)
    
    energy_loss = tf.reduce_mean(onedivsigma*(d['predEnergy'] - d['truthHitAssignedEnergies'])**2/(d['truthHitAssignedEnergies']+0.1))
    pos_loss = tf.reduce_mean(
        onedivsigma*(\
        (d['predEta'] - d['truthHitAssignedEtas'])**2 + \
        (d['predPhi'] - d['truthHitAssignedPhis'])**2 ))
    
    loss = attractive_loss + rep_loss + 100.* min_beta_loss + 100*noise_loss + energy_loss + pos_loss
    loss = tf.Print(loss,[loss,
                          attractive_loss,
                          rep_loss,
                          100.* min_beta_loss,
                          100*noise_loss,
                          energy_loss,
                          pos_loss], 'attractive_loss + rep_loss + 100.* min_beta_loss + 100*noise_loss + energy_loss + pos_loss ')
    return loss
    
    
subloss_passed_tensor=None
def min_beta_loss_rowsplits(truth, pred):
    global subloss_passed_tensor
    if subloss_passed_tensor is not None: #passed_tensor is actual truth
        return full_min_beta_loss(subloss_passed_tensor, pred, truth)
    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)


def min_beta_loss_truth(truth, pred):
    global subloss_passed_tensor
    if subloss_passed_tensor is not None: #passed_tensor is rs from other function
        return full_min_beta_loss(truth, pred, subloss_passed_tensor)
    subloss_passed_tensor = truth #=rs
    return  0.*tf.reduce_mean(pred)





    