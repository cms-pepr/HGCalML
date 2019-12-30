

import tensorflow as tf
import keras
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
    outdict['truthHitAssignementIdx']    =  truth[:,0]
    outdict['truthIsNoise']              =  tf.where(truth[:,0] < -0.1, 
                                                     tf.zeros_like(truth[:,0])+1, 
                                                     tf.zeros_like(truth[:,0]))
    outdict['truthHitAssignedEnergies']  =  truth[:,1]
    outdict['truthHitAssignedEtas']      =  truth[:,8]
    outdict['truthHitAssignedPhis']      =  truth[:,9]
    
    outdict['predBeta']       = pred[:,0]
    outdict['predEnergy']     = pred[:,1]
    outdict['predEta']        = pred[:,2]
    outdict['predPhi']        = pred[:,3]
    outdict['predCCoords']    = pred[:,4:]

    return outdict


def killNan(a):
    return a
    return tf.where(tf.is_nan(a), tf.zeros_like(a)+100., a)

def construct_ragged_matrices_indexing_tensors(row_splits):
    print('row_splits',row_splits)
    sub = row_splits[1:] - row_splits[:-1]
    a = sub**2
    b = tf.cumsum(a)
    b = tf.concat(([0], b), axis=0)
    ai = tf.ragged.row_splits_to_segment_ids(b)[..., tf.newaxis]
    b = tf.gather_nd(b, ai)
    c = tf.range(0, tf.reduce_sum(a)) - b
    vector_num_elements = tf.gather_nd(sub, ai)
    vector_range_within_batch_elements = c
    A = tf.cast(vector_range_within_batch_elements/vector_num_elements, tf.int64)[..., tf.newaxis]
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


def printAsRagged(msg, S, C , row_splits):
    print(msg, tf.RaggedTensor.from_row_splits(values=tf.RaggedTensor.from_row_splits(values =   S, row_splits=C), 
                                                     row_splits=row_splits))
    

def get_one_over_sigma(beta, beta_min=1e-2):
    return (( 1. / (1. - beta + K.epsilon()) - 1.) + beta_min)
    



def get_neighbour_loss(nneighbours, ccoords, row_splits, beta, is_noise, cluster_asso, beta_min=1e-3):
    
    row_splits = tf.reshape(row_splits, (-1,))
    batch_size_plus_1 = tf.cast(row_splits[-1], tf.int32)
    row_splits32 = tf.slice(row_splits, [0], batch_size_plus_1[..., tf.newaxis])
    row_splits32 = tf.cast(row_splits32, tf.int32)
    row_splits = row_splits[:batch_size_plus_1,...]
    
    
    from rknn_op import rknn_ragged, rknn_op
    
    #stoachastic + nearest neighbours = random close-by neighbours
    random_smeared_ccoords = ccoords + tf.random.normal(tf.shape(ccoords))
    
    ragged_split_added_indices, _ = rknn_op.RaggedKnn(num_neighbors=int(nneighbours), row_splits=row_splits32, data=random_smeared_ccoords, add_splits=True) # [SV, N+1]
    ragged_split_added_indices = ragged_split_added_indices[:,0:][..., tf.newaxis]  # [SV, N], also use self

    d_square = tf.reduce_sum((ccoords[:, tf.newaxis, :] - tf.gather_nd(ccoords, ragged_split_added_indices))**2, axis=-1)  # [SV, N]
    distance = tf.sqrt(d_square+K.epsilon())
    
    
    cluster_asso = tf.expand_dims(cluster_asso, axis=1)
    S = tf.reduce_sum((cluster_asso[:, tf.newaxis, :] - tf.gather_nd(cluster_asso, ragged_split_added_indices))**2, axis=-1) # SV x N
    
    S = tf.where(S < 0.1, tf.zeros_like(S)+1., tf.zeros_like(S))
    Snot = tf.where( S > 0.5, tf.zeros_like(S), tf.zeros_like(S)+1.)
    
   
    
    noise_matrix = tf.gather_nd(is_noise, ragged_split_added_indices)
    S *= 1. - noise_matrix
    #Snot *= 1. - noise_matrix #let's keep this as noise
    
    nS    = tf.reduce_sum(S, axis=1) + K.epsilon()
    nSnot = tf.reduce_sum(Snot, axis=1) + K.epsilon()
    
    beta_matrix = tf.gather_nd(beta, ragged_split_added_indices)
    one_over_sigma_matrix = get_one_over_sigma(beta_matrix, 1e-2)
    one_over_sigma = get_one_over_sigma(beta, 1e-2)
    

    
    
    # for now all these things work in SV x N
    attraction = tf.reduce_sum(S* distance * one_over_sigma_matrix , axis = 1) / nS
    

    repulsion = Snot * (1. - distance) * one_over_sigma_matrix
    repulsion = tf.where(repulsion<0., tf.zeros_like(repulsion), repulsion)
    repulsion = tf.reduce_sum(repulsion , axis=1) / nSnot
    
    #split by ragged not needed if noise is not weighted away
    
    
    #minbeta = S * (1. - beta_matrix) + (1. - S)*100. #beta is between 0 and 1
    #minbeta = tf.reduce_min(minbeta, axis=1)
    #minbeta = tf.where(minbeta > 1., tf.zeros_like(minbeta),minbeta) #just noise contributions
    
    #simplified:
    minbeta =  (1. - beta) * (1. - is_noise)
    
    return tf.reduce_mean(attraction), tf.reduce_mean(repulsion), tf.reduce_mean(minbeta)
    #
    #
    #
    #attraction = tf.RaggedTensor.from_row_splits(values=attraction, row_splits=row_splits)
    #repulsion = tf.RaggedTensor.from_row_splits(values=repulsion, row_splits=row_splits)
    #
    #row_splits = tf.Print(row_splits,[row_splits, attraction],'row_splits ', summarize=200)
    #return tf.reduce_mean(attraction)+tf.reduce_mean(row_splits), 0, 0
    #
    ##now these are B x V
    #
    #N_minus_N_noise = tf.RaggedTensor.from_row_splits(values=(1.-is_noise), row_splits=row_splits)
    #N_minus_N_noise = tf.reduce_sum(N_minus_N_noise, axis=1)+K.epsilon() 
    #
    #attraction_loss = tf.reduce_mean(tf.reduce_sum(attraction, axis=1)  / N_minus_N_noise)
    #repulsion_loss = tf.reduce_mean(tf.reduce_sum(repulsion, axis=1) / N_minus_N_noise)
    #
    ##now min beta loss
    ##attraction = tf.Print(attraction,[attraction],'attraction ')
    #
    ##beta_matrix, S, Snot
    ##would also work: minbeta *= (1.-is_noise)
    #
    ##now normalise per row split
    #minbeta = tf.RaggedTensor.from_row_splits(values=minbeta, row_splits=row_splits)
    #minbeta_loss = tf.reduce_mean(tf.reduce_sum(minbeta, axis=1) / N_minus_N_noise)
    #
    #return 0, 0,minbeta_loss #att, repulsion_loss, minbeta_loss



def get_arb_loss(ccoords, row_splits, beta, is_noise, cluster_asso, beta_min=1e-3,
                 rep_cutoff=100):
    
    #### get dimensions right
    #padded row splits
    row_splits = tf.reshape(row_splits, (-1,))#should not be necessary
    batch_size_plus_1 = tf.cast(row_splits[-1], tf.int32)#int32 needed?
    row_splits = tf.slice(row_splits, [0], batch_size_plus_1[..., tf.newaxis])
     
    cluster_asso = tf.expand_dims(cluster_asso, axis=1)
    
    ####
    
    A,B,C = construct_ragged_matrices_indexing_tensors(row_splits)


    # Jan's losses
    # S is given (I am just setting it to all ones)
    d_square = tf.reduce_sum((tf.gather_nd(ccoords, A) - tf.gather_nd(ccoords, B))**2, axis=-1)

    # reate S and notS
    
    is_notnoise_matrix = (1-tf.gather_nd(is_noise, A))* (1-tf.gather_nd(is_noise, B))
    S = tf.reduce_sum((tf.gather_nd(cluster_asso, A) - tf.gather_nd(cluster_asso, B))**2, axis=-1)
    
    S    = tf.where( S < 0.1, tf.zeros_like(S)+1., tf.zeros_like(S))
    Snot = tf.where( S > 0.5, tf.zeros_like(S), tf.zeros_like(S)+1.)
    S    *= is_notnoise_matrix
    Snot *= is_notnoise_matrix
    #Snot can include noise
    
    print('is_notnoise_matrix',is_notnoise_matrix)
    print('S',S)
    print('Snot',Snot)
    
    #printAsRagged('S', S, C, row_splits)
    #printAsRagged('Snot', Snot, C, row_splits)
    
    
    
    
    #S = tf.Print(S,[tf.shape(is_noise), tf.shape(S)], "S ")

    #now this is S and Snot as defined in the paper draft

    N_minus_N_noise = tf.RaggedTensor.from_row_splits(values=(1-is_noise), row_splits=row_splits)
    N_minus_N_noise = tf.reduce_sum(N_minus_N_noise, axis=1)+K.epsilon() # seems wrong? reduce sum, also axis?
    N = tf.RaggedTensor.from_row_splits(values=(tf.zeros_like(is_noise)+1.), row_splits=row_splits)
    N = tf.reduce_sum(N, axis=1)+K.epsilon() # seems wrong? reduce sum, also axis?
    
    #N_minus_N_noise = tf.Print(N_minus_N_noise, [N], 'N ',summarize=200)

    # This is given sorry it was easy to make a dummy version over here
    #N_minus_N_noise = tf.Print(N_minus_N_noise,[tf.reduce_mean(N_minus_N_noise)],'mean N_minus_N_noise ')

    one_over_collected_sigma_i = tf.gather_nd(get_one_over_sigma(beta, beta_min), A)
    one_over_collected_sigma_j = tf.gather_nd(get_one_over_sigma(beta, beta_min), B)


    

    attractive_loss = (S* tf.sqrt( d_square+K.epsilon()))*(one_over_collected_sigma_i*one_over_collected_sigma_j)
    attractive_loss = tf.RaggedTensor.from_row_splits(values=attractive_loss, row_splits=C)
    attractive_loss = tf.RaggedTensor.from_row_splits(values=attractive_loss, row_splits=row_splits)
    attractive_loss = tf.reduce_sum(attractive_loss, axis=[1,2])
    # Normalize
    
    attractive_loss = attractive_loss / (N_minus_N_noise**2+K.epsilon())
    attractive_loss = killNan(attractive_loss)
    # Mean over the batch dimension
    attractive_loss = tf.reduce_mean(attractive_loss)



    #rep_loss = (Snot)*(one_over_collected_sigma_i*one_over_collected_sigma_j)/(d_square + 1/rep_cutoff + K.epsilon())
    rep_loss = (Snot)*(one_over_collected_sigma_i*one_over_collected_sigma_j) * (1. - tf.sqrt(d_square+K.epsilon()))
    rep_loss = tf.where(rep_loss < 0., tf.zeros_like(rep_loss), rep_loss)
    
    rep_loss = tf.RaggedTensor.from_row_splits(values=rep_loss, row_splits=C)
    rep_loss = tf.RaggedTensor.from_row_splits(values=rep_loss, row_splits=row_splits)
    rep_loss = tf.reduce_sum(rep_loss, axis=[1,2])
    # Normalize
    rep_loss = rep_loss / (N_minus_N_noise**2+K.epsilon())
    rep_loss = killNan(rep_loss)
    # Mean over the batch dimension
    rep_loss = tf.reduce_mean(rep_loss)



    # It's a ragged tensor with two ragged axes
    
    ## this one is NAN directly!
    
    S_r = tf.RaggedTensor.from_row_splits(values=(tf.RaggedTensor.from_row_splits(values = S , row_splits=C)), row_splits=row_splits)
    is_not_same = tf.cast(tf.equal(tf.reduce_sum(S_r, axis=-1),0), tf.float32)
    
    #make it a reduce max
    #min_beta_loss = (1.-S)*100. + (S*(1./one_over_collected_sigma_j))
    
    min_beta_loss = (Snot)*100. + (S*(1./(one_over_collected_sigma_j + K.epsilon())))
   
   
    #min_beta_loss = S * (1. - beta)
    
    min_beta_loss = tf.RaggedTensor.from_row_splits(values = min_beta_loss , row_splits=C)
    
    min_beta_loss = tf.RaggedTensor.from_row_splits(values = min_beta_loss, row_splits=row_splits)
    
    min_beta_loss = tf.reduce_min(min_beta_loss, axis=2) 
    
    min_beta_loss *= (1-is_not_same)
    
    
    
   
   # min_beta_loss *= tf.RaggedTensor.from_row_splits(values = (1 - is_noise), row_splits=row_splits)
    
     ##THIS IS WEIRD OUTPUT: SHOULD BE 0.5 everywhere (see line 109)
    
    min_beta_losssum = tf.reduce_sum(min_beta_loss, axis=1)
    
    #rep_loss = tf.Print(rep_loss,[min_beta_losssum, N_minus_N_noise, min_beta_loss.values, min_beta_loss.row_splits],'min_beta_loss, N_minus_N_noise, min_beta_loss ', summarize=2000)
    rep_loss = tf.Print(rep_loss,[N],'N ', summarize=30)
    # Normalize
    min_beta_loss = min_beta_losssum / (N_minus_N_noise+K.epsilon())
    
    
    # this kicks in immediately for no reason! there is not gradient?!?
    #min_beta_loss = killNan(min_beta_loss)
    # Mean over the batch dimension
    
    #DEBUG: the output should be 0.5
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
    
    print('truth',truth)
    print('pred',pred)
    print('rowsplits',rowsplits)
    
    rowsplits = tf.cast(rowsplits, tf.int64)#just for first loss evaluation from stupid keras
    
    beta_min = 1e-2
    
    d = create_loss_dict(truth, pred)
    #attractive_loss, rep_loss, min_beta_loss= get_arb_loss(d['predCCoords'], 
    #                                                       rowsplits, 
    #                                                       d['predBeta'], 
    #                                                       d['truthIsNoise'], 
    #                                                       d['truthHitAssignementIdx'],
    #                                                       beta_min=beta_min)
    
    attractive_loss, rep_loss, min_beta_loss = get_neighbour_loss(24,
                                                                  d['predCCoords'], 
                                                           rowsplits, 
                                                           d['predBeta'], 
                                                           d['truthIsNoise'], 
                                                           d['truthHitAssignementIdx'],
                                                           beta_min=beta_min)
    
    onedivsigma = get_one_over_sigma(d['predBeta'],beta_min)
    noise_loss  = tf.reduce_mean((d['truthIsNoise']*d['predBeta'])**2)
    
    energy_loss = tf.reduce_mean(onedivsigma*(d['predEnergy'] - d['truthHitAssignedEnergies'])**2/(d['truthHitAssignedEnergies']+0.1))
    pos_loss = tf.reduce_mean(
        onedivsigma*(\
        (d['predEta'] - d['truthHitAssignedEtas'])**2 + \
        (d['predPhi'] - d['truthHitAssignedPhis'])**2 ))
    
    loss = attractive_loss + rep_loss +  min_beta_loss +  noise_loss + energy_loss + pos_loss
    loss = tf.Print(loss,[loss,
                          attractive_loss,
                          rep_loss,
                          min_beta_loss,
                          noise_loss,
                          energy_loss,
                          pos_loss,
                          tf.reduce_mean(onedivsigma)], 'loss , attractive_loss + rep_loss +  min_beta_loss + noise_loss + energy_loss + pos_loss, mean beta ')
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





    
