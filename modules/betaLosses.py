

import tensorflow as tf
import keras
from keras import losses
import keras.backend as K
from caloGraphNN import euclidean_squared

#factorise a bit


#
#
#
#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)
#
#

def create_pixel_loss_dict(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = colours
    
    truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    outdict={}
    #truth = tf.Print(truth,[truth],'truth',summarize=30)
    reshaping = (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],-1)
    #make it all lists
    outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], reshaping) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], reshaping, name="lala") 
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], reshaping)  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], reshaping) 
    n_objects = truth[:,0,0,8]

    print('pred',pred.shape)

    outdict['p_beta']   =  tf.reshape(pred[:,:,:,0:1], reshaping)
    outdict['p_pos']    =  tf.reshape(pred[:,:,:,1:3], reshaping, name="lulu")
    outdict['p_ID']     =  tf.reshape(pred[:,:,:,3:6], reshaping)
    outdict['p_dim']    =  tf.reshape(pred[:,:,:,6:8], reshaping)
    p_object  = pred[:,0,0,8]
    outdict['p_ccoords'] = tf.reshape(pred[:,:,:,9:], reshaping)
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    
    
    return outdict


def d_euclidean_squared(a,b):
    # B x V x F
    #output: B x V x V x 1
    a = tf.expand_dims(a, axis=1) # B x 1 x V x F
    b = tf.expand_dims(b, axis=2) # B x V x 1 x F
    return tf.expand_dims(tf.reduce_sum((a-b)**2 , axis=-1), axis=3)


def expand_by_mult(A):
    '''
    A: B x V x 1
    out: B x V x V x 1, with out_ij = Ai * Aj
    '''
    assert len(A.shape) == 3
    A1 = tf.expand_dims(A, axis=1) # B x 1 X V x 1
    A2 = tf.expand_dims(A, axis=2) # B x V X 1 x 1
    A =  A1 * A2 # B x V X V x 1
    return A

def mean_nvert_with_nactive(A, n_active):
    '''
    n_active: B x 1
    A : B x V x F
    
    out: B x F
    '''
    assert len(A.shape) == 3
    assert len(n_active.shape) == 2
    den = n_active + K.epsilon()
    ssum = tf.reduce_sum(A, axis=1)
    return ssum / den


def matrix_mean_nvert_with_nactive(AM, n_active):
    '''
    n_active: B x 1
    AM : B x V x V x F
    
    out: B x V x F # mean on second V dimension
    '''
    assert len(AM.shape) == 4
    assert len(n_active.shape) == 2
    den = tf.expand_dims(n_active, axis=1) + K.epsilon()
    ssum = tf.reduce_sum(AM, axis=2)
    return ssum/den
    
def create_others_and_self_masks(true_pos):
    true_dist = d_euclidean_squared(true_pos, true_pos)
    o = tf.where(true_dist>K.epsilon(),tf.zeros_like(true_dist)+1., tf.zeros_like(true_dist))
    s = tf.where(o>K.epsilon(),tf.zeros_like(o), tf.zeros_like(o)+1.)
    return s, o
    
    
def coord_loss(truth,pred):
    
    minimum_confidence = 1e-4
    distance_scale = .05
    repulse_exp = False
    repulse_scale = .1
    repulse_y_max = 10. #only for non exp mode
    repulsion_strength = 50.
    attraction_strength = 1.
    supress_noise_strength = 100.
    beta_strength = 100.
    
    ID_strength =  0.1
    pos_strength = 0.1
    box_strength = 0.1
    
    d = create_pixel_loss_dict(truth,pred)
    
    ### zero beta for noise
    supress_noise_loss = mean_nvert_with_nactive(((1.-d['t_mask'])*d['p_beta']), 
                                            d['n_active']-d['n_nonoise'])
    #B x 1
    supress_noise_loss = supress_noise_strength * tf.reduce_mean(supress_noise_loss)

    ### beta scaling multiplicative
    beta_scaling = 1./(( 1. - d['p_beta'])) - 1. + minimum_confidence #B x V x 1
    
    print('beta_scaling', beta_scaling.shape)
    beta_scaling_2D = expand_by_mult(beta_scaling)
    
    t_mask_2D = expand_by_mult(d['t_mask'])
    print('t_mask_2D', t_mask_2D.shape)

    distance = d_euclidean_squared(d['p_ccoords'], d['p_ccoords'])
    
    self_mask, other_mask = create_others_and_self_masks(d['t_pos'])
    print('self_mask', self_mask.shape)
    print('other_mask', other_mask.shape)
    print('t_mask_2D',t_mask_2D.shape)
    print('beta_scaling_2D',beta_scaling_2D.shape)

    #repulsion
    repulsion=None
    if repulse_exp:
        repulsion =  t_mask_2D * beta_scaling_2D * other_mask * tf.exp(-0.5/repulse_scale*tf.sqrt(distance+K.epsilon())) 
    else:
        repulsion =  t_mask_2D * beta_scaling_2D * other_mask * 1./(distance/repulse_scale**2 + 1/repulse_y_max)
    repulsion = matrix_mean_nvert_with_nactive(repulsion, d['n_nonoise'])
    print('repulsion',repulsion.shape)
    repulsion_loss = repulsion_strength * mean_nvert_with_nactive(repulsion, d['n_nonoise'])
    print('repulsion_loss',repulsion_loss.shape)
    # B x 1
    repulsion_loss = tf.reduce_mean(repulsion_loss)
    
    attraction = t_mask_2D * beta_scaling_2D * self_mask * 1./distance_scale**2 * distance
    attraction = matrix_mean_nvert_with_nactive(attraction, d['n_nonoise'])
    attraction_loss = attraction_strength * mean_nvert_with_nactive(attraction, d['n_nonoise'])
    # B x 1
    attraction_loss = tf.reduce_mean(attraction_loss)
    
    
    ######### beta penalty, one per. #B x V x 1
    self_invbeta = tf.tile(tf.expand_dims( 1. - d['p_beta'] , axis=1), [1,tf.shape(d['p_beta'])[1] ,1, 1] )
    #B x V x V x 1
    print('self_invbeta',self_invbeta.shape)
    
    #other_mask= tf.Print(other_mask,[other_mask],'other_mask ', summarize=200)
    self_invbeta += other_mask * (tf.zeros_like(other_mask)+10) #make sure the other parts don't contribute
    self_min_beta = d['t_mask']*tf.reduce_min(self_invbeta, axis=2)
    print('self_min_beta',self_min_beta.shape)
    min_beta_loss = mean_nvert_with_nactive(self_min_beta, d['n_nonoise'])
    print('min_beta_loss',min_beta_loss.shape)
    #B x 1
    min_beta_loss = beta_strength * tf.reduce_mean(min_beta_loss)
    
    
    ########### Actual losses for ID etc. #################
    tID = d['t_mask']*d['t_ID']
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-K.epsilon(),tID)
    pID = d['t_mask']*d['p_ID']
    pID = tf.where(pID<=0.,tf.zeros_like(pID)+K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-K.epsilon(),pID)
    
    xentr = beta_scaling * (-1.)* tf.reduce_sum(tID * tf.log(pID) ,axis=-1, keepdims=True)
    xentr_loss = mean_nvert_with_nactive(d['t_mask']*xentr, d['n_nonoise'])
    xentr_loss = ID_strength * tf.reduce_mean(xentr_loss)
   
   
    ######### position loss


    posl = pos_strength * box_strength * beta_scaling*(d['t_pos'] - d['p_pos'])**2
    posl = mean_nvert_with_nactive(d['t_mask']*posl,d['n_nonoise'])
    posl = tf.reduce_mean( posl)
    
    ######### bounding box loss
    
    
    bboxl = box_strength * beta_scaling*(d['t_dim'] - d['p_dim'])**2
    bboxl = mean_nvert_with_nactive(d['t_mask']*bboxl,d['n_nonoise'])
    bboxl = tf.reduce_mean( bboxl)
    
    
    #######################################################
    
    
    loss = repulsion_loss + attraction_loss + min_beta_loss + supress_noise_loss + bboxl  + xentr_loss + posl # 
    loss = tf.Print(loss,[loss,
                          repulsion_loss,
                          attraction_loss,
                          min_beta_loss,
                          supress_noise_loss,
                          posl,
                          bboxl,
                          xentr_loss
                          ],
                          'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    
    
    return loss


def true_euclidean_squared(a,b):
    # B x V x F
    #output: B x V x V 
    a = tf.expand_dims(a, axis=1) # B x 1 x V x F
    b = tf.expand_dims(b, axis=2) # B x V x 1 x F
    return tf.reduce_sum((a-b)**2 , axis=-1)


def get_n_active_pixels(p):
    flattened = tf.reshape(p,(tf.shape(p)[0],-1))
    return tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32')


def mean_with_mask(n_vertices,in_vertices, axis, addepsilon=0.):
    import keras.backend as K
    return tf.reduce_sum(in_vertices,axis=axis) / (n_vertices+K.epsilon()+addepsilon)    


    

def beta_pixel_clustercoords(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = colours
    
    truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    '''
    
    print('truth.shape',truth.shape)
    print('pred.shape',pred.shape)
    
    create_pixel_loss_dict(truth, pred)
    
    exit()
    
    #distance/repulse scale is 1/x
    cluster_distance_scale = 1.
    cluster_repulse_scale = 1.
    true_distance_threshold = 0.003
    beta_threshold_width=.01 #where it is basically gone
    beta_threshold = 0.85
    repulse_offset = 100.
    minimum_confidence = 1e-3
    conf_scaler = 20.
    low_conf_scaler = 0.
    exp_repulsion=True
    repulsion_strength = 5.
    
    reshaping = (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],-1)
    #make it all lists
    t_mask =    tf.reshape(truth[:,:,:,0:1], reshaping) 
    true_pos =  tf.reshape(truth[:,:,:,1:3], reshaping) 
    true_ID =   tf.reshape(truth[:,:,:,3:6], reshaping) 
    true_dim =  tf.reshape(truth[:,:,:,6:8], reshaping) 
    n_objects = truth[:,0,0,8]
    # B x P x P x N
    
    
    #make it all lists
    p_beta   =  tf.reshape(pred[:,:,:,0:1], reshaping)
    p_tpos   =  tf.reshape(pred[:,:,:,1:3], reshaping)
    p_ID     =  tf.reshape(pred[:,:,:,3:6], reshaping)
    p_dim    =  tf.reshape(pred[:,:,:,6:8], reshaping)
    p_object  = pred[:,0,0,8]
    p_ccoords = tf.reshape(pred[:,:,:,9:11], reshaping)
                           
    pred_confidence = p_beta[:,:,0] 
    conf_scaling = tf.abs(1. - (pred_confidence + minimum_confidence))
    
    
    ###noise
    
    
    noiseterm = tf.reduce_mean((p_beta*(1.-t_mask))**2)
    
    ###
    
    n_active_pixels = get_n_active_pixels(t_mask)
    # B
    n_vertex = tf.expand_dims(n_active_pixels,axis=1)
    
    not_filled_mask = not_filled_mask_matrix(t_mask)
    
    
    cluster_coord_distances = true_euclidean_squared(p_ccoords, p_ccoords)
    others_distance_mask = tf.where(true_euclidean_squared(true_pos, true_pos)>true_distance_threshold, 
                                  tf.zeros_like(cluster_coord_distances)+1., tf.zeros_like(cluster_coord_distances))
    self_distance_mask =   tf.where(others_distance_mask>K.epsilon(), 
                                  tf.zeros_like(cluster_coord_distances), tf.zeros_like(cluster_coord_distances)+1.)
    
    conf_scaling_exp = tf.expand_dims(conf_scaling, axis=2) #B x V x 1
    conf_scaling_exp *= tf.expand_dims(conf_scaling, axis=1)#B x V x V
    
    cluster_coord_loss=None
    if exp_repulsion:
        cluster_coord_loss =  tf.exp(-0.5*cluster_repulse_scale*tf.sqrt(cluster_coord_distances+K.epsilon())) #only take wrong associations
    else:
        cluster_coord_loss = 1/(cluster_repulse_scale*tf.sqrt(cluster_coord_distances+K.epsilon()) + 1./(repulse_offset+K.epsilon()))
    
    cluster_coord_loss *= others_distance_mask
    cluster_coord_loss *= not_filled_mask
    cluster_coord_loss *= (1./(conf_scaling_exp + K.epsilon()) - 1.) #keep some gradient
    cluster_coord_loss =  tf.reduce_sum(cluster_coord_loss, axis=2)/(n_vertex+K.epsilon())
    #B x V
    
    self_distance_loss =  cluster_distance_scale**2*(cluster_coord_distances)**2
    self_distance_loss *= self_distance_mask
    self_distance_loss *= not_filled_mask
    self_distance_loss *= (1./(conf_scaling_exp + K.epsilon()) - 1.)
    self_distance_loss =  tf.reduce_sum(self_distance_loss, axis=2)/(n_vertex+K.epsilon())
    
    
    conf_per_self = tf.tile(tf.expand_dims(conf_scaling, axis=1), [1,tf.shape(conf_scaling)[1] ,1] )  #B x V x V(same)
    conf_per_self += 10. * others_distance_mask #so that the minimum must be in 'self'
    
    max_conf_per_self = tf.reduce_min(conf_per_self, axis=-1) # it's enough to promote 1 !
    max_conf_per_self = tf.where(tf.squeeze(t_mask,axis=2)>0, max_conf_per_self, tf.zeros_like(max_conf_per_self))
    #only where there is actually something useful
    conf_mean = tf.reduce_sum(max_conf_per_self,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())#mean_with_mask(n_vertex, max_conf_per_self , axis=1, addepsilon=addepsilon)
    
    multi_conf_penalty = tf.reduce_sum(((1.-beta_threshold)-
                                       tf.clip_by_value(pred_confidence,0,(1.-beta_threshold)))*tf.squeeze(t_mask,axis=2),
                                       axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())
    
    
    import keras
    IDloss = keras.losses.categorical_crossentropy(true_ID, p_ID)
    print('IDloss',IDloss.shape)
    #only consider highest confidence one per object
    
    #don't sale with CL here otherwise it creates a pressure on CL...
    IDloss *= (1./(conf_scaling + K.epsilon()) - 1.) * tf.squeeze(t_mask, axis=2)
    
    
    beta_threshold_loss = tf.exp(- tf.abs(pred_confidence-beta_threshold)**2/(beta_threshold_width*2.)**2) 
    beta_threshold_loss = tf.exp(- tf.abs(pred_confidence-(1.-beta_threshold))**2/(beta_threshold_width*2.)**2) 
    inbetween = tf.math.logical_and(pred_confidence<beta_threshold, pred_confidence>(1.-beta_threshold))
    beta_threshold_loss = tf.where(inbetween,tf.zeros_like(beta_threshold_loss)+1., beta_threshold_loss)
    beta_threshold_loss *= tf.squeeze(t_mask, axis=2)
    
    
    true_pos_loss = tf.reduce_sum((true_pos/64. - p_tpos/64.)**2 * t_mask,axis=-1)
    true_pos_loss *= (1./(conf_scaling + K.epsilon()) - 1.)
    print('true_pos_loss.shape',true_pos_loss.shape)
    
    
    above_threshold = tf.where(tf.squeeze(t_mask,axis=2)*pred_confidence>beta_threshold, tf.zeros_like(pred_confidence)+1,tf.zeros_like(pred_confidence))
    n_object_pred = tf.cast(tf.count_nonzero(above_threshold, axis=1), dtype='float32')
    n_object_loss = 0.*(n_objects-n_object_pred)**2
    print('n_object_loss',n_object_loss.shape)
    
    ##common part, masking done again here
    
    cluster_loss =   repulsion_strength * cluster_coord_loss + 1. * self_distance_loss # + .1*IDloss  + .1 * beta_threshold_loss#+ 0.1 * multi_conf_penalty #+ 1e-3 * truth_pos_constraint
    #cluster_loss += 0.1 * true_pos_loss
    print('cluster_loss',cluster_loss.shape)
    print('conf_scaling',conf_scaling.shape)
    cluster_loss *= tf.squeeze(t_mask, axis=2)
    cluster_loss = tf.reduce_sum(cluster_loss,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())
    
    ### sum(beta)_>betathreshold,same/max(beta)_same != 1
    ### leave a bit of wiggle room between beta = 0 and beta = 0.1
    ### by extra penalty term that requires sum(beta) (not sum maxmin beta) to be small
    ### but with lower scaling than the normal conf_mean term
    max_self_sum_loss = (conf_scaling_exp - 1.) * self_distance_mask * not_filled_mask
    max_self_sum_loss = tf.where(max_self_sum_loss < (1.-beta_threshold), tf.zeros_like(max_self_sum_loss),max_self_sum_loss)
    print('max_self_sum_loss',max_self_sum_loss.shape)
    max_self_sum_loss = tf.reduce_sum(max_self_sum_loss, axis=-1) / (
        K.epsilon() + tf.reduce_max(max_self_sum_loss,axis=-1)) - 1. #B x V
    print('max_self_sum_loss',max_self_sum_loss.shape)
    max_self_sum_loss = tf.reduce_sum(max_self_sum_loss,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())
    
    
    loss_per_batchelem = cluster_loss #+ 1. * pos_loss + 1. * E_loss + 0.01 * ID_loss
    
    loss_per_batchelem = tf.Print(loss_per_batchelem,
                                  [tf.reduce_mean(mean_with_mask(n_vertex, cluster_coord_loss, axis=1)),
                                   tf.reduce_mean(mean_with_mask(n_vertex, self_distance_loss, axis=1)),
                                   tf.reduce_mean(low_conf_scaler*multi_conf_penalty),
                                   #tf.reduce_mean(conf_mean),
                                   tf.reduce_mean(pred_confidence),
                                   tf.reduce_mean(IDloss),
                                   tf.reduce_mean(.1 * beta_threshold_loss),
                                   tf.reduce_mean(0.1 * true_pos_loss),
                                   tf.reduce_mean(max_self_sum_loss),
                                   noiseterm
                                   #tf.reduce_mean(mean_with_mask(n_vertex, conf_penalty, axis=1))
                                   ],
                                   'cluster_coord_loss, self_distance_loss, multi_conf_penalty, pred_confidence, IDloss, beta_threshold_loss, true_pos_loss , max_self_sum_loss, noiseterm')
    
    

    loss = loss_per_batchelem + conf_scaler * conf_mean + low_conf_scaler* multi_conf_penalty
    loss = tf.reduce_mean( loss ) + noiseterm
    loss = tf.Print(loss,[loss],'loss ')
    return loss


def beta_clusterloss_clustercoords(truth, pred):
    
    
    #truth: posx, posy, Efull, 3x ID
    #pred:  3x linear, 3x softmax 
    
    true_pos = truth[:,:,0:2]
    true_E   = truth[:,:,2:3]
    
    isHGCalData = True
    addepsilon = 0.
    true_distance_threshold=1e-4
    minimum_confidence = 1e-2
    n_showers=1.
    cluster_distance_scale = 1.
    conf_scaler=0.3
    if isHGCalData:
        addepsilon = 0.
        true_E   = tf.expand_dims(truth[:,:,-1],axis=2)
        #true_E = tf.Print(true_E,[true_E],'true_E ', summarize=2000)
        true_pos = tf.cast(tf.math.argmax(truth[:,:,0:-1],axis=-1), dtype='float32')
        n_showers = tf.reduce_max(true_pos, axis=1) + 0.1 #mild scaling with n showers, for only one shower do almost nothing
        #n_showers = tf.Print(n_showers,[n_showers],'n_showers ')
        #for assignment use the largest
        true_pos = tf.expand_dims(true_pos, axis=2)
        true_pos = tf.tile(true_pos, [1,1,2])
        true_distance_threshold=0.1
        minimum_confidence=1e-2
        cluster_distance_scale=1.
        conf_scaler=3.
        
        #true_pos = tf.Print(true_pos,[true_pos],'true_pos ', summarize=500)
    
    #true_ID  = truth[:,:,3:6]
    
    n_vertex = tf.cast(tf.count_nonzero(true_E, axis=1), dtype='float32')
    #remove a few low hit events in the HGCal sample
    n_vertex = tf.where(n_vertex < 5., tf.zeros_like(n_vertex)+1e5,n_vertex)
    #n_vertex= tf.Print(n_vertex,[n_vertex],'n_vertex ', summarize=100)
    if isHGCalData:
        n_vertex = tf.where(n_vertex < 50., tf.zeros_like(n_vertex)+1e5,n_vertex)
        #take weight from low vertex events
    
    
    print('n_vertex',n_vertex.shape)
    
    pred_pos = pred[:,:,0:2]
    pred_E   = pred[:,:,2:3]
    pred_ID  = pred[:,:,3:6]
    pred_cluster_coords = pred[:,:,7:9]
    
    
    pred_confidence = pred[:,:,6]
    
    #pos_loss = tf.reduce_mean((true_pos-pred_pos)**2,axis=-1) * cluster_distance_scale
    #norm to about 1 being one minimum distance, only 
    
    #E_loss  = tf.reduce_mean((true_E - pred_E)**2,axis=-1) / (50.*50.)
    
    #ID_loss = 0 #losses.categorical_crossentropy(true_ID, pred_ID)
    
    #distance to each
    
    conf_scaling = 1. - pred_confidence # tf.atanh(tf.clip_by_value(1. - pred_confidence, K.epsilon(), 1.-K.epsilon()))
    #B x V
    #conf_mean = mean_with_mask(n_vertex, (1. - pred_confidence) , axis=1)
    #B x 1
    
    
    
    cluster_coord_distances = true_euclidean_squared(pred_cluster_coords, pred_cluster_coords)
    
    others_distance_mask = tf.where(true_euclidean_squared(true_pos, true_pos)>true_distance_threshold, 
                                  tf.zeros_like(cluster_coord_distances)+1., tf.zeros_like(cluster_coord_distances))
    self_distance_mask =   tf.where(others_distance_mask>K.epsilon(), 
                                  tf.zeros_like(cluster_coord_distances), tf.zeros_like(cluster_coord_distances)+1.)
    
    #self_distance_mask = tf.Print(self_distance_mask,[self_distance_mask],'self_distance_mask ', summarize=1500)
    
    not_filled_mask = not_filled_mask_matrix(true_E)
    
    conf_scaling_exp = tf.expand_dims(conf_scaling, axis=2) #B x V x 1
    conf_scaling_exp *= tf.expand_dims(conf_scaling, axis=1)#B x V x V
    
    cluster_coord_loss =  tf.exp(-0.5*cluster_distance_scale*tf.sqrt(cluster_coord_distances+K.epsilon())) #only take wrong associations
    cluster_coord_loss *= others_distance_mask
    cluster_coord_loss *= not_filled_mask
    cluster_coord_loss *= (1./(conf_scaling_exp + K.epsilon()) - 1. + minimum_confidence) #keep some gradient
    cluster_coord_loss =  tf.reduce_sum(cluster_coord_loss, axis=2)/(n_vertex+K.epsilon())
    #B x V
    
    self_distance_loss =  cluster_distance_scale*(cluster_coord_distances)**2
    self_distance_loss *= self_distance_mask
    self_distance_loss *= not_filled_mask
    self_distance_loss *= (1./(conf_scaling_exp + K.epsilon()) - 1. + minimum_confidence)
    self_distance_loss =  tf.reduce_sum(self_distance_loss, axis=2)/(n_vertex+K.epsilon())
    
    #B x V
    
    #conf per shower constraint to make the network pick one representative
    # self_distance_mask : B x V x V , get n per shower per vertex
    #n_show_vert  = tf.cast(tf.count_nonzero(self_distance_mask, axis=2), dtype='float32') #B x V
    #conf_per_self = tf.tile(tf.expand_dims(pred_confidence, axis=1), [1,tf.shape(n_show_vert)[1] ,1] )*self_distance_mask #B x V x V(same)
    #conf_per_self = conf_per_self / (tf.reduce_max(conf_per_self, axis=-1, keepdims=True) + K.epsilon())
    #conf_penalty = tf.reduce_sum(conf_per_self, axis=-1)/(n_show_vert + K.epsilon()) # B x V 
    
    conf_per_self = tf.tile(tf.expand_dims(conf_scaling, axis=1), [1,tf.shape(conf_scaling)[1] ,1] )  #B x V x V(same)
    conf_per_self += 10. * others_distance_mask #so that the minimum must be in 'self'
    
    #conf_per_self = tf.Print(conf_per_self,[conf_per_self],'conf_per_self ',summarize=1000)
    
    max_conf_per_self = tf.reduce_min(conf_per_self, axis=-1) # it's enough to promote 1 !
    max_conf_per_self = tf.where(tf.squeeze(true_E,axis=2)>0, max_conf_per_self, tf.zeros_like(max_conf_per_self))
    #only where there is actually something useful
    conf_mean = tf.reduce_sum(max_conf_per_self,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())#mean_with_mask(n_vertex, max_conf_per_self , axis=1, addepsilon=addepsilon)
    #tf.reduce_mean(max_conf_per_self , axis=1)
    ## ## ## Still needs an explicit penalty if there is more than one per shower.. 
    ## how without committing to a number of showers?
    
    #penalty for too many high conf vertices, handle with care
    multi_conf_penalty = tf.reduce_sum(pred_confidence,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())
    #set in perspective with n showers (needs add input)
    
    # penalty = (multi_conf_penalty - n_showers)**2
    
    
    #actually a max, but for consistency
    
    ## any functional form can be added here to make it stronger without screwing up the penalty asking for high confidence
    
    
    #just a mild constraint from the true position
    #truth_pos_constraint = tf.reduce_sum((true_pos-pred_cluster_coords)**2,axis=2) * cluster_distance_scale
    #B x 1
    #truth_pos_constraint *= (1./(conf_scaling + K.epsilon()) - 1. + minimum_confidence)
    
    cluster_loss =   1. * cluster_coord_loss + 1. * self_distance_loss  #+ 0.1 * multi_conf_penalty #+ 1e-3 * truth_pos_constraint
    #cluster_loss /= (conf_scaling + K.epsilon())
    cluster_loss = tf.reduce_sum(cluster_loss,axis=1)/(tf.squeeze(n_vertex,axis=1)+K.epsilon())
    
    
    loss_per_batchelem = cluster_loss #+ 1. * pos_loss + 1. * E_loss + 0.01 * ID_loss
    
    loss_per_batchelem = tf.Print(loss_per_batchelem,
                                  [tf.reduce_mean(mean_with_mask(n_vertex, cluster_coord_loss, axis=1)),
                                   tf.reduce_mean(mean_with_mask(n_vertex, self_distance_loss, axis=1)),
                                   tf.reduce_mean(0.1*multi_conf_penalty),
                                   tf.reduce_mean(conf_mean),
                                   tf.reduce_mean(pred_confidence),
                                   noiseterm,
                                   #tf.reduce_mean(mean_with_mask(n_vertex, conf_penalty, axis=1))
                                   ],
                                   'cluster_coord_loss, self_distance_loss, multi_conf_penalty, conf_mean, pred_confidence, noiseterm ')
    
    

    loss = loss_per_batchelem + conf_scaler * conf_mean
    loss = tf.reduce_mean( loss * n_showers) + noiseterm
    loss = tf.Print(loss,[loss],'loss ')
    
        
    return loss
    