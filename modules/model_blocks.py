
from tensorflow.keras.layers import Dropout, Dense, Concatenate, BatchNormalization, Add, Multiply
from Layers import OnesLike, ExpMinusOne, CondensateToPseudoRS, RaggedSumAndScatter, FusedRaggedGravNetLinParse, VertexScatterer, FusedRaggedGravNetAggAtt
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply

import tensorflow as tf
from initializers import EyeInitializer


def indep_energy_block(x, ccoords, beta, x_row_splits):
    x = StopGradient()(x)
    ccoords = StopGradient()(ccoords)
    beta = StopGradient()(beta)
    feat=[x]
    
    sx, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=0.8,  
                                                    soft=True, 
                                                    threshold=0.1)([x, ccoords, beta, x_row_splits])

    sx = Concatenate()([RaggedSumAndScatter()([sx, psrs, belongs_to_prs]) ,sx])                                            
    feat.append(VertexScatterer()([sx, sids, sx]))
    
    sx,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_prs')([sx, psrs])
                                 
    x = VertexScatterer()([sx, sids, sx])
    feat.append(x)
    x,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_last')([x, x_row_splits])
    feat.append(x)
    x = Concatenate()(feat)
    
    x = Dense(64, activation='elu',name="dense_last_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_enblock_2")(x)
    energy = Dense(1, activation=None,name="dense_enblock_final")(x)
    energy = energy #linear
    return energy

def indep_energy_block2(x, energy, ccoords, beta, x_row_splits, energy_proxy=None, stopxgrad=True):
    if stopxgrad:
        x = StopGradient()(x)
    energy = StopGradient()(energy)
    ccoords = StopGradient()(ccoords)
    beta = StopGradient()(beta)
    feat=[x]
    
    x = Dense(64, activation='elu',name="dense_last_start_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_start_enblock_2")(x)
    x = Concatenate()([energy,x])
    
    sx, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=0.8,  
                                                    soft=True, 
                                                    threshold=0.2)([x, ccoords, beta, x_row_splits])
                                                    
    sx = Dense(128, activation='elu',name="dense_set_sum_input")(sx)
    sx = Dense(128, activation='elu',name="dense_set_sum_input_b")(sx)
    sx = Dense(128, activation='elu',name="dense_set_sum_input_c")(sx)
    
    #deep set like approach
    sx = Concatenate()([RaggedSumAndScatter()([sx, psrs, belongs_to_prs]) ,sx])
                                                   
    feat.append(VertexScatterer()([sx, sids, sx]))
    
    sx,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_prs')([sx, psrs])
                                 
    x = VertexScatterer()([sx, sids, sx])
    feat.append(x)
    x = Concatenate()([x,energy])
    x,_ = FusedRaggedGravNetAggAtt(n_neighbours=256,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_last')([x, x_row_splits, beta])
    feat.append(x)
    x = Concatenate()(feat)
    
    x = Dense(64, activation='elu',name="dense_last_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_enblock_2")(x)
    #energy = None

    energy = Dense(1, activation=None,name="predicted_energy")(x)
    
    return energy







from datastructures import TrainData_OC,TrainData_NanoML
#new format!
def create_outputs(x, feat, energy=None, n_ccoords=3, n_classes=6, td=TrainData_NanoML(), add_features=True, fix_distance_scale=False,name_prefix="output_module"):
    '''
    returns pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id
    '''
    
    feat = td.createFeatureDict(feat)
    
    pred_beta = Dense(1, activation='sigmoid',name = name_prefix+'_beta')(x)
    pred_ccoords = Dense(n_ccoords,
                         #this initialisation is much better than standard glorot
                         kernel_initializer=EyeInitializer(stddev=0.01),
                         use_bias=False,
                         name = name_prefix+'_clustercoords'
                         )(x) #bias has no effect
    
    pred_energy = ScalarMultiply(10.)(Dense(1,name = name_prefix+'_energy')(x))
    if energy is not None:
        pred_energy = Multiply()([pred_energy,energy])
        
    pred_pos =  Dense(2,use_bias=False,name = name_prefix+'_pos')(x)
    pred_time = ScalarMultiply(10.)(Dense(1)(x))
    
    if add_features:
        pred_pos =  Add()([feat['recHitXY'],pred_pos])
    pred_id = Dense(n_classes, activation="softmax",name = name_prefix+'_class')(x)
    
    pred_dist = OnesLike()(pred_time)
    if not fix_distance_scale:
        pred_dist = Dense(1, activation='sigmoid',name = name_prefix+'_dist')(x) 
        #this needs to be bound otherwise fully anti-correlated with coordates scale
    return pred_beta, pred_ccoords, pred_dist, pred_energy, pred_pos, pred_time, pred_id
    
    



def noise_pre_filter(x, coords, rs, listofrest, t_idx, threshold=0.025):
    from GravNetLayersRagged import ElementScaling, KNN, LocalDistanceScaling, DistanceWeightedMessagePassing, NoiseFilter
    coords = ElementScaling(name='initial_nf_coord_scaling')(coords)#just rotation and scaling

    ### Noise filter
    #see whats there, strict distance weighted so radius=1 is fine (and fast)
    nidx, dist = KNN(K=24,radius=-1.0, name='initial_nf_knn')([coords,rs])
    dist = LocalDistanceScaling(name='initial_nf_dist_scale')([dist,Dense(1)(x)])
    x = DistanceWeightedMessagePassing([32,16,16,8],name='initial_nf_mp')([x,nidx,dist])
    
    noise_score = Dense(1, activation='sigmoid', name='initial_nf_noise_score')(x)
    rs, bg,\
    *other = NoiseFilter(threshold=threshold,#
                                      loss_enabled=True, 
                                      loss_scale=1.,
                                      print_loss=True,
                                      print_reduction=True,
                                      name='initial_nf_noise_filter' #so that we can load it back
                                      )([ noise_score, rs] +
                                        listofrest +
                                        [t_idx ])
    return coords, nidx, dist, noise_score, rs, bg, other
