
from tensorflow.keras.layers import Dropout, Dense, Concatenate, BatchNormalization, Add, Multiply
from Layers import OnesLike, ExpMinusOne
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply

import tensorflow as tf
from initializers import EyeInitializer



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
    
    



def noise_pre_filter(x, coords, rs, listofrest, t_idx, threshold=0.025,
                     dmps=[32,16,16,8],
                     K=24,
                     return_non_filtered_nidx=False
                     ):
    from GravNetLayersRagged import ElementScaling, KNN, LocalDistanceScaling, DistanceWeightedMessagePassing, NoiseFilter
    coords = ElementScaling(name='initial_nf_coord_scaling')(coords)#just rotation and scaling

    ### Noise filter
    #see whats there, strict distance weighted so radius=1 is fine (and fast)
    nidx, dist = KNN(K=K,radius=-1.0, name='initial_nf_knn')([coords,rs])
    dist = LocalDistanceScaling(name='initial_nf_dist_scale')([dist,Dense(1)(x)])
    x = DistanceWeightedMessagePassing(dmps,name='initial_nf_mp')([x,nidx,dist])
    
    listofrest+=[x]
    
    noise_score = Dense(1, activation='sigmoid', name='initial_nf_noise_score')(x)
    rs, bg,\
    *other = NoiseFilter(threshold=threshold,#
                                      loss_enabled=True, 
                                      loss_scale=1.,
                                      print_loss=True,
                                      return_backscatter=True,
                                      print_reduction=True,
                                      name='initial_nf_noise_filter' #so that we can load it back
                                      )([ noise_score, rs] +
                                        listofrest +
                                        [t_idx ])
    if return_non_filtered_nidx:
        return nidx, dist, noise_score, rs, bg, other[:-1],other[-1]#last one is x
    else:
        return noise_score, rs, bg, other[:-1],other[-1]#last one is x
    
    
    
    
def first_coordinate_adjustment(coords, x, energy, rs, t_idx, 
                                debug_outdir,
                                trainable,
                                name='first_coords',
                                 debugplots_after=-1): 
    
    from GravNetLayersRagged import ElementScaling, KNN, DistanceWeightedMessagePassing, WeightedNeighbourMeans, RecalcDistances
    from GravNetLayersRagged import EdgeCreator, DampenGradient, GooeyBatchNorm
    from debugLayers import PlotCoordinates
    
    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)

    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,
                                 outdir=debug_outdir,
                                 name=name+'plt1')([coords,energy,t_idx,rs])
    
    
    if debugplots_after>0 and False:
        coords = PlotCoordinates(debugplots_after,
                                 outdir=debug_outdir,
                                 name=name+'plt1a')([coords,energy,t_idx,rs])
    
    nidx,dist = KNN(K=32,radius=1)([coords,rs])#all distance weighted afterwards
    
    x = Dense(32,activation='relu',name=name+'dense1',trainable=trainable)(x)
    #the last 8 and 4 just add 5% more gpu
    x = DistanceWeightedMessagePassing([32,32,8,8], #sumwnorm=True,
                                       name=name+'dmp1',trainable=trainable)([x,nidx,dist])# hops are rather light 
    
    #
    #this actually takes quite some resources 
    
    #coordsdiff = WeightedNeighbourMeans(name=name+'wnm1')([coords,energy,dist,nidx])
    #coords = Add()([coords,coordsdiff])
    #
    # this does not come at a high cost
    x = Dense(64,activation='relu')(x)
    x = Dense(32,activation='relu')(x)
    
    learnedcoorddiff = Dense(2,kernel_initializer='zeros',use_bias=False,
                             name=name+'dense2',trainable=trainable)(x)
    learnedcoorddiff = Concatenate()([ learnedcoorddiff,
                                      Dense(1,use_bias=False,
                                            name=name+'dense3',trainable=trainable)
                                      (x)])#prefer z axis
    learnedcoorddiff = ScalarMultiply(0.1)(learnedcoorddiff)#make it a soft start
    
                                 
    coords = Add()([coords,learnedcoorddiff])
    
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,
                                 name=name+'plt3')([coords,energy,t_idx,rs])
    
    coords = ElementScaling(name=name+'es2',trainable=trainable)(coords)#simple scaling
    
    
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,
                                 name=name+'plt2')([coords,energy,t_idx,rs])
                                 
    dist = RecalcDistances()([coords,nidx])
    
    return coords,nidx,dist,x
    
    
    
    
    
    
    
    
    
    
    
    
    
