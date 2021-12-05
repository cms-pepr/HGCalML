
from tensorflow.keras.layers import Dropout, Dense, Concatenate, BatchNormalization, Add, Multiply
from Layers import OnesLike, ExpMinusOne
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply

import tensorflow as tf
from initializers import EyeInitializer



from datastructures import TrainData_OC,TrainData_NanoML
#new format!
def create_outputs(x, feat, energy=None, n_ccoords=3, 
                   n_classes=6, td=TrainData_NanoML(), add_features=True, 
                   fix_distance_scale=False,
                   scale_energy=True,
                   energy_factor=False,
                   energy_proxy=None,
                   name_prefix="output_module"):
    '''
    returns pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id
    '''
    
    assert scale_energy != energy_factor
    
    feat = td.createFeatureDict(feat)
    
    pred_beta = Dense(1, activation='sigmoid',name = name_prefix+'_beta')(x)
    pred_ccoords = Dense(n_ccoords,
                         #this initialisation is much better than standard glorot
                         kernel_initializer=EyeInitializer(stddev=0.01),
                         use_bias=False,
                         name = name_prefix+'_clustercoords'
                         )(x) #bias has no effect
    
    if energy_proxy is None:
        energy_proxy = x
    else:
        energy_proxy = Concatenate()([energy_proxy,x])
    energy_act=None
    if energy_factor:
        energy_act='relu'
    pred_energy = Dense(1,name = name_prefix+'_energy',
                        bias_initializer='ones',#no effect if full scale, useful if corr factor
                        activation=energy_act
                        )(energy_proxy)
    if scale_energy:
        pred_energy = ScalarMultiply(10.)(pred_energy)
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
    
    from GravNetLayersRagged import ElementScaling, KNN, DistanceWeightedMessagePassing, RecalcDistances
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
    
    nidx,dist = KNN(K=24,radius='dynamic' #use dynamic feature
                    )([coords,rs])#all distance weighted afterwards
    
    x = Dense(32,activation='relu',name=name+'dense1',trainable=trainable)(x)
    #the last 8 and 4 just add 5% more gpu
    x = DistanceWeightedMessagePassing([32,32,8,8], #sumwnorm=True,
                                       name=name+'dmp1',trainable=trainable)([x,nidx,dist])# hops are rather light 
    
    
    # this does not come at a high cost
    x = Dense(64,activation='relu',name=name+'dense1b',trainable=trainable)(x)
    x = Dense(32,activation='relu',name=name+'dense1c',trainable=trainable)(x)
    
    learnedcoorddiff = Dense(3,kernel_initializer='zeros',use_bias=False,
                             name=name+'dense2',trainable=trainable)(x)
    
    coords = Concatenate()([coords,learnedcoorddiff])                             
    coords = Dense(3,kernel_initializer=EyeInitializer(mean=0,stddev=0.1),
                   use_bias=False)(coords)
    
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,
                                 name=name+'plt3')([coords,energy,t_idx,rs])
                                 
    dist = RecalcDistances()([coords,nidx])
    
    return coords,nidx,dist,x
    
    
def reduce_indices(x,dist, nidx, rs, t_idx, 
           threshold = 0.5,
           name='reduce_indices',
           trainable=True,
           print_reduction=True,
           use_edges = True,
           return_backscatter=False):   
    
    from tensorflow.keras.layers import Reshape, Flatten
    from GravNetLayersRagged import EdgeCreator, RemoveSelfRef
    from GravNetLayersRagged import EdgeSelector, GroupScoreFromEdgeScores
    from GravNetLayersRagged import NeighbourGroups
    
    from lossLayers import LLEdgeClassifier, LLNeighbourhoodClassifier
    
    goodneighbours = x
    groupthreshold=threshold
    
    if use_edges:
        x_e = Dense(6,activation='relu',
                    name=name+'_ed1',
                    trainable=trainable)(x) #6 x 12 .. still ok
        x_e = EdgeCreator()([nidx,x_e])
        dist = RemoveSelfRef()(dist)#also make it V x K-1
        dist = Reshape((dist.shape[-1],1))(dist)
        x_e = Concatenate()([x_e,dist])
        x_e = Dense(4,activation='relu',
                    name=name+'_ed2',
                    trainable=trainable)(x_e)#keep this simple!
        
        s_e = Dense(1,activation='sigmoid',
                    name=name+'_ed3',trainable=trainable)(x_e)#edge classifier
        #loss
        s_e = LLEdgeClassifier(
            print_loss=True,
            active=trainable,
            scale=1.
            )([s_e,nidx,t_idx])#don't use spectators here yet
    
        nidx = EdgeSelector(
            threshold=threshold
            )([s_e,nidx])
            
        #for nidx, the -1 padding is broken here, but it's ok 
        #because it gets reintroducted with NeighbourGroups
        #
        
        groupthreshold=1e-3#done by edges
        goodneighbours = GroupScoreFromEdgeScores()([s_e,nidx])#flatten edge scores
    
    else: 
        goodneighbours = Dense(1, activation='sigmoid',
                    name=name+'_ngd1',
                    trainable=trainable)(goodneighbours)
        
        goodneighbours = LLNeighbourhoodClassifier(
            print_loss=True,
            active=trainable,
            scale=1.,
            print_batch_time=False
            )([goodneighbours,nidx,t_idx])
        
    
    gnidx, gsel, bg, srs = NeighbourGroups(
        threshold = groupthreshold,
        return_backscatter=return_backscatter,
        print_reduction=print_reduction,
        )([goodneighbours, nidx, rs])
    
    
    return  gnidx, gsel, bg, srs

def reduce(x,coords,energy,dist, nidx, rs, t_idx, t_spectator_weight, 
           threshold = 0.5,
           print_reduction=True,
           name='reduce',
           trainable=True,
           use_edges = True,
           return_backscatter=False):
    
    
    from GravNetLayersRagged import SelectFromIndices, AccumulateNeighbours
    
    gnidx, gsel, bg, srs = reduce_indices(x,dist, nidx, rs, t_idx, 
           threshold = threshold,
           name=name+'_indices',
           trainable=trainable,
           print_reduction=print_reduction,
           use_edges = use_edges,
           return_backscatter=return_backscatter)
    
    
    #these are needed in reduced form
    t_idx, t_spectator_weight = SelectFromIndices()([gsel,t_idx, t_spectator_weight])
    
    coords = AccumulateNeighbours('mean')([coords, gnidx])
    coords = SelectFromIndices()([gsel,coords])
    energy = AccumulateNeighbours('sum')([energy, gnidx])
    energy = SelectFromIndices()([gsel,energy])
    x = AccumulateNeighbours('minmeanmax')([x, gnidx])
    x = SelectFromIndices()([gsel,x])

    rs = srs #set new row splits
    
    return x,coords,energy,nidx, rs, bg, t_idx, t_spectator_weight    
    
    
    
    
    
def pre_selection_model_full(orig_inputs,
                             debug_outdir,
                             trainable,
                             name='pre_selection',
                             debugplots_after=-1,
                             reduction_threshold=0.5,
                             use_edges=True,
                             omit_reduction=False #only trains coordinate transform. useful for pretrain phase
                             ):
    
    from GravNetLayersRagged import AccumulateNeighbours, SelectFromIndices
    from GravNetLayersRagged import SortAndSelectNeighbours, NoiseFilter
    from GravNetLayersRagged import CastRowSplits, ProcessFeatures
    from debugLayers import PlotCoordinates
    from lossLayers import LLClusterCoordinates, LLNotNoiseClassifier, LLFillSpace
    
    rs = CastRowSplits()(orig_inputs['row_splits'])
    t_idx = orig_inputs['t_idx']

    x = ProcessFeatures()(orig_inputs['features'])
    energy = SelectFeatures(0, 1)(orig_inputs['features'])
    coords = SelectFeatures(5, 8)(x)

    # here the actual network starts
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,name=name+'_initial')([coords,energy,t_idx,rs])
    ############## Keep this part to reload the noise filter with pre-trained weights for other trainings
    
    #this takes O(200ms) for 100k hits
    coords,nidx,dist, x = first_coordinate_adjustment(
        coords, x, energy, rs, t_idx, 
        debug_outdir,
        trainable=trainable,
        name=name+'_first_coords',
        debugplots_after=debugplots_after
        )
    #create the gradients
    coords = LLClusterCoordinates(
        print_loss = trainable,
        active = trainable,
        print_batch_time=trainable,
        scale=1.
        )([coords,t_idx,rs])
    
    coords = LLFillSpace(
        print_loss = trainable,
        active = trainable,
        scale=0.1,#just mild
        runevery=20, #give it a kick only every now and then - hat's enough
        )([coords,rs])
    
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,name=name+'_bef_red')([coords,
                                                                           energy,
                                                                           t_idx,rs])
    
    if omit_reduction:
        return {'coords': coords,'dist':dist,'x':x}
    
    dist,nidx = SortAndSelectNeighbours(K=16)([dist,nidx])#only run reduction on 12 closest
    
    '''
    run a full reduction block
    return the noise score in addition - don't select yet
    '''
    
    gnidx, gsel, group_backgather, rs = reduce_indices(x,dist, nidx, rs, t_idx, 
           threshold = reduction_threshold,
           print_reduction=True,
           trainable=trainable,
           name=name+'_reduce_indices',
           use_edges = use_edges,
           return_backscatter=False)
    
    
    out={}
    #do it explicitly
    
    selfeat = orig_inputs['features']
    selfeat = SelectFromIndices()([gsel,selfeat])
    
    #save for later
    orig_dim_coords = coords
    
    x = AccumulateNeighbours('minmeanmax')([x, gnidx])
    x = SelectFromIndices()([gsel,x])
    #add more useful things
    coords = AccumulateNeighbours('mean')([coords, gnidx])
    coords = SelectFromIndices()([gsel,coords])
    energy = AccumulateNeighbours('sum')([energy, gnidx])
    energy = SelectFromIndices()([gsel,energy])
    
    #re-build standard feature layout
    out['features'] = selfeat
    out['coords'] = coords
    out['addfeat'] = x #add more
    out['energy'] = energy
    
    
    ## all the truth
    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectFromIndices()([gsel,orig_inputs[k]])
    
    #debug
    if debugplots_after>0:
        out['coords'] = PlotCoordinates(debugplots_after,
                                        outdir=debug_outdir,name=name+'_after_red')(
                                            [out['coords'],
                                             out['energy'],
                                             out['t_idx'],rs])
    
    #this does not work, but also might not be an issue for the studies        
    #out['backscatter']=bg

    isnotnoise = Dense(1, activation='sigmoid',
                       trainable=trainable,
                       name=name+'_noisescore_d1',
                       )(Concatenate()([out['addfeat'],out['coords']]))
    isnotnoise = LLNotNoiseClassifier(
        print_loss=True,
        scale=1.,
        active=trainable
        )([isnotnoise, out['t_idx']])

    sel, rs, noise_backscatter = NoiseFilter(
        threshold = 0.025, #high signal efficiency filter
        print_reduction=True,
        )([isnotnoise,rs])
    for k in out.keys():
        out[k] = SelectFromIndices()([sel,out[k]])
        
    out['group_backgather']=group_backgather
    out['noise_backscatter_N']=noise_backscatter[0]
    out['noise_backscatter_idx']=noise_backscatter[1]
    out['orig_t_idx'] = orig_inputs['t_idx']
    out['orig_t_energy'] = orig_inputs['t_energy'] #for validation
    out['orig_dim_coords'] = orig_dim_coords
    out['rs']=rs

    return out
    
    
