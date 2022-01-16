
from tensorflow.keras.layers import Dropout, Dense, Concatenate, BatchNormalization, Add, Multiply
from Layers import OnesLike, ExpMinusOne
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply

import tensorflow as tf
from Initializers import EyeInitializer



from datastructures import TrainData_NanoML


def extent_coords_if_needed(coords, x, n_cluster_space_coordinates,name='coord_extend'):
    if n_cluster_space_coordinates > 3:
        x = Concatenate()([coords,x])
        extendcoords = Dense(n_cluster_space_coordinates-3,
                             use_bias=False,
                             name=name+'_dense',
                             kernel_initializer=EyeInitializer(stddev=0.001)
                             )(x)
        coords = Concatenate()([coords, extendcoords])
    return coords

#new format!
def create_outputs(x, feat, energy=None, n_ccoords=3, 
                   n_classes=4, td=TrainData_NanoML(), add_features=True, 
                   fix_distance_scale=False,
                   scale_energy=False,
                   energy_factor=True,
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
                         kernel_initializer=EyeInitializer(stddev=0.001),
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
        pred_dist = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name = name_prefix+'_dist')(x))+1e-2
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
                                n_coords=3,
                                record_metrics=False,
                                debugplots_after=-1,
                                use_multigrav=True): 
    
    from GravNetLayersRagged import ElementScaling, KNN, DistanceWeightedMessagePassing, RecalcDistances, MultiAttentionGravNetAdd
    from DebugLayers import PlotCoordinates
    
    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)
    
    coords = extent_coords_if_needed(coords, x, n_coords, name=name+'_coord_ext')

    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,
                                 outdir=debug_outdir,
                                 name=name+'plt1')([coords,energy,t_idx,rs])
    
    
    nidx,dist = KNN(K=32,radius='dynamic', #use dynamic feature # 24
                    record_metrics=record_metrics,
                    name=name+'_knn',
                    min_bins=[7,7]
                    )([coords,rs])#all distance weighted afterwards
    
    x = Dense(32,activation='relu',name=name+'dense1',trainable=trainable)(x)
    #the last 8 and 4 just add 5% more gpu
    if use_multigrav:
        x = DistanceWeightedMessagePassing([16,16], #sumwnorm=True,
                                       name=name+'matt_dmp1',trainable=trainable)([x,nidx,dist])# hops are rather light 
        x_matt = Dense(8,activation='relu',name=name+'dense_matt_1',trainable=trainable)(x)
        x_matt = MultiAttentionGravNetAdd(
            4,record_metrics=False,#gets too big
            name=name+'multi_att_gn')([x,x_matt,coords,nidx])
        x = Concatenate()([x,x_matt])
        #x = DistanceWeightedMessagePassing([16], #sumwnorm=True,
        #                               name=name+'matt_dmp2',trainable=trainable)([x,nidx,dist])# hops are rather light 
    else:
        x = DistanceWeightedMessagePassing([32,32,8,8], #sumwnorm=True,
                                       name=name+'dmp1',trainable=trainable)([x,nidx,dist])# hops are rather light 
    
    
    # this does not come at a high cost
    x = Dense(64,activation='relu',name=name+'dense1b',trainable=trainable)(x)
    x = Dense(32,activation='relu',name=name+'dense1c',trainable=trainable)(x)
    
    learnedcoorddiff = Dense(n_coords,kernel_initializer='zeros',use_bias=False,
                             name=name+'dense2',trainable=trainable)(x)
    
    coords = Concatenate()([coords,learnedcoorddiff])                             
    coords = Dense(n_coords,kernel_initializer=EyeInitializer(mean=0,stddev=0.1),
                   trainable=trainable,
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
           edge_nodes_0 = 6,
           edge_nodes_1 = 4,
           record_metrics=False,
           return_backscatter=False):   
    
    from tensorflow.keras.layers import Reshape
    from GravNetLayersRagged import EdgeCreator, RemoveSelfRef
    from GravNetLayersRagged import EdgeSelector, GroupScoreFromEdgeScores
    from GravNetLayersRagged import NeighbourGroups
    
    from LossLayers import LLEdgeClassifier, LLNeighbourhoodClassifier
    
    goodneighbours = x
    groupthreshold=threshold
    
    if use_edges:
        x_e = Dense(edge_nodes_0,activation='relu',
                    name=name+'_ed1',
                    trainable=trainable)(x) #6 x 12 .. still ok
        x_e = EdgeCreator()([nidx,x_e])
        dist = RemoveSelfRef()(dist)#also make it V x K-1
        dist = Reshape((dist.shape[-1],1))(dist)
        x_e = Concatenate()([x_e,dist])
        x_e = Dense(edge_nodes_1,activation='relu',
                    name=name+'_ed2',
                    trainable=trainable)(x_e)#keep this simple!
        
        s_e = Dense(1,activation='sigmoid',
                    name=name+'_ed3',trainable=trainable)(x_e)#edge classifier
        #loss
        s_e = LLEdgeClassifier(
            print_loss=False,
            active=trainable,
            record_metrics=record_metrics,
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
        record_metrics = False,#can be picked up by metrics layers later
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
    
    return x,coords,energy, rs, bg, t_idx, t_spectator_weight    
    
    
    
    
    
def pre_selection_model_full(orig_inputs,
                             debug_outdir='',
                             trainable=False,
                             name='pre_selection',
                             debugplots_after=-1,
                             reduction_threshold=0.75,
                             noise_threshold = 0.025,
                             use_edges=True,
                             n_coords=3,
                             pass_through=False,
                             print_info=False,
                             record_metrics=False,
                             omit_reduction=False, #only trains coordinate transform. useful for pretrain phase
                             use_multigrav=True
                             ):
    
    from GravNetLayersRagged import AccumulateNeighbours, SelectFromIndices
    from GravNetLayersRagged import SortAndSelectNeighbours, NoiseFilter
    from GravNetLayersRagged import CastRowSplits, ProcessFeatures
    from GravNetLayersRagged import GooeyBatchNorm, MaskTracksAsNoise
    from DebugLayers import PlotCoordinates
    from LossLayers import LLClusterCoordinates, LLNotNoiseClassifier, LLFillSpace
    from MetricsLayers import MLReductionMetrics
    
    rs = CastRowSplits()(orig_inputs['row_splits'])
    t_idx = orig_inputs['t_idx']
    
        

    orig_processed_features = ProcessFeatures()(orig_inputs['features'])
    x = orig_processed_features
    energy = SelectFeatures(0, 1)(orig_inputs['features'])
    coords = SelectFeatures(5, 8)(x)
    track_charge = SelectFeatures(2,3)(orig_inputs['features']) #zero for calo hits
    phys_coords = coords

    # here the actual network starts
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,outdir=debug_outdir,name=name+'_initial')([coords,energy,t_idx,rs])
    ############## Keep this part to reload the noise filter with pre-trained weights for other trainings
    
    out={}
    if pass_through: #do nothing but make output compatible
        for k in orig_inputs.keys():
            out[k] = orig_inputs[k]
        out['features'] = x
        out['coords'] = coords
        out['addfeat'] = x #add more
        out['energy'] = energy
        out['not_noise_score']=Dense(1,name=name+'_passthrough_noise')(x)
        out['orig_t_idx'] = orig_inputs['t_idx']
        out['orig_t_energy'] = orig_inputs['t_energy'] #for validation
        out['orig_dim_coords'] = coords
        out['rs']=rs
        out['orig_row_splits'] = rs
        return out
    
    #this takes O(200ms) for 100k hits
    coords,nidx,dist, x = first_coordinate_adjustment(
        coords, x, energy, rs, t_idx, 
        debug_outdir,
        trainable=trainable,
        name=name+'_first_coords',
        debugplots_after=debugplots_after,
        n_coords=n_coords,
        record_metrics=record_metrics,
        use_multigrav=use_multigrav
        )
    #create the gradients
    coords = LLClusterCoordinates(
        print_loss = trainable and print_info,
        active = trainable,
        print_batch_time=False,
        record_metrics = record_metrics,
        scale=5.
        )([coords,t_idx,rs])
    
    
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
    
    do not cluster tracks with anything here
    '''
    
    cluster_tidx = MaskTracksAsNoise(active=trainable)([t_idx,track_charge])
    
    unred_rs = rs
    gnidx, gsel, group_backgather, rs = reduce_indices(x,dist, nidx, rs, cluster_tidx, 
           threshold = reduction_threshold,
           print_reduction=print_info,
           trainable=trainable,
           name=name+'_reduce_indices',
           use_edges = use_edges,
           record_metrics=record_metrics,
           return_backscatter=False)
    
    
    gsel = MLReductionMetrics(
        name=name+'_reduction_0',
        record_metrics = record_metrics
        )([gsel,t_idx,orig_inputs['t_energy'],unred_rs,rs])
    
    #do it explicitly
    
    #selfeat = orig_inputs['features']
    selfeat = SelectFromIndices()([gsel,orig_processed_features])
    unproc_features = SelectFromIndices()([gsel,orig_inputs['features']])
    
    #save for later
    orig_dim_coords = coords
    
    x = AccumulateNeighbours('minmeanmax')([x, gnidx])
    x = SelectFromIndices()([gsel,x])
    #add more useful things
    coords = AccumulateNeighbours('mean')([coords, gnidx])
    coords = SelectFromIndices()([gsel,coords])
    
    phys_coords = AccumulateNeighbours('mean')([phys_coords, gnidx])
    phys_coords = SelectFromIndices()([gsel,phys_coords])
    
    energy = AccumulateNeighbours('sum')([energy, gnidx])
    energy = SelectFromIndices()([gsel,energy])
    
    #re-build standard feature layout
    out['features'] = selfeat
    out['unproc_features'] = unproc_features
    out['coords'] = coords
    out['phys_coords'] = phys_coords
    out['addfeat'] = GooeyBatchNorm(trainable=trainable)(x) #norm them
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
    
    
    ######## below is noise classifier
    
    
    #this does not work, but also might not be an issue for the studies        
    #out['backscatter']=bg

    isnotnoise = Dense(1, activation='sigmoid',
                       trainable=trainable,
                       name=name+'_noisescore_d1',
                       )(Concatenate()([out['addfeat'],out['coords']]))
    isnotnoise = LLNotNoiseClassifier(
        print_loss=trainable and print_info,
        scale=1.,
        active=trainable,
        record_metrics=record_metrics,
        )([isnotnoise, out['t_idx']])

    unred_rs = rs
    sel, rs, noise_backscatter = NoiseFilter(
        threshold = noise_threshold, #high signal efficiency filter
        print_reduction=print_info,
        record_metrics=record_metrics
        )([isnotnoise,rs])
        
        
    out['not_noise_score']=isnotnoise
        
    for k in out.keys():
        out[k] = SelectFromIndices()([sel,out[k]])
        
    
    out['coords'] = LLFillSpace(
        print_loss = trainable and print_info,
        active = trainable,
        record_metrics = record_metrics,
        scale=0.1,#just mild
        runevery=-1, #give it a kick only every now and then - hat's enough
        )([out['coords'],rs])
        
    
    out['scatterids'] = [group_backgather, noise_backscatter] #add them here directly
    out['orig_t_idx'] = orig_inputs['t_idx']
    out['orig_t_energy'] = orig_inputs['t_energy'] #for validation
    out['orig_dim_coords'] = orig_dim_coords
    out['rs']=rs
    out['orig_row_splits'] = orig_inputs['row_splits']
    
    '''
    So we have the following outputs at this stage:
    
    out['group_backgather']
    out['noise_backscatter_N']
    out['noise_backscatter_idx']
    out['orig_t_idx'] 
    out['orig_t_energy'] 
    out['orig_dim_coords']
    out['rs']
    out['orig_row_splits']
    
    out['features'] 
    out['unproc_features']
    out['coords'] 
    out['addfeat']
    out['energy']
    
    '''

    return out
    
def pre_selection_staged(indict,
                         debug_outdir,
                         trainable,
                         name='pre_selection_add_stage_0',
                         debugplots_after=-1,
                         reduction_threshold=0.75,
                         use_edges=True,
                         print_info=False,
                         record_metrics=False,
                         n_coords=3,
                         edge_nodes_0 = 16,
                         edge_nodes_1 = 8,
                         ):
    '''
    Takes the output of the preselection model and selects again :)
    But the outputs are compatible, this one can be chained
    
    This one uses full blown GravNet
    
    Gets as inputs:
    
    indict['scatterids']
    indict['orig_t_idx'] 
    indict['orig_t_energy'] 
    indict['orig_dim_coords']
    indict['rs']
    indict['orig_row_splits']
    
    indict['features'] 
    indict['orig_features']
    indict['coords'] 
    indict['addfeat']
    indict['energy']
    
    
    indict['t_idx']
    indict['t_energy']
    ... all the truth info
    
    '''
    
    from GravNetLayersRagged import RaggedGravNet, DistanceWeightedMessagePassing, ElementScaling
    from GravNetLayersRagged import SelectFromIndices, GooeyBatchNorm, MaskTracksAsNoise
    from GravNetLayersRagged import AccumulateNeighbours, KNN, MultiAttentionGravNetAdd
    from LossLayers import LLClusterCoordinates
    from DebugLayers import PlotCoordinates
    from MetricsLayers import MLReductionMetrics
    from Regularizers import MeanMaxDistanceRegularizer, AverageDistanceRegularizer
    
    
    #assume the inputs are normalised
    rs = indict['rs']
    t_idx = indict['t_idx']
    
    
    track_charge = SelectFeatures(2,3)(indict['unproc_features']) #zero for calo hits
    x = Concatenate()([indict['features'] , indict['addfeat']])
    x = Dense(64,activation='elu',trainable=trainable)(x)
    gn_pre_coords = indict['coords']
    gn_pre_coords =  ElementScaling(name=name+'es1',
                                    trainable=trainable)(gn_pre_coords) 
    x = Concatenate()([gn_pre_coords , x])
    
    x, coords, nidx, dist = RaggedGravNet(n_neighbours=32,
                                                 n_dimensions=n_coords,
                                                 n_filters=64,
                                                 n_propagate=64,
                                                 coord_initialiser_noise=1e-5,
                                                 feature_activation=None,
                                                 record_metrics=record_metrics,
                                                 use_approximate_knn=True,
                                                 use_dynamic_knn=True,
                                                 trainable=trainable,
                                                 name = name+'_gn1'
                                                 )([x, rs])
    
    #the two below are mostly running to record metrics and kill very bad coordinate scalings 
    dist = MeanMaxDistanceRegularizer(
        strength=1e-6 if trainable else 0.,
        record_metrics = record_metrics
        )(dist)
        
    dist = AverageDistanceRegularizer(
        strength=1e-6 if trainable else 0.,
        record_metrics = record_metrics
        )(dist)
        
    
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,
                                        outdir=debug_outdir,name=name+'_gn1_coords')(
                                            [coords,
                                             indict['energy'],
                                             t_idx,rs])
    
    x = DistanceWeightedMessagePassing([32,32,8,8], 
                                       name=name+'dmp1',
                                       trainable=trainable)([x,nidx,dist])
    
    x_matt = Dense(16,activation='elu',name=name+'_matt_dense')(x)
                                       
    x_matt = MultiAttentionGravNetAdd(5,name=name+'_att_gn1',record_metrics=record_metrics)([x,x_matt,coords,nidx])
    x = Concatenate()([x,x_matt]) 
    x = Dense(64,activation='elu',name=name+'_bef_coord_dense')(x)
    
    coords = Add()([Dense(n_coords,
                          name=name+'_coord_add_dense',
                          kernel_initializer='zeros')(x),coords])
    if debugplots_after>0:
        coords = PlotCoordinates(debugplots_after,
                                    outdir=debug_outdir,name=name+'_red_coords')(
                                        [coords,
                                         indict['energy'],
                                         t_idx,rs])
                                    
    nidx,dist = KNN(K=16,radius='dynamic', #use dynamic feature
                record_metrics=record_metrics,
                name=name+'_knn',
                min_bins=[20,20] #this can be fine grained
                )([coords,rs])
    
                                       
    coords = LLClusterCoordinates(
        print_loss = print_info,
        record_metrics=record_metrics,
        active = trainable,
        print_batch_time=False,
        scale=5.
        )([coords,t_idx,rs])
        
    unred_rs=rs
    
    cluster_tidx = MaskTracksAsNoise(active=trainable)([t_idx,track_charge])
    
    gnidx, gsel, group_backgather, rs = reduce_indices(x,dist, nidx, rs, cluster_tidx, 
           threshold = reduction_threshold,
           print_reduction=print_info,
           trainable=trainable,
           name=name+'_reduce_indices',
           use_edges = use_edges,
           edge_nodes_0=edge_nodes_0,
           edge_nodes_1=edge_nodes_1,
           return_backscatter=False)
    
    
    gsel = MLReductionMetrics(
        name=name+'_reduction',
        record_metrics = True
        )([gsel,t_idx,indict['t_energy'],unred_rs,rs])
    
    
    selfeat = SelectFromIndices()([gsel,indict['features']])
    unproc_features = SelectFromIndices()([gsel,indict['unproc_features']])
    
    
    x = AccumulateNeighbours('minmeanmax')([x, gnidx])
    x = SelectFromIndices()([gsel,x])
    #add more useful things
    coords = AccumulateNeighbours('mean')([coords, gnidx])
    coords = SelectFromIndices()([gsel,coords])
    energy = AccumulateNeighbours('sum')([indict['energy'], gnidx])
    energy = SelectFromIndices()([gsel,energy])
    
    
    out={}
    out['not_noise_score'] = AccumulateNeighbours('mean')([indict['not_noise_score'], gnidx]) 
    out['not_noise_score'] = SelectFromIndices()([gsel,out['not_noise_score']])
    
    out['scatterids'] = indict['scatterids']+[group_backgather] #append new selection 
    
    #re-build standard feature layout
    out['features'] = selfeat
    out['unproc_features'] = unproc_features
    out['coords'] = coords
    out['addfeat'] = GooeyBatchNorm(
        name=name+'_gooey_norm',
        trainable=trainable)(x) #norm them
    out['energy'] = energy
    out['rs'] = rs
    
    for k in indict.keys():
        if 't_' == k[0:2]:
            out[k] = SelectFromIndices()([gsel,indict[k]])
    
    #some pass throughs:
    out['orig_dim_coords'] = indict['orig_dim_coords']
    out['orig_t_idx'] = indict['orig_t_idx']
    out['orig_t_energy'] = indict['orig_t_energy']
    out['orig_row_splits'] = indict['orig_row_splits']
    
    #check
    anymissing=False
    for k in indict.keys():
        if not k in out.keys():
            anymissing=True
            print(k, 'missing')
    if anymissing:
        raise ValueError("key not found")
        
    return out


def N_pre_selection_staged(N,
                           indict,
                           debug_outdir,
                           trainable,
                           name='pre_selection_stage_',
                           debugplots_after=-1,
                           reduction_threshold=0.75,
                           use_edges=True,
                           print_info=False,
                           record_metrics=False,
                           n_coords=3,
                           edge_nodes_0 = 16,
                           edge_nodes_1 = 8):
    
    for i in range(N):
        indict=pre_selection_staged(indict,
                               debug_outdir=debug_outdir,
                               trainable=trainable,
                               name=name+str(i),
                               debugplots_after=debugplots_after,
                               reduction_threshold=reduction_threshold,
                               use_edges=use_edges,
                               print_info=print_info,
                               record_metrics=record_metrics,
                               n_coords=n_coords,
                               edge_nodes_0 = edge_nodes_0,
                               edge_nodes_1 = edge_nodes_1,
                               )
    return indict

def re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=False
        ):
    '''
    To be called after OC loss is applied to pre-selected outputs to bring it all back to the full dimensionality
    all hits that have been selected before and cannot be backgathered will be assigned coordinates far away,
    and a zero beta value.
    
    This is the counterpart of the pre_selection_model_full.
    
    returns full suite
    ('pred_beta', pred_beta), 
    ('pred_ccoords', pred_ccoords),
    ('pred_energy_corr_factor', pred_energy_corr),
    ('pred_pos', pred_pos),
    ('pred_time', pred_time),
    ('pred_id', pred_id),
    ('pred_dist', pred_dist),
    ('row_splits', row_splits)
    '''
    from GravNetLayersRagged import MultiBackScatterOrGather
    from globals import cluster_space as  cs
    
    scatterids = pre_selection['scatterids']
    pred_ccoords = MultiBackScatterOrGather(default=cs.noise_coord)([pred_ccoords, scatterids])#set it far away for noise
    pred_beta = MultiBackScatterOrGather(default=0.)([pred_beta, scatterids])
    pred_energy_corr = MultiBackScatterOrGather(default=1.)([pred_energy_corr, scatterids])
    pred_pos = MultiBackScatterOrGather(default=0.)([pred_pos, scatterids])
    pred_time = MultiBackScatterOrGather(default=10.)([pred_time, scatterids])
    pred_id = MultiBackScatterOrGather(default=0.)([pred_id, scatterids])
    pred_dist = MultiBackScatterOrGather(default=1.)([pred_dist, scatterids])
    
    if dict_output:
        return {
            'pred_beta': pred_beta, 
            'pred_ccoords': pred_ccoords,
            'pred_energy_corr_factor': pred_energy_corr,
            'pred_pos': pred_pos,
            'pred_time': pred_time,
            'pred_id': pred_id,
            'pred_dist': pred_dist,
            'row_splits': pre_selection['orig_row_splits'] }
        
    return [
        ('pred_beta', pred_beta), 
        ('pred_ccoords', pred_ccoords),
        ('pred_energy_corr_factor', pred_energy_corr),
        ('pred_pos', pred_pos),
        ('pred_time', pred_time),
        ('pred_id', pred_id),
        ('pred_dist', pred_dist),
        ('row_splits', pre_selection['orig_row_splits'])]
    
    




    
