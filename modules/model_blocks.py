
from tensorflow.keras.layers import Dropout, Dense, Concatenate, BatchNormalization, Add, Multiply, LeakyReLU
from Layers import OnesLike
from DeepJetCore.DJCLayers import  SelectFeatures, ScalarMultiply, StopGradient
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from Initializers import EyeInitializer
from GravNetLayersRagged import CondensateToIdxs

from datastructures.TrainData_NanoML import n_id_classes

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
def create_outputs(x, n_ccoords=3, 
                   n_classes=n_id_classes,
                   fix_distance_scale=False,
                   energy_factor=True,
                   name_prefix="output_module"):
    '''
    returns pred_beta, pred_ccoords, pred_energy, pred_energy_low_quantile,pred_energy_high_quantile,pred_pos, pred_time, pred_id
    '''
    
    pred_beta = Dense(1, activation='sigmoid',name = name_prefix+'_beta')(x)
    pred_ccoords = Dense(n_ccoords,
                         #this initialisation is much better than standard glorot
                         kernel_initializer=EyeInitializer(stddev=0.001),
                         use_bias=False,
                         name = name_prefix+'_clustercoords'
                         )(x) #bias has no effect
    

    energy_act=None
    if energy_factor:
        energy_act='elu'
    energy_res_act = None 
    pred_energy = Dense(1,name = name_prefix+'_energy',
                        kernel_initializer='zeros',
                        activation=energy_act
                        )(ScalarMultiply(0.01)(x))
                        
    if energy_factor:
        pred_energy = Add(name= name_prefix+'_one_plus_energy')([OnesLike()(pred_energy),pred_energy])    
                        
    pred_energy_low_quantile = Dense(1,name = name_prefix+'_energy_low_quantile',
                                     kernel_initializer='zeros',
                        activation=energy_res_act)(x)
    
    pred_energy_high_quantile = Dense(1,name = name_prefix+'_energy_high_quantile',
                                      kernel_initializer='zeros',
                        activation=energy_res_act)(x)
    
    pred_pos =  Dense(2,use_bias=False,name = name_prefix+'_pos')(x)
    
    pred_time = Dense(1,name=name_prefix + '_time_proxy')(ScalarMultiply(0.01)(x))
    pred_time = Add(
        name=name_prefix + '_time'
        )([ScalarMultiply(10.)(OnesLike()(pred_time)),pred_time])
    
    pred_time_unc = Dense(1,activation='elu',name = name_prefix+'_time_unc')(ScalarMultiply(0.01)(x))
    pred_time_unc = Add()([pred_time_unc, OnesLike()(pred_time_unc)])#strict positive with small turn on
    
    pred_id = Dense(n_classes, activation="softmax",name = name_prefix+'_class')(x)
    
    pred_dist = OnesLike()(pred_time)
    if not fix_distance_scale:
        pred_dist = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name = name_prefix+'_dist')(x))
        #this needs to be bound otherwise fully anti-correlated with coordates scale
    return pred_beta, pred_ccoords, pred_dist, pred_energy, pred_energy_low_quantile, pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id




from GravNetLayersRagged import MultiBackScatterOrGather
def re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=False,
        is_preselected_dataset=False,
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
    ('pred_energy_low_quantile', pred_energy_low_quantile),
    ('pred_energy_high_quantile', pred_energy_high_quantile),
    ('pred_pos', pred_pos),
    ('pred_time', pred_time),
    ('pred_id', pred_id),
    ('pred_dist', pred_dist),
    ('row_splits', row_splits)
    '''
    from globals import cluster_space as  cs
    
    #this is only true if the preselection is run in situ - no preselected data set
    if 'scatterids' in pre_selection.keys():
        scatterids = pre_selection['scatterids']
        pred_ccoords = MultiBackScatterOrGather(default=cs.noise_coord)([pred_ccoords, scatterids])#set it far away for noise
        pred_beta = MultiBackScatterOrGather(default=0.)([pred_beta, scatterids])
        pred_energy_corr = MultiBackScatterOrGather(default=1.)([pred_energy_corr, scatterids])
        pred_energy_low_quantile = MultiBackScatterOrGather(default=1.)([pred_energy_low_quantile, scatterids])
        pred_energy_high_quantile = MultiBackScatterOrGather(default=1.)([pred_energy_high_quantile, scatterids])
        pred_pos = MultiBackScatterOrGather(default=0.)([pred_pos, scatterids])
        pred_time = MultiBackScatterOrGather(default=10.)([pred_time, scatterids])
        pred_id = MultiBackScatterOrGather(default=0.)([pred_id, scatterids])
        pred_dist = MultiBackScatterOrGather(default=1.)([pred_dist, scatterids])
        
        rechit_energy = ScalarMultiply(0.)(pred_beta) #FIXME, will duplicate.
    
    row_splits = None
    if is_preselected_dataset:
        row_splits = pre_selection['row_splits']
        rechit_energy = pre_selection['rechit_energy']
    else:
        row_splits = pre_selection['orig_row_splits']
        rechit_energy = pre_selection['rechit_energy'] #FIXME if not included here, return statement will fail, probably should be moved outside of if-statement
    
    if dict_output:
        return {
            'pred_beta': pred_beta, 
            'pred_ccoords': pred_ccoords,
            'pred_energy_corr_factor': pred_energy_corr,
            'pred_energy_low_quantile': pred_energy_low_quantile,
            'pred_energy_high_quantile': pred_energy_high_quantile,
            'pred_pos': pred_pos,
            'pred_time': pred_time,
            'pred_id': pred_id,
            'pred_dist': pred_dist,
            'rechit_energy': rechit_energy, #can also be summed if pre-selection
            'row_splits': row_splits }
        
    raise ValueError("only dict output")
    
    


from GravNetLayersRagged import AccumulateNeighbours, SelectFromIndices, SelectFromIndicesWithPad
from GravNetLayersRagged import SortAndSelectNeighbours, NoiseFilter
from GravNetLayersRagged import CastRowSplits, ProcessFeatures
from GravNetLayersRagged import ScaledGooeyBatchNorm, GooeyBatchNorm, Where, MaskTracksAsNoise
from LossLayers import LLClusterCoordinates, AmbiguousTruthToNoiseSpectator, LLNotNoiseClassifier, LLBasicObjectCondensation, LLFillSpace, LLEdgeClassifier
from MetricsLayers import MLReductionMetrics, SimpleReductionMetrics, OCReductionMetrics
from Layers import CreateTruthSpectatorWeights

from tensorflow.keras.layers import Flatten, Reshape

from GravNetLayersRagged import NeighbourGroups,GroupScoreFromEdgeScores,ElementScaling, EdgeSelector, KNN, DistanceWeightedMessagePassing, RecalcDistances, MultiAttentionGravNetAdd
from DebugLayers import PlotCoordinates, PlotEdgeDiscriminator, PlotNoiseDiscriminator
    
from GravNetLayersRagged import XYZtoXYZPrime, CondensatesToPseudoRS, ReversePseudoRS, AssertEqual, CleanCondensations, CreateMask
from LossLayers import LLGoodNeighbourHood, LLOCThresholds, LLKnnPushPullObjectCondensation, LLKnnSimpleObjectCondensation
from LossLayers import NormaliseTruthIdxs
#also move this to the standard pre-selection  model
def condition_input(orig_inputs):
    
    if not 't_spectator_weight' in orig_inputs.keys(): #compat layer
        orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=5.,minimum=1e-1,active=True
                                                         )([orig_inputs['t_spectator'], 
                                                            orig_inputs['t_idx']])
        orig_inputs['t_spectator_weight'] = orig_t_spectator_weight
        
        
    if not 'is_track' in orig_inputs.keys():
        orig_inputs['is_track'] = SelectFeatures(2,3)(orig_inputs['features'])
        
    if not 'rechit_energy' in orig_inputs.keys():
        orig_inputs['rechit_energy'] = SelectFeatures(0, 1)(orig_inputs['features'])    
    
    processed_features =  orig_inputs['features']  
    orig_inputs['orig_features'] = orig_inputs['features']  
    
    #get some things to work with    
    orig_inputs['row_splits'] = CastRowSplits()(orig_inputs['row_splits'])
    orig_inputs['orig_row_splits'] = orig_inputs['row_splits'] 
    
    #coords have not been built so features not processed, so this is the first time this is called
    if not 'coords' in orig_inputs.keys():
        processed_features = ProcessFeatures(name='precondition_process_features')(orig_inputs['features'])
        orig_inputs['coords'] = SelectFeatures(5, 8)(processed_features)
        orig_inputs['features'] = processed_features
        
        #create starting point for cluster coords
        orig_inputs['prime_coords'] = XYZtoXYZPrime()(SelectFeatures(5, 8)(orig_inputs['orig_features']))
    
    return orig_inputs
    

def expand_coords_if_needed(coords, x, ndims, name, trainable):
    if coords.shape[-1] == ndims:
        return coords
    if coords.shape[-1] > ndims:
        raise ValueError("only expanding coordinates")
    return Concatenate()([ coords, Dense(ndims-coords.shape[-1],
                                         kernel_initializer='zeros',
                                         name=name,
                                         trainable=trainable)(x) ])



def mini_gravnet_block(K, x, coords, rs, trainable, record_metrics, name):
    
    x_in = x
    knncoords = ElementScaling(name=name+'gn_es1',trainable=trainable)(coords)                                    
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([knncoords,rs])#hard code it here, this is optimised given our datasets
    
    x = Dense(64,activation='elu',name=name+'dense0',trainable=trainable)(x)
    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])# hops are rather light
    
    return Concatenate()([x, x_in])

def mini_noise_block(sel, noise_threshold, trainable, record_metrics, name):
    
    isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,
                           name=name+'_noisescore_d1',
                               )(sel['x'])
        
    
    
    isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
        scale=1.,
        purity_metrics_threshold = noise_threshold,
        record_efficiency=True
        )([isnotnoise, sel['t_idx']])
    
    isnotnoise = Where(outputval=1.,condition='!=0')([sel['is_track'], isnotnoise])
    sel['isnotnoise_score'] = isnotnoise
       
                                    
    no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(
        threshold = noise_threshold,
        record_metrics=record_metrics
        )([sel['isnotnoise_score'],sel['row_splits']])
        
    for k in sel.keys():
        sel[k] = SelectFromIndices()([no_noise_sel,sel[k]]) #also  good check, will fail if dimensions don't match
    
    sel['row_splits'] = no_noise_rs
    sel['noise_backscatter'] = noise_backscatter
    
    #this has removed hits, rebuild regular truth indices
    sel['t_idx'] = NormaliseTruthIdxs()([sel['t_idx'], sel['row_splits'] ])
    
    return sel
    
    
def mini_pre_condensation_model(inputs,
                           record_metrics=False,
                           trainable=False,
                           t_d=0.05,
                           t_b=0.9,
                           q_min=0.2,
                           purity_target=0.95,
                           print_batch_time=False,
                           name='pre_condensation',
                           condensation_mode = 'std',
                           noise_threshold=0.15,
                           cleaning_threshold=0.1,
                           cluster_dims=3,
                           condensate=True,
                           debug_outdir='',
                           debugplots_after=-1,
                           publishpath=None
                           ):    
    
    K = 12
    
    orig_inputs = condition_input(inputs)
    
    #return {'coords': orig_inputs['prime_coords']}

    sel = orig_inputs.copy()
    sel['x'] = mini_gravnet_block(K, orig_inputs['features'], 
                                  orig_inputs['prime_coords'], 
                                  orig_inputs['row_splits'], 
                                  trainable, record_metrics, name)
    
    ## REMOVE NOISE ##
    if noise_threshold > 0:
        sel = mini_noise_block(sel, noise_threshold, trainable, record_metrics, name)
    
    sel['x'] = Dense(64, activation='elu', name = name+'seld1')(sel['x'])
    ### object condensation part
    beta = Dense(1, activation='sigmoid',name=name+'dense_beta')(sel['x'])
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(sel['x'])) 
    
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(sel['x'])
    ccoords = ScalarMultiply(0.01)(ccoords)
    
    prime_coords = ElementScaling(name=name+'es2',trainable=trainable)(sel['prime_coords'])
    prime_coords = expand_coords_if_needed(prime_coords, sel['x'],
                                             cluster_dims, name+'ccoords_exp',
                                             trainable=trainable)
    
    ccoords = Add()([prime_coords, ccoords])
    
    beta = LLBasicObjectCondensation(
           q_min=q_min,
           implementation = condensation_mode,
           print_batch_time = print_batch_time,
           record_batch_time = record_metrics,
           active=trainable,
           record_metrics = record_metrics,
           use_average_cc_pos=0.5
        )([beta, ccoords, d,sel['t_spectator_weight'], 
                          sel['t_idx'], sel['row_splits']])
    
    out = sel
    out['orig_rowsplits'] = orig_inputs['row_splits']
    out['beta'] = beta
    out['features'] = Concatenate()([sel['features'],sel['x']])
    
    
    #track or masked
    #no_condensation_mask = tf.keras.layers.Maximum()([no_condensation_mask, sel['is_track']])
    no_condensation_mask = sel['is_track']
    
    ch_idx, c_idx, _, revflat, asso_idx, dyn_t_d, dyn_t_b = RaggedCreateCondensatesIdxs(t_d=t_d,t_b=t_b, 
                                       active = condensate, 
                                       keepnoise = True,
                                       return_thresholds = True,
                                       trainable = trainable,
                                       record_metrics = record_metrics,
                                       name = name+'_r_condensation')([beta, ccoords, d, 
                                                                     no_condensation_mask, 
                                                                     sel['row_splits']])
    
    out['cond_idx'] = OCReductionMetrics(name = name+'_metric',        
        record_metrics=record_metrics
        )([asso_idx,  sel['t_idx']])
    
    if False:    
        out['beta'] = LLOCThresholds(
                   name=name+'_ll_oc_thresholds',
                   active=trainable,
                   highest_t_d = 1.,
                   print_batch_time = print_batch_time,
                   record_batch_time = record_metrics,
                   lowest_t_b = 0.7,
                   purity_target = purity_target,
                   record_metrics=record_metrics)([beta, out['cond_idx'], d, 
                                                   ccoords , sel['t_idx'], dyn_t_d, dyn_t_b])
    
    else:
        # betas, coords, d, c_idx, ch_idx, energy, t_idx, t_depe, t_d, t_b
        out['beta'] = LLFullOCThresholds(
            name=name+'_ll_full_oc_thresholds',
            active=trainable,
            purity_weight = 0.9,
            record_metrics=record_metrics,
             )([beta, ccoords, d, c_idx, ch_idx, 
                sel['rechit_energy'], sel['t_idx'], 
                sel['t_rec_energy'], dyn_t_d, dyn_t_b])
    
                                       
    if debugplots_after>0:
        out['ccoords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords',
                                        publish=publishpath)(
                                            [ccoords,
                                             beta, 
                                             d,#additional features to add as hover data
                                             sel['t_idx'],sel['row_splits']])
        
                                        
        out['dummy'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords_cleaning',
                                        publish=publishpath)(
                                            [sel['coords'],
                                             sel['rechit_energy'], 
                                             sel['t_idx'],#additional features to add as hover data
                                             asso_idx,sel['row_splits']])
    
    

    return out
        
def pre_condensation_model(inputs,
                           record_metrics=False,
                           trainable=False,
                           t_d=0.05,
                           t_b=0.9,
                           q_min=0.2,
                           purity_target=0.95,
                           print_batch_time=False,
                           name='pre_condensation',
                           condensation_mode = 'std',
                           noise_threshold=0.15,
                           cleaning_threshold=0.1,
                           cluster_dims=3,
                           condensate=True,
                           debug_outdir='',
                           debugplots_after=-1,
                           publishpath=None
                           ):    
    
    K = 12
    
    orig_inputs = condition_input(inputs)
    
    coords = orig_inputs['prime_coords']
    energy = orig_inputs['rechit_energy']
    
    if debugplots_after>0:
        coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_prime_coords',
                                        publish=publishpath)(
                                            [coords,
                                             energy, 
                                             orig_inputs['t_idx'],orig_inputs['row_splits']])
    
    
    
    is_track = orig_inputs['is_track']
    x = orig_inputs['features']
    
    x = ScaledGooeyBatchNorm(record_metrics=record_metrics, 
                             fluidity_decay = 1e-1, #very fast decay
                             trainable=trainable, name=name+'_gooey_0')(x)
    
    rs = orig_inputs['row_splits']
    
    ##### one simplified gravnet run
    
    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([coords,rs])#hard code it here, this is optimised given our datasets
    
    ### this is the "meat" of the model
    
    x = Dense(32,activation='elu',name=name+'dense0',trainable=trainable)(x)
    
    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])# hops are rather light 
    
    x = ScaledGooeyBatchNorm(record_metrics=record_metrics,trainable=trainable, 
                             fluidity_decay = 1e-2, #fast decay
                             name=name+'_gooey_1')(x)
    
    x_skip = x
    x = Dense(64,activation='elu',name=name+'dense1a',trainable=trainable)(x)
    x = Dense(64,activation='elu',name=name+'dense1b',trainable=trainable)(x) 
    x = Dense(32,activation='elu',name=name+'dense1c',trainable=trainable)(x) 
    x = Concatenate()([x, x_skip])
    
    
    ## for fair comparison reasons add something similar to the edge block in the standard pre-selection model
    
    dist,nidx = SortAndSelectNeighbours(K=-1,descending=True)([dist,nidx,energy])
    #create reduction, very simple, this gets expanded with K! be careful
    x_dist = StopGradient()(dist)
    x = Concatenate()([x,x_dist])
    x_e = Dense(8,activation='elu',name=name+'dense_x_e',trainable=trainable)(x)
    x_e = SelectFromIndicesWithPad()([nidx, x_e])#this will be big
    x_e = Flatten()(x_e)
    x_e = Dense(32,activation='elu',name=name+'dense_flat_x_e_0',trainable=trainable)(x_e)
    x_e = Dense(32,activation='elu',name=name+'dense_flat_x_e_1',trainable=trainable)(x_e)
    
    x_e = ScaledGooeyBatchNorm(record_metrics=record_metrics, 
                             fluidity_decay = 1e-2, #fast decay
                             trainable=trainable , name=name+'_gooey_2')(x_e)
    
    x = Concatenate()([x, x_e]) #in this case just concat
    x = Dense(64, activation='elu',name=name+'dense_bef_lastmp',trainable=trainable)(x)
    
    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp2',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])
    x = Dense(64, activation='elu',name=name+'dense_last0',trainable=trainable)(x)
    x = Dense(64, activation='elu',name=name+'dense_last1',trainable=trainable)(x)
    
    x = ScaledGooeyBatchNorm(record_metrics=record_metrics, 
                             fluidity_decay = 1e-2, #fast decay
                             trainable=trainable , name=name+'_gooey_3')(x)
    
    sel = orig_inputs.copy()
    sel['x'] = x
    
    ## REMOVE NOISE ##
    
    if noise_threshold > 0:
    
        isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,
                           name=name+'_noisescore_d1',
                               )(sel['x'])
        
        isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
            scale=1.,
            purity_metrics_threshold = noise_threshold,
            record_efficiency=True
            )([isnotnoise, orig_inputs['t_idx']])
            
        #tracks are never noise here
        isnotnoise = Where(outputval=1.,condition='!=0')([is_track, isnotnoise])
                                        
        no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(threshold = noise_threshold,
                                                                   record_metrics=record_metrics
            )([isnotnoise,rs])
            
        
        for k in sel.keys():
            sel[k] = SelectFromIndices()([no_noise_sel,sel[k]]) #also  good check, will fail if dimensions don't match
        
        sel['row_splits'] = no_noise_rs
        
        # for later
        #scatterids = [noise_backscatter]
    
    
    ### object condensation part
    beta = Dense(1, activation='sigmoid',name=name+'dense_beta')(sel['x'])
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(sel['x'])) 
    
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(sel['x'])
    selcoordsscaled = ElementScaling(name=name+'es2',trainable=trainable)(sel['coords'])
    exporigcoords = expand_coords_if_needed(selcoordsscaled,sel['x'],
                                             cluster_dims, name+'ccoords_exp',
                                             trainable=trainable)
    ccoords = Add()([exporigcoords, ccoords])
    
    
    #ccoords = LLFillSpace(active = trainable,
    #                       scale=0.1,
    #                       runevery=20,
    #                       record_metrics=record_metrics)([ccoords, sel['row_splits'],sel['t_idx']])
    
    #beta = LLBasicObjectCondensation(
    #    q_min=q_min,
    #    implementation = condensation_mode,
    #    print_batch_time = print_batch_time,
    #    record_batch_time = record_metrics,
    #    #scale = 0.1,
    #    active=trainable,
    #    record_metrics = record_metrics,
    #    use_average_cc_pos=0.
    #    )([beta, ccoords, d,sel['t_spectator_weight'], 
    #                                    sel['t_idx'], sel['row_splits']])
    
    if condensation_mode == 'pushpull':
        beta = LLKnnPushPullObjectCondensation(q_min=q_min,
                                               mode='dippedsq',
                                           active=trainable,
                                           record_metrics = record_metrics)(
                                               [beta, ccoords, d, sel['t_idx'], sel['row_splits']])
    
    elif condensation_mode == 'simpleknn':
        beta = LLKnnSimpleObjectCondensation(active=trainable,
                                             name = name+'_simple_knn_oc',
                                           record_metrics = record_metrics)(
                                               [beta, ccoords, d, sel['t_idx'], sel['row_splits']]
                                               )
    
    else:
        beta = LLBasicObjectCondensation(
           q_min=q_min,
           implementation = condensation_mode,
           print_batch_time = print_batch_time,
           record_batch_time = record_metrics,
           #scale = 0.1,
           active=trainable,
           record_metrics = record_metrics,
           use_average_cc_pos=0.5
        )([beta, ccoords, d,sel['t_spectator_weight'], 
                                        sel['t_idx'], sel['row_splits']])
    
    out = sel
    out['orig_rowsplits'] = orig_inputs['row_splits']
    out['beta'] = beta
    out['features'] = Concatenate()([sel['features'],sel['x']])
    
    #confscore = Dense(1,activation='sigmoid',name=name+'dense_conf')(sel['x'])
    
    #out['confscore'] = LLGoodNeighbourHood(
    #    active=trainable,
    #    distscale = t_d/4.,
    #    #scale=3.,#as it samples
    #    sampling = 0.03, #2k sampling for 100k batch
    #    record_metrics = record_metrics)([confscore, ccoords, d, sel['t_idx'], sel['row_splits']])
        
    #no_condensation_mask = CreateMask(threshold = cleaning_threshold, 
    #                                   invert=True)(out['confscore'])
    
    #track or masked
    #no_condensation_mask = tf.keras.layers.Maximum()([no_condensation_mask, sel['is_track']])
    no_condensation_mask = sel['is_track']
    
    ch_idx, c_idx, _, revflat, asso_idx, dyn_t_d, dyn_t_b = RaggedCreateCondensatesIdxs(t_d=t_d,t_b=t_b, 
                                       active = condensate, 
                                       keepnoise = True,
                                       return_thresholds = True,
                                       trainable = trainable,
                                       record_metrics = record_metrics,
                                       name = name+'_r_condensation')([beta, ccoords, d, 
                                                                     no_condensation_mask, 
                                                                     sel['row_splits']])
    
    
    
    out['cond_idx'] = OCReductionMetrics(name = name+'_metric',        
        record_metrics=record_metrics
        )([asso_idx,  sel['t_idx']])
    
    if False:    
        beta = LLOCThresholds(
                   name=name+'_ll_oc_thresholds',
                   active=trainable,
                   highest_t_d = 1.,
                   print_batch_time = print_batch_time,
                   record_batch_time = record_metrics,
                   lowest_t_b = 0.7,
                   purity_target = purity_target,
                   record_metrics=record_metrics)([beta, out['cond_idx'], d, 
                                                   ccoords , sel['t_idx'], dyn_t_d, dyn_t_b])
    
    else:
        # betas, coords, d, c_idx, ch_idx, energy, t_idx, t_depe, t_d, t_b
        beta = LLFullOCThresholds(
            name=name+'_ll_full_oc_thresholds',
            active=trainable,
            purity_weight = 0.9,
            record_metrics=record_metrics,
             )([beta, ccoords, d, c_idx, ch_idx, 
                sel['rechit_energy'], sel['t_idx'], 
                sel['t_rec_energy'], dyn_t_d, dyn_t_b])
    
                                       
    if debugplots_after>0:
        ccoords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords',
                                        publish=publishpath)(
                                            [ccoords,
                                             beta, 
                                             d,#additional features to add as hover data
                                             sel['t_idx'],sel['row_splits']])
        
                                        
        out['dummy'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords_cleaning',
                                        publish=publishpath)(
                                            [sel['coords'],
                                             sel['rechit_energy'], 
                                             sel['t_idx'],#additional features to add as hover data
                                             asso_idx,sel['row_splits']])
    
    out['ccoords'] = ccoords

    
    return out
    
    
#make this much simpler
def pre_selection_model(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='pre_selection',
        debugplots_after=-1,
        reduction_threshold=0.8,#doesn't make a huge difference, this is higher purity than for layer clusters
        noise_threshold=0.1, #0.4 % false-positive, 96% noise removal
        K=12,
        record_metrics=True,
        filter_noise=True,
        double_knn=False,
        pass_through=False
        ):
    
    '''
    inputnames ['recHitFeatures', 'recHitFeatures_rowsplits', 't_idx', 't_idx_rowsplits', 't_energy', 't_energy_rowsplits', 't_pos', 't_pos_rowsplits', 
    't_time', 't_time_rowsplits', 't_pid', 't_pid_rowsplits', 
    't_spectator', 't_spectator_rowsplits', 't_fully_contained', 't_fully_contained_rowsplits', 
    't_rec_energy', 't_rec_energy_rowsplits', 't_is_unique', 't_is_unique_rowsplits']

    ['t_idx', 't_energy', 't_pos', 't_time', 't_pid', 't_spectator', 't_fully_contained', 
    't_rec_energy', 't_is_unique', 't_spectator_weight', 'coords', 'rechit_energy', 
    'features', 'is_track', 'row_splits', 'scatterids', 'orig_row_splits']
    '''
    
    orig_inputs = condition_input(orig_inputs)
    
    if pass_through:
        orig_inputs['orig_row_splits'] = orig_inputs['row_splits'] 
        return orig_inputs
    
    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    coords = orig_inputs['coords']
    is_track = orig_inputs['is_track']
    x = orig_inputs['features']
    
    #truth
    t_spec_w = orig_inputs['t_spectator_weight']
    t_idx = orig_inputs['t_idx']
    
    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)
    coords = LLClusterCoordinates(active = trainable,
        name = name+'_LLClusterCoordinates',record_metrics = record_metrics,
        scale=1.
        )([coords,t_idx,t_spec_w,energy, rs])
    
    
    
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([coords,rs])#hard code it here, this is optimised given our datasets
    
    ### this is the "meat" of the model
    
    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',trainable=trainable)([x,nidx,dist])# hops are rather light 
    x = Dense(32,activation='relu',name=name+'dense1a',trainable=trainable)(x)
    x = Dense(32,activation='relu',name=name+'dense1b',trainable=trainable)(x)   
    
    if double_knn:
        
        xc = Dense(3,name=name+'dense_xc_knn_2',trainable=trainable)(x) 
        coords = Add()([coords, xc])
        nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn_2',
                    min_bins=20)([coords,rs])  
                       
        coords = LLClusterCoordinates(
            record_metrics=record_metrics,
            #print_batch_time=True,
            record_batch_time=record_metrics,
            name = name+'_LLClusterCoordinates_coords_knn_2',
            active = trainable,
            scale=1.
            )([coords,t_idx,t_spec_w,energy,rs])   
            
        x = DistanceWeightedMessagePassing([32,32],name=name+'dmp2',trainable=trainable)([x,nidx,dist])# hops are rather light 
        x = Dense(32,activation='relu',name=name+'dense2a',trainable=trainable)(x)
        x = Dense(32,activation='relu',name=name+'dense2b',trainable=trainable)(x)  
    
    #sort by energy, descending (not that it matters really)
    dist,nidx = SortAndSelectNeighbours(K=-1,descending=True)([dist,nidx,energy])
    #create reduction, very simple, this gets expanded with K! be careful
    x_e = Dense(6,activation='relu',name=name+'dense_x_e',trainable=trainable)(x)
    x_e = SelectFromIndicesWithPad()([nidx, x_e])#this will be big
    x_e = Flatten()(x_e)
    x_e = Dense(4*K,activation='relu',name=name+'dense_flat_x_e',trainable=trainable)(x_e)
    x_e = Reshape((K,4))(x_e)
    x_e = Dense(1,activation='sigmoid',name=name+'_ed3',trainable=trainable)(x_e)#edge classifier    
    
    #not just where so that it ca be switched off without truth
    cluster_tidx = MaskTracksAsNoise(active=trainable)([t_idx,is_track])
    
    x_e = LLEdgeClassifier( name = name+'_LLEdgeClassifier',active=trainable,record_metrics=record_metrics,
            scale=5.#high scale
            )([x_e,nidx,cluster_tidx, t_spec_w, energy])    
    
    
    if debugplots_after > 0:
        x_e = PlotEdgeDiscriminator(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_edges')([x_e,nidx,cluster_tidx])
    
    sel_nidx = EdgeSelector(
            threshold=reduction_threshold
            )([x_e,nidx])
            
    sel_t_spec_w, sel_t_idx = AmbiguousTruthToNoiseSpectator(
        record_metrics=record_metrics)([sel_nidx, t_spec_w, t_idx, energy])
                 
    hierarchy = GroupScoreFromEdgeScores()([x_e,sel_nidx])
    
    g_sel_nidx, g_sel, group_backgather, g_sel_rs = NeighbourGroups(threshold = 1e-3,return_backscatter=False,
        record_metrics = False)([hierarchy, sel_nidx, rs])
    
    
    g_sel = MLReductionMetrics(
        name=name+'_reduction_0',
        record_metrics = record_metrics
        )([g_sel,t_idx,orig_inputs['t_energy'],rs,g_sel_rs])
    
    #safety, these row splits are obsolete now  
    rs = None
    
    # g_sel_sel_nidx: selected selected indices (first dimension output vertex multi)
    # g_sel_nidx: 'old' dimension, but with -1s where neighbours are clusters elsewhere
    g_sel_sel_nidx = SelectFromIndices()([g_sel,g_sel_nidx])
    
    #create selected output truth
    out={}
    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectFromIndices()([g_sel,orig_inputs[k]])
    
    #consider ambiguities
    out['t_idx'] = SelectFromIndices()([g_sel,sel_t_idx])
    out['t_spectator_weight'] = SelectFromIndices()([g_sel,sel_t_spec_w])
    
    ## create reduced features
    
    x_to_flat = Concatenate(name=name+'concat_x_to_flat')([orig_inputs['features'],x])
    x_to_flat = Dense(4, activation='relu',name=name+'flatten_dense')(x_to_flat)
    #this has output dimension now
    x_flat_o = SelectFromIndicesWithPad()([g_sel_sel_nidx, x_to_flat])
    x_flat_o = Flatten()(x_flat_o)
    
    x_o = AccumulateNeighbours('minmeanmax')([x, g_sel_nidx, energy])
    x_o = SelectFromIndices()([g_sel,x_o])
    x_o = Concatenate(name=name+'concat_x_o')([x_o,x_flat_o])
    x_o = GooeyBatchNorm(trainable=trainable)(x_o)
    
    #explicitly sum energy    
    energy_o = AccumulateNeighbours('sum')([energy, g_sel_nidx])
    energy_o = SelectFromIndices()([g_sel,energy_o])
    
    #build preselection coordinates
    coords_o = AccumulateNeighbours('mean')([coords, g_sel_nidx, energy])
    coords_o = SelectFromIndices()([g_sel,coords_o])
    
    #pass original features
    mean_orig_feat = AccumulateNeighbours('mean')([orig_inputs['orig_features'], g_sel_nidx, energy])
    mean_orig_feat = SelectFromIndices()([g_sel,mean_orig_feat])
    
    ## selection done work on selected ones
    
    coord_add_o = Dense(16,activation='relu',name=name+'dense_coord_add1')(x_o)
    coord_add_o = Dense(3,name=name+'dense_coord_add3')(coord_add_o)
    coords_o = Add()([coords_o,coord_add_o])
    
    #was not clustered
    is_track_o = SelectFromIndices()([g_sel,is_track])
    
    #create a gradient for this
    coords_o = LLClusterCoordinates(
        record_metrics=record_metrics,
        record_batch_time=record_metrics,
        name = name+'_LLClusterCoordinates_coords_o',
        active = trainable,
        scale=1.
        )([coords_o,out['t_idx'],out['t_spectator_weight'],energy_o,g_sel_rs])
    
    #coords_o = LLFillSpace(active = trainable,
    #                       scale=0.1,
    #                       record_metrics=record_metrics)([coords_o,g_sel_rs,out['t_idx']])
    
    #add to dict:
    out['coords'] = coords_o
    out['rechit_energy'] = energy_o
    out['features'] = x_o
    out['orig_features'] = mean_orig_feat
    out['is_track'] = is_track_o
    
    if debugplots_after>0:
        out['coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_after_red')(
                                            [out['coords'],
                                             out['rechit_energy'],
                                             out['t_idx'],g_sel_rs])
    
    ####### noise filter part #########
    
    scatterids=None
    
    if filter_noise:
    
        isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,name=name+'_noisescore_d1',
                           )(Concatenate(name=name+'concat_outf_outc')([out['features'],out['coords']]))
        
        #spectators are never noise here
        notnoisetruth = Where(outputval=1,condition='>0')([out['t_spectator_weight'], out['t_idx']])
        
        isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
            scale=1.)([isnotnoise, notnoisetruth])
            
        #tracks are never noise here
        isnotnoise = Where(outputval=1.,condition='>0')([out['is_track'], isnotnoise])
        
        if debugplots_after > 0:
            isnotnoise = PlotNoiseDiscriminator(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_noise_score')([isnotnoise,notnoisetruth])
                                        
        no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(threshold = noise_threshold,record_metrics=record_metrics
            )([isnotnoise,g_sel_rs])
        
        
        #select not-noise
        for k in out.keys():
            out[k] = SelectFromIndices()([no_noise_sel,out[k]]) #also  good check, will fail if dimensions don't match
        
        #safety, these row splits are obsolete now 
        g_sel_rs = None
    
        out['row_splits'] = no_noise_rs
        scatterids = [group_backgather, noise_backscatter]
        
    else:
        out['row_splits'] = g_sel_rs
        scatterids = [group_backgather]
        
    if 'scatterids' in orig_inputs.keys():
        out['scatterids'] = orig_inputs['scatterids'] + scatterids
    else:
        out['scatterids'] = scatterids
        
    if not 'orig_row_splits' in orig_inputs.keys():
        out['orig_row_splits'] = orig_inputs['row_splits']
    else:
        out['orig_row_splits'] = orig_inputs['orig_row_splits']#pass through
    
    # full reduction metric
    out['row_splits'] = SimpleReductionMetrics(
        name=name+'full_reduction',
        record_metrics=record_metrics
        )([out['row_splits'],orig_inputs['row_splits']])
        
        
    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(out['features'])
    d = ScalarMultiply(2.)(Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_d')(out['features']))
    if trainable:
        beta = LLBasicObjectCondensation(
            scale=0.1,
            active=trainable,
            record_batch_time = record_metrics,
            record_metrics = record_metrics,
            use_average_cc_pos=0.1,
            name=name+'_basic_object_condensation' 
            )([beta, out['coords'], d,out['t_spectator_weight'],out['t_idx'], out['row_splits']])    
    
    out['features'] = Concatenate()([out['features'],d,beta])
    
    return out
    
    
from RaggedLayers import RaggedCreateCondensatesIdxs, RaggedSelectFromIndices, RaggedMixHitAndCondInfo 
from RaggedLayers import RaggedCollapseHitInfo, RaggedDense, RaggedToFlatRS, FlatRSToRagged, ForceExec
    
from GravNetLayersRagged import RaggedGravNet   
from LossLayers import LLFullOCThresholds 
    
    
def tiny_intermediate_condensation(
        x, rs, energy, t_specweight, t_idx,
        trainable=True,
        record_metrics=True,
        cluster_dims=3,
        name='intermediate_condensation'
        ):
    
    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(x)
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(x))
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(x)
    
    beta = LLBasicObjectCondensation(
        q_min=.2,#high qmin here
        implementation = 'std',
        record_batch_time = record_metrics,
        scale = 0.2,
        active=trainable,
        print_batch_time = True,
        record_metrics = record_metrics,
        use_average_cc_pos=0.1 #small
        )([beta, ccoords, d,t_specweight,t_idx, rs])
        
    ch_idx, c_idx, rev_r, revflat,\
     asso_idx = RaggedCreateCondensatesIdxs(t_d = 0.25, t_b = 0.1,
                                                     return_thresholds=False)([beta,ccoords,d,rs])
                                                     
    c_x = RaggedSelectFromIndices()([x, c_idx])
    
    return RaggedSelectFromIndices()([c_x, rev_r])
    
        
def intermediate_condensation(
        x, rs, energy, t_specweight, t_idx,
        trainable=True,
        record_metrics=True,
        cluster_dims=3,
        name='intermediate_condensation'
        ):
    
    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(x)
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(x))
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(x)
    
    beta = LLBasicObjectCondensation(
        q_min=.2,#high qmin here
        implementation = 'std',
        record_batch_time = record_metrics,
        #scale = 0.1,
        active=trainable,
        print_batch_time = True,
        record_metrics = record_metrics,
        use_average_cc_pos=0.1 #small
        )([beta, ccoords, d,t_specweight,t_idx, rs])
    
    
    #low thresholds
    ch_idx, c_idx, _, revflat,\
     asso_idx = RaggedCreateCondensatesIdxs(t_d = 0.25, t_b = 0.1,
                                            collapse_noise=False,
                                                     return_thresholds=False)([beta,ccoords,d,rs])
    # t_d, t_b
    #if False:
    #    beta = LLOCThresholds(
    #               name=name+'_ll_oc_thresholds',
    #               active=trainable,
    #               highest_t_d = 1.,
    #               record_batch_time = record_metrics,
    #               lowest_t_b = 0.5,
    #               purity_target = 0.8,
    #               record_metrics=record_metrics)([beta, asso_idx, d, 
    #                                               ccoords , t_idx, t_d, t_b])
    #
    #
    #else:
    #    beta = LLFullOCThresholds(
    #        name=name+'_ll_full_oc_thresholds',
    #        active=trainable,
    #        purity_weight = 0.9,
    #         )([beta, ccoords, d, c_idx, ch_idx, 
    #            energy, t_idx, t_depe, rs, t_d, t_b])
    
    x = Dense(64, activation='elu', name=name+'_predense')(x)
    c_x = RaggedSelectFromIndices()([x, c_idx])
    ch_x = RaggedSelectFromIndices()([x, ch_idx])
    
    #print('c_x, ch_x',c_x.shape, ch_x.shape)
    c_x_skip_flat,_ = RaggedToFlatRS()(c_x)
    for c in ['mean','max']:
        ch_x = RaggedMixHitAndCondInfo('add')([ch_x, c_x])
        ch_x = RaggedDense(64, activation='elu', name=name+'_rdense1_'+c)(ch_x)
        c_x = RaggedCollapseHitInfo(c)(ch_x)
    
    #exchange through condensation points
    xf, xfrs = RaggedToFlatRS()(c_x)
    xf = Concatenate()([xf, c_x_skip_flat])

    if True:
        xf, _, gnnidx, gndist = RaggedGravNet(n_neighbours=32,
                                                     n_dimensions=3,
                                                     n_filters=64,
                                                     n_propagate=64,
                                                     record_metrics=True,
                                                     feature_activation='elu',
                                                     name=name+'_cpgn1',
                                                     debug=True
                                                     )([xf, xfrs])
        
        #xf = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
        #                                   trainable=trainable,
        #                                   activation='elu')([xf,gnnidx,gndist])

    xf = Dense(64, activation='elu',name=name+'_d1',trainable=trainable)(xf)
    # don't make it ragged again. no reason
    #c_x = FlatRSToRagged()([xf, xfrs])
    #
    #for c in ['mean','max']:
    #    ch_x = RaggedMixHitAndCondInfo('add')([ch_x, c_x])
    #    ch_x = RaggedDense(64, activation='elu', name=name+'_rdense2_'+c)(ch_x)
    #    c_x = RaggedCollapseHitInfo(c)(ch_x)
    #
    #c_x,_ = RaggedToFlatRS()(c_x)
    #print('c_x',c_x.shape)
    xf = Concatenate()([xf, c_x_skip_flat])
    #backgather to all hits, x is flat again
    x = RaggedSelectFromIndices()([xf, revflat])
    print('x',x.shape)
    #print('x out',x.shape, 'revflat',revflat.shape)
    return x #RaggedToFlatRS()(c_x)
    
    
    
