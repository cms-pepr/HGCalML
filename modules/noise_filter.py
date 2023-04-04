from tensorflow.keras.layers import Dense

from model_blocks import condition_input
from GravNetLayersRagged import ElementScaling
from GravNetLayersRagged import KNN, DistanceWeightedMessagePassing
from GravNetLayersRagged import Where
from GravNetLayersRagged import NoiseFilter
from GravNetLayersRagged import SelectFromIndices

from LossLayers import LLClusterCoordinates, LLNotNoiseClassifier
from DebugLayers import PlotNoiseDiscriminator
    
    
def noise_filter(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='noiseFilter',
        debugplots_after=-1,
        noise_threshold=0.1,
        K=12,
        record_metrics=True,
        pass_through=False,
        pf_mode=False,
        activation='elu',
        print_time = False,
        publish=None
        ):
    
    '''
    inputnames [
        'recHitFeatures', 'recHitFeatures_rowsplits', 
        't_idx', 't_idx_rowsplits', 't_energy', 't_energy_rowsplits', 't_pos',
        't_pos_rowsplits', 't_time', 't_time_rowsplits', 't_pid',
        't_pid_rowsplits', 't_spectator', 't_spectator_rowsplits',
        't_fully_contained', 't_fully_contained_rowsplits', 't_rec_energy',
        't_rec_energy_rowsplits', 't_is_unique', 't_is_unique_rowsplits'] [
        't_idx', 't_energy', 't_pos', 't_time', 't_pid', 't_spectator',
        't_fully_contained', 't_rec_energy', 't_is_unique',
        't_spectator_weight', 'coords', 'rechit_energy', 'features', 'is_track',
        'row_splits', 'scatterids', 'orig_row_splits']
    '''
    
    orig_inputs = condition_input(orig_inputs, no_scaling=pf_mode)
    
    if pass_through:
        orig_inputs['orig_row_splits'] = orig_inputs['row_splits'] 
        return orig_inputs
    
    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    coords = orig_inputs['coords']
    is_track = orig_inputs['is_track']
    t_spec_w = orig_inputs['t_spectator_weight']
    t_idx = orig_inputs['t_idx']
    x = orig_inputs['features']
    
    coords = ElementScaling(name=name+'es1', trainable=trainable)(coords)
    coords = LLClusterCoordinates(active = trainable,
        name = name+'_LLClusterCoordinates',record_metrics = record_metrics,
        scale=1.
        )([coords, t_idx, t_spec_w, energy, rs])

    nidx, dist = KNN(K=K, name=name+'_knn',
            record_metrics=record_metrics, 
            min_bins=20)([coords,rs])
    
    x = DistanceWeightedMessagePassing([32,32],
            name=name+'dmp1', 
            activation=activation,
            trainable=trainable)([x,nidx,dist])

    x = Dense(32, activation=activation,name=name+'dense1a',trainable=trainable)(x)
    x = Dense(32, activation=activation,name=name+'dense1b',trainable=trainable)(x)

    filtered_out = noise_filter_block(
            orig_inputs,
            x,
            name=name+'_noiseFilterBlock_',
            trainable=trainable,
            record_metrics=record_metrics,
            noise_threshold=noise_threshold,
            rs=rs)
    
    return filtered_out
    
    
def noise_filter_block(orig_inputs, x, name, trainable, 
                       record_metrics, noise_threshold, rs, 
                       print_time=False,
                       debugplots_after=-1,
                       debug_outdir=None,
                       publish = None):   
    
    isnotnoise = Dense(1, 
            activation='sigmoid',
            trainable=trainable,
            name=name+'_noisescore_d1')(x)
    
    #spectators are never noise here
    notnoisetruth = Where(outputval=1,condition='>0')(
            [orig_inputs['t_spectator_weight'], orig_inputs['t_idx']])
    notnoisetruth = Where(outputval=1,condition='>0')(
            [orig_inputs['is_track'], notnoisetruth])
    
    isnotnoise = LLNotNoiseClassifier(active=trainable,
            record_metrics=record_metrics,
            print_time=print_time,
            scale=1.)([isnotnoise, notnoisetruth])
        
    #tracks are never noise here**2
    isnotnoise = Where(outputval=1.,condition='>0')(
            [orig_inputs['is_track'], isnotnoise])
    
    if debugplots_after > 0:
        isnotnoise  = PlotNoiseDiscriminator(
                name= name+'_noise',
                plot_every=debugplots_after,
                outdir=debug_outdir,
                publish=publish)([isnotnoise, notnoisetruth])
    
    no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(
            threshold = noise_threshold,
            #print_reduction=True,
            record_metrics=record_metrics)([isnotnoise,rs])
    
    out = orig_inputs
    out['x'] = x
    #select not-noise
    for k in out.keys():
        out[k] = SelectFromIndices()([no_noise_sel, out[k]]) 
    
    out['row_splits'] = no_noise_rs
    scatterids = noise_backscatter
        
    if 'scatterids' in orig_inputs.keys():
        out['scatterids'] = orig_inputs['scatterids'] + scatterids
    else:
        out['scatterids'] = scatterids
        
    if not 'orig_row_splits' in orig_inputs.keys():
        out['orig_row_splits'] = orig_inputs['row_splits']
    else:
        out['orig_row_splits'] = orig_inputs['orig_row_splits']#pass through 
    out['no_noise_sel'] = no_noise_sel
    out['no_noise_rs'] = no_noise_rs
    out['noise_backscatter'] = noise_backscatter
        
    return out
