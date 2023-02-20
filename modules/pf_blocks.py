
from tensorflow.keras.layers import Concatenate, Add, Dense

from RaggedLayers import RaggedPFCMomentum, RaggedSelectFromIndices, RaggedPFCIsNoise
from RaggedLayers import RaggedCollapseHitInfo, RaggedCreateCondensatesIdxs, RaggedDense, RaggedMixHitAndCondInfo, FlatRSToRagged, RaggedToFlatRS

from GravNetLayersRagged import RaggedGravNet
from LossLayers import LLPFCondensates,LLLocalEnergyConservation 

from Layers import OnesLike, Where
from DeepJetCore.DJCLayers import StopGradient

def create_adj_pf_outputs(out, pfc_x, name = 'create_adj_pf_outputs'):
    
    out['pfc_mom_corr'] = Add()([out['pfc_mom_corr'], 
                                 RaggedDense(1, activation='tanh',
                                             name=name+'_mom_corr',
                                             kernel_initializer='zeros')(pfc_x)])
    #these corrections can be pretty heavy
    out['pfc_mom_corr1'] = Add()([out['pfc_mom_corr'], 
                                 RaggedDense(1, activation='elu',
                                             name=name+'_mom_corr1',
                                             kernel_initializer='zeros')(pfc_x)])
    
    out['pfc_mom_corr_up'] = Add()([out['pfc_mom_corr_up'],
                                    RaggedDense(1,  activation='tanh',
                                                name=name+'_mom_corr_up',
                                             kernel_initializer='zeros')(pfc_x)])
    
    out['pfc_mom_corr_down'] = Add()([out['pfc_mom_corr_down'],
                                    RaggedDense(1,  activation='tanh',
                                                name=name+'_mom_corr_down',
                                             kernel_initializer='zeros')(pfc_x)])
    
    pfc_pos_add = RaggedDense(out['pfc_pos'].shape[-1], 
                              name=name+'_pos', kernel_initializer = 'zeros')(pfc_x)
    pfc_time_add = RaggedDense(1, name=name+'_pfc_time', kernel_initializer = 'zeros')(pfc_x)
    pfc_time_unc_add = RaggedDense(1, name=name+'_pfc_time_unc', kernel_initializer = 'zeros')(pfc_x)
    
    pfc_id = Concatenate()([out['pfc_id'], pfc_x])
    pfc_id = RaggedDense(out['pfc_id'].shape[-1], activation='softmax', name=name+'_id')(pfc_id)
    
    out['pfc_pos'] = Add()([out['pfc_pos'], pfc_pos_add])
    out['pfc_time'] = Add()([out['pfc_time'], pfc_time_add])
    out['pfc_time_unc'] = Add()([out['pfc_time_unc'], pfc_time_unc_add])
    out['pfc_id'] = pfc_id
    
    return out

def minimum_afterburner(x, out, pfc_h_idx, pfc_idx, name='minimum_afterburner'):
    
    
    x_h_rag = RaggedSelectFromIndices()([x, pfc_h_idx])
    x_rag = RaggedSelectFromIndices()([x, pfc_idx])
    x_h_rag = RaggedDense(64, activation='elu', name=name+'_d_h_1')(x_h_rag)
    x_rag = RaggedDense(64, activation='elu', name=name+'_d_1')(x_rag)
    
    ## non explicitly ragged block for gravnet
    
    xf,rs = RaggedToFlatRS()(x_rag)
    xc, _ = RaggedToFlatRS()(out['pfc_ccoords'])
    
    xc = StopGradient()(xc)#don't spoil the clustering coords
    xf = Concatenate()([xc,xf])
    
    xf, *_ = RaggedGravNet(name = name+'_gn',
                  n_neighbours=32,
                 n_dimensions=xc.shape[-1],
                 n_filters=32,
                 n_propagate=32,
                 feature_activation='elu'
                  )([xf,rs])
    
    xf = Dense(64, activation='elu', name=name+'_gnd_1')(xf)
    xf = Dense(64, activation='elu', name=name+'_gnd_2')(xf)
    x_rag = FlatRSToRagged()([xf,rs])
    
    # explicitly ragged again
    
    x_h_rag = RaggedMixHitAndCondInfo('add')([x_h_rag, x_rag])
    x_h_rag = RaggedDense(64, activation='elu', name=name+'_d_h_2')(x_h_rag)
    #x_h_rag = RaggedDense(64, activation='elu', name=name+'_d_h_3')(x_h_rag)
    
    x_rag_me = RaggedCollapseHitInfo('mean')(x_h_rag)
    x_rag_mx = RaggedCollapseHitInfo('max')(x_h_rag)
    x_rag_s = RaggedCollapseHitInfo('sum')(x_h_rag)
    
    #this is relatively cheap
    x_rag = Concatenate()([x_rag_me, x_rag_mx, x_rag_s])
    #x_rag = RaggedDense(128, activation='elu', name=name+'_d_all_1')(x_rag)
    x_rag = RaggedDense(64, activation='elu', name=name+'_d_all_2')(x_rag)
    
    out = create_adj_pf_outputs(out, x_rag)

    out['pfc_mom'] = RaggedPFCMomentum()([
        out['pfc_mom_corr'], out['pfc_ensum'], out['pfc_track_mom'], out['pfc_istrack']
        ])
    
    out['pfc_mom1'] = RaggedPFCMomentum()([
        out['pfc_mom_corr1'], out['pfc_ensum'], out['pfc_track_mom'], out['pfc_istrack']
        ])
    
    return out


def create_pf_condensates(
        beta, d, ccoords, 
        pred_energy, pred_energy_low_quantile, pred_energy_high_quantile, 
        pred_pos, pred_time, pred_time_unc, pred_id,
        
        is_track,
        hit_energy,
        rs,
        
        t_idx, t_energy, t_pos, t_time,
        t_pid, t_fully_contained, t_rec_energy,
        
        start_t_d = 0.5,
        start_t_b = 0.1,
        
        add_loss = True,
        
        x = None, #features for possible corrections
        afterburner = minimum_afterburner,
        
        loss_args : dict ={} 
        ):
    out = {}
    
    pfc_h_idx, pfc_idx, reverse, revflat,\
    pred_sid, asso_idx, t_d, t_b = RaggedCreateCondensatesIdxs(t_d = start_t_d, t_b = start_t_b,
                                            collapse_noise=True,
                                            return_thresholds=True)([beta,ccoords,d,rs])
    
    out['hit_pfc_asso'] = pred_sid  #for matching                         
    out['pfc_hit_idx'] = pfc_h_idx #this is for hit matching mostly             
    out['pfc_idx'] = pfc_idx
    
    #for info
    out['pfc_ccoords'] = RaggedSelectFromIndices()([ccoords, pfc_idx])
    
    out['pfc_istrack'] = RaggedSelectFromIndices()([is_track, pfc_idx])
    pf_h_energy = RaggedSelectFromIndices()([hit_energy, pfc_h_idx])
    
    out['pfc_ensum'] = RaggedCollapseHitInfo('sum')(pf_h_energy)
    out['pfc_mom_corr'] = RaggedSelectFromIndices()([pred_energy, pfc_idx])
    out['pfc_mom_corr_up'] = RaggedSelectFromIndices()([pred_energy_high_quantile, pfc_idx])
    out['pfc_mom_corr_down'] = RaggedSelectFromIndices()([pred_energy_low_quantile, pfc_idx])
    out['pfc_track_mom'] =  RaggedSelectFromIndices()([hit_energy, pfc_idx])
    
    out['pfc_mom'] = RaggedPFCMomentum()([
        out['pfc_mom_corr'], out['pfc_ensum'], out['pfc_track_mom'], out['pfc_istrack']
        ])
    #this is local energy corrected, if afterburner is running this
    out['pfc_mom_corr1'] = out['pfc_mom_corr']
    out['pfc_mom1'] = out['pfc_mom']
    
    out['pfc_pos'] =  RaggedSelectFromIndices()([pred_pos, pfc_idx])
    out['pfc_time'] =  RaggedSelectFromIndices()([pred_time, pfc_idx])
    out['pfc_time_unc'] =  RaggedSelectFromIndices()([pred_time_unc, pfc_idx])
    out['pfc_id'] =  RaggedSelectFromIndices()([pred_id, pfc_idx])
    
    pfc_asso = RaggedSelectFromIndices()([asso_idx, pfc_idx])
    out['pfc_isnoise'] = RaggedPFCIsNoise()(pfc_asso)
    
    
    #add corrections here
    if x is not None:
        assert afterburner is not None
        out = afterburner(x, out, pfc_h_idx, pfc_idx)
    
    if add_loss:
        
        out['pfc_mom_corr'] = LLPFCondensates(
            **loss_args
            )([
        out['pfc_mom_corr'],
        out['pfc_ensum'],
        out['pfc_mom_corr_down'],
        out['pfc_mom_corr_up'],
        out['pfc_pos'],
        out['pfc_time'],
        out['pfc_time_unc'],
        out['pfc_id'],
        out['pfc_ccoords'],
        out['pfc_istrack'],
        out['pfc_isnoise'],
        
        hit_energy,
        
        pfc_h_idx, pfc_idx,
        
        is_track,
        
        t_idx,
        t_energy,
        t_pos,
        t_time,
        t_pid,
        t_fully_contained,
        t_rec_energy,
        
        ])
            
        out['dummy_beta'] = LLLocalEnergyConservation(
            print_loss=True,
            record_metrics=True,
            print_batch_time=True,
            scale=.5
            )([ beta, ccoords, pred_sid, out['pfc_mom1'], 
                pfc_idx, t_idx, t_energy,  rs ])
            
            
    return out
    
    
