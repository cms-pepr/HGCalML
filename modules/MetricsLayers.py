from baseModules import LayerWithMetrics
from GravNetLayersRagged import SelectFromIndices as SelIdx
import tensorflow as tf

class MLBase(LayerWithMetrics):
    def __init__(self, 
                 active=True,
                 **kwargs):
        '''
        Output is always inputs[0]
        Use active=False to mask truth inputs in the inheriting class.
        No need to activate or deactivate metrics recording (done in LayerWithMetrics)
        
        Inherit and implement metrics_call method
        
        '''
        self.active=active
        super(MLBase, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'active': self.active}
        base_config = super(MLBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return  input_shapes[0]
    
    def build(self, input_shapes):
        super(MLBase, self).build(input_shapes)
    
    def metrics_call(self, inputs):
        pass
    
    def call(self, inputs):
        self.metrics_call(inputs)
        return inputs[0]

class SimpleReductionMetrics(MLBase):
    
    def __init__(self, **kwargs):
        '''
        Inputs:
        - after reduction row splits
        - previous row splits
        
        
        Output:
        - previous row splits (unchanged)
        
        '''
        super(SimpleReductionMetrics, self).__init__(**kwargs)
        
    def metrics_call(self, inputs):
        if len(inputs)==2:
            after,bef = inputs
            self.add_prompt_metric(tf.reduce_mean(after[-1]/bef[-1]),self.name+'_rel_reduction')
        return 0.
        
  


from oc_helper_ops import per_rs_segids_to_unique
class OCReductionMetrics(MLBase):
    
    def __init__(self, return_threshold_updates=False, **kwargs):
        '''
        Inputs:
        - asso idx (V) 
        - pred_sid (V)
        - truth indices (V x 1)
        - row splits (for V)
        
        Calculates the condensation purity (fraction of clusters with identical truth index)
        
        '''
        super(OCReductionMetrics, self).__init__(**kwargs)
        self.return_threshold_updates = return_threshold_updates

    def get_config(self):
        config = {'return_threshold_updates': self.return_threshold_updates}
        base_config = super(OCReductionMetrics, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def metrics_call(self, inputs):
        
        assert len(inputs)==2 or len(inputs)==3
        if len(inputs)==2:
            asso_idx, tidx = inputs
            energy = tf.ones_like(tidx, dtype='float32')
        else:
            asso_idx, tidx, energy = inputs
            
        energy = energy[:,0]
        
        asso_idx = tf.where(asso_idx<0, tf.range(tf.shape(asso_idx)[0]), asso_idx)
        
        nafter = tf.cast(tf.shape(tf.unique(asso_idx)[0])[0], dtype='float32')
        nbefore = tf.cast(tf.shape(asso_idx)[0], dtype='float32')
        
        reduction = nafter/nbefore
        
        self.add_prompt_metric(reduction, self.name + '_reduction')
        
        #use asso index
        asso_tidx = tf.gather(tidx[:,0], asso_idx)
        sames = tf.cast(asso_tidx == tidx[:,0], 'float32')
        purity = tf.reduce_sum( energy * sames ) / tf.reduce_sum(energy)
        
        self.add_prompt_metric(purity, self.name+'_purity')
        
        #print(self.name, 'purity',purity,'reduction',reduction)
    
        return purity*0., purity*0. #TBI
    
        #rpsid = tf.RaggedTensor.from_row_splits(pred_sid[:,0],rs)
        #
        #rtidx = tf.RaggedTensor.from_row_splits(tidx[:,0],rs)
        ## remove noise
        #rtidx = tf.ragged.boolean_mask(rtidx, rpsid>=0)
        #rpsid = tf.ragged.boolean_mask(rpsid, rpsid>=0)
        #
        ##check if they are consecutive
        #
        #pred_sid, tidx, rs = rpsid.values, rtidx.values, rpsid.row_splits
        
        #can contain empty segments here
        orig_pred_sid = pred_sid
        pred_sid, nseg = per_rs_segids_to_unique(pred_sid+1, rs, 
                                                 return_nseg=True, 
                                                 strict_check=False)#pred_sid should be regular
        
        mintidx = tf.math.unsorted_segment_min(tidx, pred_sid, 
                                               num_segments=nseg)
        maxtidx = tf.math.unsorted_segment_max(tidx, pred_sid, 
                                               num_segments=nseg)
        
        pidx = tf.math.unsorted_segment_min(orig_pred_sid, pred_sid, 
                                               num_segments=nseg)
        
        ispure = tf.cast(mintidx == maxtidx, dtype='float32')
        
        ispure = tf.boolean_mask(ispure, pidx>=0)

        purity = tf.reduce_mean(ispure)
        
        

        

class MLReductionMetrics(MLBase):
    
    def __init__(self, **kwargs):
        '''
        Inputs:
        - neighbour group selection indices (V_red x 1)
        - truth indices (V x 1)
        - truth energies (V x 1)
        - row splits (for V)
        - sel row splits (for V_red)
        
        '''
        self.extended = False
        super(MLReductionMetrics, self).__init__(**kwargs)
        
        
    
    
    def metrics_call(self, inputs):
        #tren = None
        if len(inputs)==5:
            gsel,tidx,ten,rs,srs = inputs
        #tf.assert_equal(tidx.shape,ten.shape)#safety
        
        alltruthcount = None
        seltruthcount = None
        nonoisecounts_bef = []
        nonoisecounts_after = []
        
        if rs.shape[0] is None:
            return
        
        stidx, sten = tf.constant([[0]],dtype='int32'), tf.constant([[0.]],dtype='float32')
        
        if self.active:
            stidx, sten = SelIdx.raw_call(gsel,[tidx,ten])
            for i in tf.range(rs.shape[0]-1):
                u,_,c = tf.unique_with_counts(tidx[rs[i]:rs[i+1],0])
                nonoisecounts_bef.append(c[u>=0])
                if alltruthcount is None:
                    alltruthcount = u.shape[0]
                else:
                    alltruthcount += u.shape[0]
            
                u,_,c = tf.unique_with_counts(stidx[srs[i]:srs[i+1],0])
                nonoisecounts_after.append(c[u>=0])
                if seltruthcount is None:
                    seltruthcount = u.shape[0]
                else:
                    seltruthcount += u.shape[0]
        
        nonoisecounts_bef = tf.concat(nonoisecounts_bef,axis=0)
        nonoisecounts_after = tf.concat(nonoisecounts_after,axis=0)
        
        lostfraction = 1. - tf.cast(seltruthcount,dtype='float32')/(tf.cast(alltruthcount,dtype='float32'))
        self.add_prompt_metric(lostfraction,self.name+'_lost_objects')
        #done with fractions
        
        #for simplicity assume that no energy is an exact duplicate (definitely good enough here)
        no_noise_sel = tidx[:,0]>=0
        ue,_,c = tf.unique_with_counts(tf.boolean_mask(ten, no_noise_sel)[:,0])
        #ue = ue[c>3] #don't count <4 hit showers
        no_noise_sel = stidx[:,0]>=0
        uesel,_,c = tf.unique_with_counts(tf.boolean_mask(sten, no_noise_sel)[:,0])#only non-noise
        
        tot_lost_en_sum = tf.reduce_sum(ue) - tf.reduce_sum(uesel)
        
        allen = tf.concat([ue,uesel],axis=0)
        ue,_,c = tf.unique_with_counts(allen)
        
        
        lostenergies = ue[c<2]
        #print(lostenergies)
        
        self.add_prompt_metric(tf.reduce_mean(nonoisecounts_bef),self.name+'_hits_pobj_bef_mean')
        self.add_prompt_metric(tf.reduce_max(nonoisecounts_bef),self.name+'_hits_pobj_bef_max')
        
        self.add_prompt_metric(tf.reduce_mean(nonoisecounts_after),self.name+'_hits_pobj_after_mean')
        self.add_prompt_metric(tf.reduce_max(nonoisecounts_after),self.name+'_hits_pobj_after_max')

        self.add_prompt_metric(tf.reduce_mean(lostenergies),self.name+'_lost_energy_mean')
        self.add_prompt_metric(tf.reduce_max(lostenergies),self.name+'_lost_energy_max')
        self.add_prompt_metric(tot_lost_en_sum,self.name+'_lost_energy_sum')
        
        reduced_to_fraction = tf.cast(srs[-1],dtype='float32')/tf.cast(rs[-1],dtype='float32')
        self.add_prompt_metric(reduced_to_fraction,self.name+'_reduction')
        
        no_noise_hits_bef = tf.cast(tf.math.count_nonzero(tidx+1)  ,dtype='float32')
        no_noise_hits_aft = tf.cast(tf.math.count_nonzero(stidx+1) ,dtype='float32')
        self.add_prompt_metric(no_noise_hits_aft/no_noise_hits_bef,self.name+'_no_noise_reduction')
        
        
        
        
    
