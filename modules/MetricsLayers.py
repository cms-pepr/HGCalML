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
        super(MLReductionMetrics, self).__init__(**kwargs)
    
    
    def metrics_call(self, inputs):
        assert len(inputs)==5
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
        
        ue,_ = tf.unique(ten[:,0])
        uesel,_ = tf.unique(sten[:,0])
        
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
        
        reduced_to_fraction = tf.cast(srs[-1],dtype='float32')/tf.cast(rs[-1],dtype='float32')
        self.add_prompt_metric(reduced_to_fraction,self.name+'_reduction')
        
        
        
        
    