import tensorflow as tf
import pdb
import yaml
import os
from select_knn_op import SelectKnn
from slicing_knn_op import SlicingKnn
from binned_select_knn_op import BinnedSelectKnn
from select_mod_knn_op import SelectModKnn
from accknn_op import AccumulateKnn, AccumulateLinKnn
from local_cluster_op import LocalCluster
from local_group_op import LocalGroup
from local_distance_op import LocalDistance
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
import numpy as np
#just for the moment
#### helper###
from Initializers import EyeInitializer

from oc_helper_ops import SelectWithDefault

from baseModules import LayerWithMetrics

#helper
#def AccumulateKnnRS(distances,  features, indices, 
#                  mean_and_max=True,):
#    
#    out,midx=AccumulateKnn(distances,  features, indices, 
#                  mean_and_max=mean_and_max)
#    
#    outshape = tf.shape(features)[1]
#    if mean_and_max:
#        outshape *= 2
#    return tf.reshape(out, [-1, outshape]),midx
 
def AccumulateKnnSumw(distances,  features, indices, mean_and_max=False):
    
    origshape = features.shape[1]
    features = tf.concat([features, tf.ones_like(features[:,0:1])],axis=1)
    f,midx = AccumulateKnn(distances,  features, indices,mean_and_max=mean_and_max)
    
    fmean = f[:,:origshape]
    fnorm = f[:,origshape:origshape+1]
    fmean = tf.math.divide_no_nan(fmean,fnorm)
    fmean = tf.reshape(fmean, [-1,origshape])
    if mean_and_max:
        fmean = tf.concat([fmean, f[:,origshape+1:-1]],axis=1)
    return fmean,midx
  

def AccumulateLinKnnSumw(weights,  features, indices, mean_and_max=False):
    
    origshape = features.shape[1]
    features = tf.concat([features, tf.ones_like(features[:,0:1])],axis=1)
    f,midx = AccumulateLinKnn(weights,  features, indices,mean_and_max=mean_and_max)
    
    fmean = f[:,:origshape]
    fnorm = f[:,origshape:origshape+1]
    fmean = tf.math.divide_no_nan(fmean,fnorm)
    fmean = tf.reshape(fmean, [-1,origshape])
    if mean_and_max:
        fmean = tf.concat([fmean, f[:,origshape+1:-1]],axis=1)
    return fmean,midx
  

def check_type_return_shape(s):
    if not isinstance(s, tf.TensorSpec):
        raise TypeError('Only TensorSpec signature types are supported, '
                      'but saw signature entry: {}.'.format(s))
    return s.shape



def select_threshold_with_backgather(score, threshold, row_splits):
    '''
    Selects all above threshold plus the lowest score vertex as representation of all below threshold.
    The backgather indices are constructed such that all below threshold will be assigned to the
    representative point below threshold.
    
    returns:
    - selection indices
    - backgather indices
    - new row splits
    '''
    glidxs = tf.range(tf.shape(score)[0]) #for backgather
    allbackgather = []
    allkeepidxs = []
    newrs = [0]
    row_splits = tf.squeeze(row_splits) #can never have dim 0
    score= tf.squeeze(score, axis=1)#remove additional 1 dim
    
    #debug_indices = glidxs
    
    for i in tf.range(tf.shape(row_splits)[0]-1):
        rs_glidxs = glidxs[row_splits[i]:row_splits[i+1]]
        rs_score = score[row_splits[i]:row_splits[i+1]]
        
        above_threshold = rs_score > threshold
        argminscore = tf.argmin(rs_score)
        isminscore = rs_score == rs_score[argminscore]
        
        keep = tf.logical_or(above_threshold, isminscore)
        

        keepindxs = rs_glidxs[keep]
        rs_keepindxs = tf.range(tf.shape(rs_glidxs)[0])[keep]
        
        keeprange = tf.range(keepindxs.shape[0])
        
        minidxlocal_rs = rs_score[argminscore] == rs_score[keep]
        minidx = keeprange[minidxlocal_rs][0]#just in the very very inlikley case the above gives>1 results

        scatidxs = tf.expand_dims(rs_keepindxs, axis=-1)
        
        n_previous_rs = newrs[-1]
        scatupds = keeprange + 1 + n_previous_rs
        
        scatshape = tf.shape(rs_glidxs)
        
        back = tf.scatter_nd(scatidxs, scatupds, scatshape)
        back = tf.where(back == 0, minidx + n_previous_rs, back -1)
        
        allbackgather.append(back)
        allkeepidxs.append(keepindxs)
        
        newrs.append(keepindxs.shape[0]+n_previous_rs)
        
    allbackgather = tf.expand_dims(tf.concat(allbackgather,axis=0),axis=1)
    sel = tf.expand_dims(tf.concat(allkeepidxs,axis=0),axis=1)#standard tf index format
    newrs = tf.concat(newrs,axis=0)
     
    #sanity check
    selglidxs = SelectWithDefault(sel, tf.expand_dims(glidxs,axis=1), 0)[:,:,0]
    mess = "select_threshold_with_backgather: assert indices "+str(selglidxs.shape)+" vs "+str(sel.shape)
    with tf.control_dependencies([tf.assert_equal(selglidxs, sel, message=mess)]):
    #end sanity check 
        return sel, allbackgather, newrs



def countsame(idxs):
    '''
    idxs are supposed to have form V x K
    returns number of sames (repeats) and ntotal
    '''
    ntotal = idxs.shape[1]
    sqdiff = tf.expand_dims(idxs,axis=1)-tf.expand_dims(idxs,axis=2)
    nsames = idxs.shape[1] - tf.math.count_nonzero(sqdiff,axis=2) 
    nsames = tf.cast(nsames,dtype='int32')
    ntotal = tf.cast(ntotal,dtype='int32')
    return nsames,ntotal
    
    
    
    
######################### actual layers: simple helpers first

class AssertEqual(tf.keras.layers.Layer): 
    def call(self,inputs):
        assert len(inputs) == 2
        tf.assert_equal(tf.shape(inputs[0]),tf.shape(inputs[1]), "shape not equal")
        tf.assert_equal(inputs[0],inputs[1], "elements not equal")
        return inputs[0]


class Abs(tf.keras.layers.Layer): 
    def call(self,inputs):
        return tf.abs(inputs)

class CastRowSplits(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        This layer casts the row splits as they come from the data (int64, N_RS x 1) 
        to (int32, N_RS), which is needed for subsequent processing. That's all
        '''
        super(CastRowSplits, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(CastRowSplits, self).build(input_shape)
    
    def compute_output_shape(self, input_shapes):
        return (input_shapes[0],)
            
    def call(self,inputs):
        assert inputs.dtype=='int64' or inputs.dtype=='int32'
        if len(inputs.shape)==2:
            return tf.cast(inputs[:,0],dtype='int32')
        elif inputs.dtype=='int64':
            return tf.cast(inputs,dtype='int32')
        else:
            return inputs

class CreateMask(tf.keras.layers.Layer):
    def __init__(self, threshold, invert=False, **kwargs):
        '''
        Creates a mask.
        For values below threshold: 0, above: 1
        If invert=True, this is inverted
        '''
        super(CreateMask, self).__init__(**kwargs)
        self.threshold = threshold
        self.invert = invert
        
    def get_config(self):
        base_config = super(CreateMask, self).get_config()
        return dict(list(base_config.items()) + list({'threshold': self.threshold ,
                                                      'invert': self.invert
                                                      }.items()))    
    def call(self, inputs ):
        zeros = tf.zeros_like(inputs)
        ones = tf.ones_like(inputs)
        if self.invert:
            return tf.where(inputs >= self.threshold, zeros, ones)
        else:
            return tf.where(inputs < self.threshold, zeros, ones)

        

class Where(tf.keras.layers.Layer):
    def __init__(self, outputval, condition = '>0', **kwargs):
        conditions = ['>0','>=0','<0','<=0','==0', '!=0']
        assert condition in conditions
        self.condition = condition
        self.outputval = outputval
        
        super(Where, self).__init__(**kwargs)
        
    def get_config(self):
        base_config = super(Where, self).get_config()
        return dict(list(base_config.items()) + list({'outputval': self.outputval ,
                                                      'condition': self.condition
                                                      }.items()))
        
    def build(self, input_shape):
        super(Where, self).build(input_shape)
        
    def compute_output_shape(self, input_shapes):
        return (input_shapes[1],)
    
    def call(self,inputs):
        assert len(inputs)==2
        izero = tf.constant(0,dtype=inputs[0].dtype)
        if self.condition == '>0':
            return tf.where(inputs[0]> izero,  self.outputval, inputs[1])
        elif self.condition == '>=0':
            return tf.where(inputs[0]>=izero, self.outputval, inputs[1])
        elif self.condition == '<0':
            return tf.where(inputs[0]< izero,  self.outputval, inputs[1])
        elif self.condition == '<=0':
            return tf.where(inputs[0]<=izero, self.outputval, inputs[1])
        elif self.condition == '!=0':
            return tf.where(inputs[0]!=izero, self.outputval, inputs[1])
        else:
            return tf.where(inputs[0]==izero, self.outputval, inputs[1])


class MixWhere(tf.keras.layers.Layer):
    def call(self,inputs):
        assert len(inputs)==3
        out = tf.where(inputs[0]>0, inputs[1], inputs[2])
        return out


class ValAndSign(tf.keras.layers.Layer):
    '''
    Returns the absolute value and sign independently
    '''
    def call(self,inputs):
        s = tf.sign(inputs)
        v = tf.abs(inputs)
        return tf.concat([s,v],axis=-1)


class SplitOffTracks(tf.keras.layers.Layer):
    
    def __init__(self,**kwargs):
        '''
        This layer does not assume that tracks are at the end of the
        input array.
        
        inputs:
        - A: track identifier (non-zero for tracks)
        - B: a list of other inputs that the detach should act on
        - C: row splits
        
        Outputs:
        - 1) list of all inputs (B) that are not tracks
        - 2) list of all inputs (B) that are tracks
        - row splits for outputs 1
        - row splits for outputs 2
        '''
        super(SplitOffTracks, self).__init__(**kwargs)
    
    def call(self, inputs):
        assert len(inputs) > 2
        istrack, other, rs = inputs

        ristrack = tf.RaggedTensor.from_row_splits(istrack,rs)
        ristrack = tf.squeeze(ristrack,axis=-1)
        rnottrack = ristrack == 0.
        ristrack = ristrack != 0.
        
        outnotrack = []
        outistrack = []
        trs, ntrs = None, None
        for a in other:
            ra = tf.RaggedTensor.from_row_splits(a,rs)
            nta = tf.ragged.boolean_mask(ra, rnottrack)
            ta  = tf.ragged.boolean_mask(ra, ristrack)
            
            outnotrack.append(nta.values)
            outistrack.append(ta.values)
            
            trs, ntrs = ta.row_splits, nta.row_splits
            
        return outnotrack, outistrack, ntrs, trs
         
###
class ConcatRaggedTensors(tf.keras.layers.Layer):
    
    def __init__(self,**kwargs):
        '''
        This layer concatenates two lists of ragged tensors in values, row_split format.
        It is assumed that the same row splits apply to all tensors in an individual list.
        When used to merge-back tracks, make sure that the tracks are trailing to
        keep same format as the input to the model.
        
        inputs:
        - A: A list of value tensors to be concatenated at leading position
        - B: A list of value tensors to be concatenated at trailing position
        - C: common row splits for all tensors in A
        - D: common row splits for all tensors in B
        
        Outputs:
        - 1) a list of concatenated value tensors
        - 2) row splits common to all tensors in output 1
        '''
        super(ConcatRaggedTensors, self).__init__(**kwargs)
        
    def call(self, inputs):
        assert len(inputs) > 3
        a, b, rsa, rsb = inputs
        assert len(a) == len(b)
        
        merged = []
        mrs = None
        for ai,bi in zip(a, b):
    
            ra = tf.RaggedTensor.from_row_splits(ai,rsa)
            rb = tf.RaggedTensor.from_row_splits(bi,rsb)
            m = tf.concat([ra,rb],axis=1)#here defined as axis 1
            
            merged.append(m.values)
            mrs = m.row_splits
    
        return merged, mrs


class MaskTracksAsNoise(tf.keras.layers.Layer):
    def __init__(self, 
                 active=True,
                 maskidx=-1,
                 **kwargs):
        '''
        Inputs:
        - truth index
        - track_charge
        '''
        super(MaskTracksAsNoise, self).__init__(**kwargs)
        self.active = active
        self.maskidx = maskidx
    
    def get_config(self):
        base_config = super(MaskTracksAsNoise, self).get_config()
        return dict(list(base_config.items()) + list({'active': self.active ,
                                                      'maskidx': self.maskidx
                                                      }.items()))

    def build(self, input_shape):
        super(MaskTracksAsNoise, self).build(input_shape)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
            
    def call(self,inputs):
        assert len(inputs)==2
        assert inputs[0].dtype=='int32'
        tidx, trackcharge = inputs
        if self.active:
            tidx = tf.where(tf.abs(trackcharge)>1e-3, self.maskidx, tidx)
        return tidx

from assign_condensate_op import BuildAndAssignCondensatesBinned as ba_cond
class CondensateToIdxs(LayerWithMetrics):
    
    def __init__(self, t_d, t_b,
                 active=True,
                 keepnoise = False,
                 return_thresholds = False,
                 **kwargs):
        '''
        This layer builds condensates.
        
        Inputs:
        
        - beta
        - cluster space coords
        - individual distance scaler
        - no condensation mask (1: do not condensate point) (optional)
        - row splits
        
        Outputs:
        - asso_idx (index of condensation point a point is associated to)
        - predicted associated ID (0,1,2,3,... resets at each row split)
        
        Options:
        - t_d: distance threshold for inference clustering
        - t_b: beta threshold for inference clustering
        - active: can switch off condensation, just for faster pre-training
                  in this case (off) it each input point will be its own condensation point
        - keepnoise: noise will not be assigned a negative association index but a 'self' index
        - return_thresholds: returns t_b and t_d in addition to the outputs
                             and initialises them dynamically, so they have to be trained
        
        
        '''
        assert 0. <= t_b <= 1.
        self.t_d=t_d
        self.t_b=t_b
        self.active = active
        self.keepnoise = keepnoise
        self.return_thresholds = return_thresholds
        self.return_full_assignment = False
        super(CondensateToIdxs, self).__init__(**kwargs)
        
    def get_config(self):
            #when saving always assume active!
            config = {'t_d': self.t_d,
                      't_b': self.t_b,
                      'keepnoise': self.keepnoise,
                      'return_thresholds': self.return_thresholds}
            base_config = super(CondensateToIdxs, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    def build(self,input_shape):
        
        def tb_init(shape, dtype=None):
            return -tf.math.log( 1 / tf.constant(self.t_b)[...,tf.newaxis] - 1)
        def td_init(shape, dtype=None):
            return tf.constant(self.t_d)[...,tf.newaxis]
        
        if self.return_thresholds:
            self.dyn_t_d = self.add_weight(name = 'dyn_td', shape=(1,),
                                       initializer = td_init, 
                                        constraint = 'non_neg',
                                        trainable = self.trainable and self.return_thresholds) 
            
            self._dyn_t_b = self.add_weight(name = 'dyn_tb', shape=(1,),
                                        initializer = tb_init, 
                                        #constraint = tf.keras.constraints.MinMaxNorm(1e-4, 1.),
                                        trainable = self.trainable and self.return_thresholds) 
        
        else:
            self.dyn_t_d = td_init(None)
            self._dyn_t_b = tb_init(None)
        
        super(CondensateToIdxs, self).build(input_shape)
    
    @property
    def dyn_t_b(self):
        return tf.nn.sigmoid(self._dyn_t_b)
    
    def call(self, inputs):
        assert len(inputs) == 4 or len(inputs) == 5
        if len(inputs) == 4:
            beta, ccoords, d, rs = inputs
            nocondmask = None
        else:
            beta, ccoords, d, nocondmask, rs = inputs
        
        ridxs = tf.range(tf.shape(beta)[0])
        pred_sid = ridxs
        
        if rs.shape[0] == None:
            pred_sid = ridxs[...,tf.newaxis]
            asso = ridxs
            alpha ,ncond = ridxs, ridxs
            
            if self.return_thresholds:
                if self.return_full_assignment:
                    return pred_sid, asso, alpha, ncond,\
                         self.dyn_t_d,self.dyn_t_b
                return pred_sid, asso, alpha, self.dyn_t_d,self.dyn_t_b
            if self.return_full_assignment:
                return pred_sid, asso, alpha, ncond
            return pred_sid, asso, alpha
        
        asso_idx = ridxs
        if self.active:
            t_d = tf.stop_gradient(self.dyn_t_d[0])
            d = d * t_d
            t_b = tf.abs(self.dyn_t_b).numpy()[0] #abs is safety guard as the constraint does not seem to be inf
            
            pred_sid, asso_idx, alpha, _ ,ncond = ba_cond(ccoords, beta, row_splits=rs, 
                                beta_threshold=t_b ,
                                 dist=d, 
                                 no_condensation_mask=nocondmask,
                                 assign_by_max_beta=False)

            # doesn't work (anymore)
            # if self.keepnoise:
            #     asso_idx = tf.where(asso_idx<1, ridxs, asso_idx)
                
            # check pred_sid
            if False:
                for i in tf.range(rs.shape[0]-1):
                    with tf.control_dependencies([pred_sid]):
                        check = pred_sid[rs[i]:rs[i+1]]+1
                        nseg_a = tf.reduce_sum(tf.cast(tf.unique(tf.squeeze(check))[0],dtype='int32') * 0 +  1)
                        nseg_b = tf.reduce_max(check)+1
                        try:
                            tf.assert_equal(nseg_a, nseg_b)
                        except Exception as e:
                            print('error', check, t_b, self.dyn_t_d)
                            import pickle
                            with open('pred_sid_test','wb') as f:
                                pickle.dump({'ccoords':ccoords.numpy(),
                                 'beta':beta.numpy(),
                                 'row_splits':rs.numpy(),
                                 'd':d.numpy(),
                                 'pred_sid':pred_sid.numpy(),
                                 't_b':t_b,
                                 't_d': self.dyn_t_d.numpy()
                                 },f)
                            raise e
                    
        
        else: #output dummy 1:1 values
            pred_sid = tf.ragged.row_splits_to_segment_ids(rs, out_type=tf.int32)
            pred_sid = tf.gather(rs, pred_sid)
            pred_sid = (ridxs - pred_sid)[:, tf.newaxis]
           
        if self.return_thresholds:
            self.add_prompt_metric(self.dyn_t_d, self.name+'_dyn_t_d')
            self.add_prompt_metric(self.dyn_t_b, self.name+'_dyn_t_b')
            if self.return_full_assignment:
                return pred_sid, asso_idx, alpha ,ncond,\
                    self.dyn_t_d, \
                    self.dyn_t_b
            else:
                return pred_sid, asso_idx, alpha, \
                    self.dyn_t_d, \
                    self.dyn_t_b
        
        if self.return_full_assignment:
            return pred_sid, asso_idx, alpha ,ncond
        return pred_sid, asso_idx, alpha

from pseudo_rs_op import create_prs_indices, revert_prs
class CondensatesToPseudoRS(tf.keras.layers.Layer):            
        def __init__(self, **kwargs):
            '''
            creates pseudo row splits, one for each condensate (and one for noise)
            
            Input:
            - association index
            - list of data
            
            Output:
            - indices to revert operation (to be used with ReversePseudoRS layer)
            - pseudo row splits
            - sorted data
            
            '''
            super(CondensatesToPseudoRS, self).__init__(**kwargs)
        def call(self, inputs):
            assert len(inputs)>1
            asso_idx, *other = inputs
            
            toidxs, prs = create_prs_indices(asso_idx)
            out = []
            for a in other:
                out.append(tf.gather_nd(a, toidxs))
            
            return [toidxs, prs] + out

            

class ReversePseudoRS(tf.keras.layers.Layer):            
        def __init__(self, **kwargs):
            '''
            Input:
            - reverting index
            - list of data
            
            Output:
            - reverted data (list)
            
            '''
            super(ReversePseudoRS, self).__init__(**kwargs)
        def call(self, inputs):
            assert len(inputs)>1
            backidxs, *other = inputs
            out = []
            for a in other:
                out.append(revert_prs(a, backidxs))
                
            return out
        

class CleanCondensations(LayerWithMetrics):
    def __init__(self, threshold, **kwargs):
        '''
        Resets the association index for individual points back to itself
        and the pred_sid to -1 if a point is below threshold
        
        That means pred_sid is not regular anymore after this!!!
        
        inputs:
        - asso idx
        - pred_sid
        - score
        
        output:
        - asso idx 
        - pred_sid
        '''
        super(CleanCondensations, self).__init__(**kwargs)
        self.threshold = threshold
        
    def get_config(self):
        base_config = super(CleanCondensations, self).get_config()
        return dict(list(base_config.items()) + list({'threshold': self.threshold }.items()))

    def call(self, inputs):
        assert len(inputs) == 3
        asso, pred_sid, score = inputs
        
        orig_pred_sid_cond = tf.cast(pred_sid >= 0, 'float32')
        
        pred_sid = tf.where(score[:,0] < self.threshold, -1, pred_sid[:,0])[:,tf.newaxis]
        asso = tf.where(score[:,0] < self.threshold, tf.range(tf.shape(asso)[0]), asso)
        #raise ValueError("lala")
        
        rem = tf.cast(score < self.threshold, 'float32')
        sel = tf.reduce_sum(rem*orig_pred_sid_cond)#actually cleaned
        allv = tf.reduce_sum(orig_pred_sid_cond)
        self.add_prompt_metric(sel/allv, self.name+'_cleaned_fraction')
        
        return asso, pred_sid
            
class ScaleBackpropGradient(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        '''
        Scales the gradient in back propagation. 
        Useful for strong reduction models where low-statistics parts 
        should get lower effective learning rates
        
        Please notice that this is applied backwards (as it affects back propagation)
        '''
        super(ScaleBackpropGradient, self).__init__(**kwargs)
        self.scale = scale
        
    def get_config(self):
        base_config = super(ScaleBackpropGradient, self).get_config()
        return dict(list(base_config.items()) + list({'scale': self.scale }.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, inputs):
        @tf.custom_gradient
        def scale_grad(x):
            def grad(dy):
                return self.scale * dy
            return tf.identity(x), grad
        
        islist = isinstance(inputs, list)
        
        if not islist:
            inputs = [inputs]
            
        out = []
        for i in inputs:
            out.append(scale_grad(i))
            
        if islist:  
            return out
        return out[0]    
    
class RemoveSelfRef(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RemoveSelfRef, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs[:,1:]
    
        
class CreateIndexFromMajority(tf.keras.layers.Layer):
    
    def __init__(self, min_threshold=0.6, ignore: int=-10, assign_else: int=-1, active=True, **kwargs):
        '''
        Takes as input (truth) indices and selects one from majority if majority is over threshold
        If majority is not over threshold it assigns <assign_else>
        It will ignore the index specified as <ignore> 
        
        If not active it will just return <assign_else> (no truth mode)
        
        Input:
        - indices V x K
        
        Output:
        - index V x 1
        '''
        super(CreateIndexFromMajority, self).__init__(**kwargs)
        assert min_threshold>=0
        self.min_threshold = min_threshold
        self.ignore = ignore
        self.active = active 
        self.assign_else = assign_else
        
        raise ValueError("TBI. There seems to be an issue somewhere, needs to be investigated before using this layer")
    
    def get_config(self):
        config = {'min_threshold': self.min_threshold,
                  'ignore': self.ignore,
                  'active': self.active,
                  'assign_else': self.assign_else,
                  }
        base_config = super(CreateIndexFromMajority, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @staticmethod
    def raw_call(inputs, min_threshold, ignore, assign_else, row_splits):
        
        # this needs to be done per row split???!!!!
        #0/0
        
        idxs = inputs
        nsames,ntot = countsame(idxs)
        #mask ignore
        nsames = tf.where(idxs == ignore, 0, nsames)
        #get max
        nmaxarg = tf.expand_dims(tf.argmax(nsames,axis=1),axis=1)
        maxcount = tf.gather_nd(nsames,nmaxarg,batch_dims=1)
        maxidx = tf.gather_nd(idxs,nmaxarg,batch_dims=1)
        fracmax = tf.cast(maxcount,dtype='float32')/tf.cast(ntot,dtype='float32')
        
        ### for info purposes

        ###
        
        out = tf.where(fracmax>=min_threshold, maxidx, assign_else)
        out = tf.expand_dims(out,axis=1)#make it V x 1 again
        return out
    
    def call(self, inputs):
        if not self.active:
            return tf.zeros_like(inputs) + self.assign_else
        
        return CreateIndexFromMajority.raw_call(inputs, self.min_threshold, self.ignore, self.assign_else)
        
    
class DownSample(tf.keras.layers.Layer):
    
    def __init__(self, sample_to: int=1000, **kwargs):
        '''
        Takes as input (truth) indices and selects one from majority if majority is over threshold
        If majority is not over threshold it assigns <assign_else>
        It will ignore the index specified as <ignore> 
        
        If not active it will just return <assign_else> (no truth mode)
        
        Input:
        - indices V x K
        
        Output:
        - index V x 1
        '''
        super(DownSample, self).__init__(**kwargs)
        assert sample_to>=0
        self.sample_to = sample_to
    
    def get_config(self):
        config = {'sample_to': self.sample_to,
                  }
        base_config = super(DownSample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        if input_shapes[-1] > self.sample_to:
            return input_shapes[:-1]+[self.sample_to]
        else:
            return input_shapes
    
    def call(self, inputs):
        if inputs.shape[-1]<=self.sample_to:
            return inputs
        
        idxs = tf.range(tf.shape(inputs)[-1])
        ridxs = tf.random.shuffle(idxs)[:self.sample_to]
        ridxs = tf.expand_dims(ridxs,axis=0)
        ridxs = tf.tile(ridxs, (tf.shape(inputs)[0],1))
        ridxs = tf.expand_dims(ridxs,axis=-1)
        rinput = tf.gather_nd(inputs, ridxs,batch_dims=1)
        rinput = tf.reshape(rinput, (-1,self.sample_to))
        return rinput
    
    
############# Some layers for convenience ############

class PrintMeanAndStd(tf.keras.layers.Layer):
    def __init__(self,
                 print_mean=True,
                 print_std=True,
                 print_shape=True,
                 **kwargs):
        super(PrintMeanAndStd, self).__init__(**kwargs)
        self.print_mean = print_mean
        self.print_std = print_std
        self.print_shape = print_shape
        
    def get_config(self):
        config = {'print_mean': self.print_mean,
                  'print_std': self.print_std,
                  'print_shape': self.print_shape}
        base_config = super(PrintMeanAndStd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        return input_shapes
        
    def call(self, inputs):
        try:
            if self.print_mean:
                tf.print(self.name,'mean',tf.reduce_mean(inputs),summarize=100)
            if self.print_std:
                tf.print(self.name,'std',tf.math.reduce_std(inputs),summarize=100)
            if self.print_shape:
                tf.print(self.name,'shape',inputs.shape,summarize=100)
        except Exception as e:
            print('exception in',self.name)
            #raise e
        return inputs


class ElementScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ElementScaling, self).__init__(**kwargs)

    def get_config(self):
        return super(ElementScaling, self).get_config()
    
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        return input_shapes
    
    def build(self, input_shape):
        shape = [1 for _ in range(len(input_shape)-1)]+[input_shape[-1]]
        self.scales = self.add_weight(name = 'scales',shape = shape, 
                                    initializer = 'ones', trainable = True) 
        
        super(ElementScaling, self).build(input_shape)
        
    def call(self, inputs, training=None):
        
        return inputs * self.scales


class ConditionalNormalizationLayer(tf.keras.layers.Layer):
    """
    Layer that normalizes the input to zero mean and unit variance
    This is written for inputs that are made out of two types of data
    such that they can be normalized separately (e.g. hits and track).
    The call function takes the condtion as an argument (e.g. is_track == 1)
    - mean0: shape (n_features,)
        array with the mean of the first type of data
    - mean1: shape (n_features,)
        array with the mean of the second type of data
    - std0: shape (n_features,)
        array with the standard deviation of the first type of data
    - std1: shape (n_features,)
        array with the standard deviation of the second type of data
    """
    def __init__(self,
        mean0=[
            2.9607140e-02,  2.6290069e+00,  0.0000000e+00,  1.5184632e-01, 3.3951630e+02,
            -1.4371893e+00, -9.8884726e-01,  3.3424982e+02, 0.0000000e+00,  8.6794233e-01],
        mean1=[
            1.2764481e+01,  2.2522159e+00,  1.0000000e+00,  2.2571583e-01, 3.2457993e+02,
            -1.1017900e+01, -3.2713771e+00,  3.1500000e+02, 0.0000000e+00,  0.0000000e+00],
        std0=[
            0.04933382, 0.3124265, 0.0, 0.05499146, 11.042463,
            35.32781, 34.771965, 9.525104, 0.0, 0.34948865],
        std1=[
            31.369707, 0.3951621, 0.0, 0.08696821, 7.2839437,
            50.60379, 49.66359, 0.0, 0.0, 0.0 ],
        eps=1e-8):
        super(ConditionalNormalizationLayer, self).__init__()
        self.mean0 = tf.cast(mean0, tf.float32)
        self.mean1 = tf.cast(mean1, tf.float32)
        self.std0 = tf.cast(std0, tf.float32)
        self.std1 = tf.cast(std1, tf.float32)
        self.eps = tf.cast(eps, tf.float32)

    def call(self, inputs):
        """
        inputs: [data, condition]
        - data: tensor of shape (batch_size, n_features)
        - condition: tensor of shape (batch_size, 1) or (batch_size,)
        """
        data, condition = inputs

        condition = tf.reshape(condition, (-1, 1))
        condition = tf.cast(condition, tf.bool)

        mean = tf.where(condition, self.mean1, self.mean0)
        std = tf.where(condition, self.std1, self.std0)

        data = (data - mean) / (std + self.eps)
        return data


class ConditionalBatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()


    """
    def build(self, input_shape):

        # Call the build function for the batch normalization layers
        self.bn0.build(input_shape[0])
        self.bn1.build(input_shape[0])

        # Call the build function of the parent class
        super(ConditionalBatchNorm, self).build(input_shape)
    """

    def call(self, inputs):
        x, condition = inputs[0], inputs[1] 
        condition = tf.cast(condition > 0.5, tf.bool)
        idx0 = tf.where(condition)
        idx1 = tf.where(tf.logical_not(condition))
        x0 = tf.gather_nd(x, idx0)
        x1 = tf.gather_nd(x, idx1)
        x0 = tf.expand_dims(x0, axis=1)  # add a new dimension to make it 2D
        x1 = tf.expand_dims(x1, axis=1)  # add a new dimension to make it 2D
        x0 = self.bn0(x0)
        x1 = self.bn1(x1)
        x0 = tf.squeeze(x0, axis=1)  # remove the added dimension to make it 1D again
        x1 = tf.squeeze(x1, axis=1)  # remove the added dimension to make it 1D again
        x = tf.tensor_scatter_nd_update(x, idx0, x0)
        x = tf.tensor_scatter_nd_update(x, idx1, x1)
        return x


class ConditionalBatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, N=32, **kwargs):
        super(ConditionalBatchEmbedding, self).__init__(**kwargs)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.d0 = tf.keras.layers.Dense(N, activation='elu')
        self.d1 = tf.keras.layers.Dense(N, activation='elu')

    """
    def build(self, input_shape):
        # Initialize the weights for the dense layers
        self.d0.build(input_shape[0])
        self.d1.build(input_shape[0])

        # Call the build function for the batch normalization layers
        self.bn0.build((input_shape[0][0], 32))
        self.bn1.build((input_shape[0][0], 32))

        # Call the build function of the parent class
        super(ConditionalBatchEmbedding, self).build(input_shape)
    """


    def call(self, inputs):
        x, condition = inputs[0], inputs[1]
        condition = tf.cast(condition > 0.5, tf.bool)  # Convert condition to boolean
        idx0 = tf.where(condition)
        idx1 = tf.where(tf.logical_not(condition))
        x0 = self.d0(x)
        x1 = self.d1(x)
        x0 = tf.gather_nd(x, idx0)
        x1 = tf.gather_nd(x, idx1)
        x0 = tf.expand_dims(x0, axis=1)  # add a new dimension to make it 2D
        x1 = tf.expand_dims(x1, axis=1)  # add a new dimension to make it 2D
        x0 = self.bn0(x0)
        x1 = self.bn1(x1)
        x0 = tf.squeeze(x0, axis=1)  # remove the added dimension to make it 1D again
        x1 = tf.squeeze(x1, axis=1)  # remove the added dimension to make it 1D again
        x = tf.tensor_scatter_nd_update(x, idx0, x0)
        x = tf.tensor_scatter_nd_update(x, idx1, x1)
        return x

        
class GooeyBatchNorm(LayerWithMetrics):
    def __init__(self,
                 viscosity=0.2,
                 fluidity_decay=1e-4,
                 max_viscosity=0.99,
                 epsilon=1e-4,
                 print_viscosity=False,
                 variance_only=False,
                 soft_mean=False,
                 soften_update: float = 0.,
                 soft_mean_hardness=3.,
                 soft_mean_turn_on=6.,
                 learn = False, #this is set to false for compatibility reasons, however USE TRUE AS DEFAULT!
                 **kwargs):
        super(GooeyBatchNorm, self).__init__(**kwargs)
        
        assert viscosity >= 0 and viscosity <= 1.
        assert fluidity_decay >= 0 and fluidity_decay <= 1.
        assert max_viscosity >= viscosity
        assert soften_update >= 0
        
        self.fluidity_decay = fluidity_decay
        self.max_viscosity = max_viscosity
        self.viscosity_init = viscosity
        self.epsilon = epsilon
        self.print_viscosity = print_viscosity
        self.soften_update = soften_update
        self.variance_only = variance_only
        self.soft_mean = soft_mean
        self.soft_mean_hardness = soft_mean_hardness
        self.soft_mean_turn_on = soft_mean_turn_on
        self.learn = learn

        if soften_update > 0:
            print(self.name,'has been configured for soften_update. Function not implemented yet. Will have no effect.')
        
    def get_config(self):
        config = {'viscosity': self.viscosity_init,
                  'fluidity_decay': self.fluidity_decay,
                  'max_viscosity': self.max_viscosity,
                  'epsilon': self.epsilon,
                  'print_viscosity': self.print_viscosity,
                  'variance_only': self.variance_only,
                  'soft_mean': self.soft_mean,
                  'soften_update': self.soften_update,
                  'soft_mean_hardness': self.soft_mean_hardness,
                  'soft_mean_turn_on': self.soft_mean_turn_on,
                  'learn': self.learn
                  }
        base_config = super(GooeyBatchNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        return input_shapes
    
    
    def build(self, input_shapes):
        
        #shape = (1,)+input_shapes[0][1:]
        shape = (1,)+input_shapes[1:]
        
        self.mean = self.add_weight(name = 'mean',shape = shape, 
                                    initializer = 'zeros', trainable =  self.learn) 
        
        self.variance = self.add_weight(name = 'variance',shape = shape, 
                                    initializer = 'ones', trainable =  self.learn) 
        
        self.viscosity = tf.Variable(initial_value=self.viscosity_init, 
                                         name='viscosity',
                                         trainable=False,dtype='float32')
            
        super(GooeyBatchNorm, self).build(input_shapes)
    
    def _calc_update(self, old, new, training, ismean):
        delta = new-old
        update = old + (1. - self.viscosity)*delta
        return tf.keras.backend.in_train_phase(update,old,training=training)

    def _calc_soft_diff(self,x):
        turnon = self.soft_mean_turn_on
        hardness = self.soft_mean_hardness
        possig = tf.nn.sigmoid(hardness*(x-turnon))
        negsig = tf.nn.sigmoid(-hardness*(x+turnon))
        mod = tf.where(x>0,possig,negsig)
        return x*mod
    
    def _update_viscosity(self, training):
        if self.fluidity_decay > 0:
            newvisc = self.viscosity + (self.max_viscosity - self.viscosity)*self.fluidity_decay
            newvisc = tf.keras.backend.in_train_phase(newvisc,self.viscosity,training=training)
            tf.keras.backend.update(self.viscosity,newvisc)
            if self.print_viscosity:
                tf.print(self.name, 'viscosity',newvisc)

    def call(self, inputs, training=None):
        #x, _ = inputs
        x = inputs
        
        #update only if trainable flag is set, AND in training mode
        if self.trainable and not self.learn:
            
            currentmean = tf.reduce_mean(x,axis=0,keepdims=True) #FIXME
            newmean = currentmean
            if self.soft_mean: #soft diff between current mean and zero
                newmean = self._calc_soft_diff(newmean)
            update = self._calc_update(self.mean,newmean,training,ismean=True)
            tf.keras.backend.update(self.mean,update)
            
            #use the actual mean here, not self.mean
            newvar = tf.math.reduce_std(x-currentmean,axis=0,keepdims=True) #FIXME
            update = self._calc_update(self.variance,newvar,training,ismean=False)
            tf.keras.backend.update(self.variance,update)
            
            #increase viscosity
            self._update_viscosity(training)
                    
        elif self.trainable and self.learn: #trainable is anyway given by applying the loss or not
            currentmean = tf.reduce_mean(x,axis=0,keepdims=True)
            currentmean = tf.stop_gradient(currentmean)
            currentmean = tf.where(tf.math.is_finite(currentmean),currentmean, self.mean)#protect against empty batches

            self.add_loss(10.*(1.01-self.viscosity) * tf.reduce_mean(tf.abs(self.mean-currentmean)))
            
            newvar = tf.math.reduce_std(x-currentmean,axis=0,keepdims=True)
            newvar = tf.stop_gradient(newvar)
            newvar = tf.where(tf.math.is_finite(newvar),newvar, self.variance)#protect against empty batches

            self.add_loss(10.*(1.01-self.viscosity) * tf.reduce_mean(tf.abs(self.variance-newvar)))
            
            self._update_viscosity(training)
            
        else:
            currentmean = self.mean
            newvar = self.variance
                        
        self.add_prompt_metric(tf.reduce_mean(newvar), self.name+'_variance')
        self.add_prompt_metric(tf.reduce_mean(currentmean), self.name+'_mean')
        
        #apply
        x -= self.mean
        x = tf.math.divide_no_nan(x, tf.abs(self.variance) + self.epsilon)
        if self.variance_only:
            x += self.mean
        
        return x
    

class ScaledGooeyBatchNorm(GooeyBatchNorm):
    
    def __init__(self, **kwargs):
        super(ScaledGooeyBatchNorm, self).__init__(**kwargs)
        
    def build(self, input_shapes):
        shape = (1,)+input_shapes[1:]
        
        self.bias = self.add_weight(name = 'bias',shape = shape, 
                                    initializer = 'zeros', trainable = self.trainable) 
        
        self.gamma = self.add_weight(name = 'gamma',shape = shape, 
                                    initializer = 'ones', trainable = self.trainable) 
        
        super(ScaledGooeyBatchNorm, self).build(input_shapes)
    
    def call(self, inputs, training=None):
        out = super(ScaledGooeyBatchNorm, self).call(inputs, training)
        return out*self.gamma + self.bias
    


class SignedScaledGooeyBatchNorm(ScaledGooeyBatchNorm):
    def call(self, inputs, training=None):
        s,v = tf.sign(inputs), tf.abs(inputs)
        out = super(SignedScaledGooeyBatchNorm, self).call(v, training)
        return s*out

class ScaledGooeyBatchNorm2(LayerWithMetrics):
    def __init__(self, 
                 viscosity=0.01,
                 fluidity_decay=1e-4,
                 max_viscosity=0.99,
                 no_gaus = True,
                 epsilon=1e-2,
                 **kwargs):
        '''
        Input features (or [features, condition]), output: normed features
        
        Options:
        - viscosity:      starting viscosity, where the update for the next batch is
                          update = old + (1. - viscosity)*(new - old).
                          This means, low values will create strong updates with each batch
        - fluidity_decay: 'thickening' of the viscosity (see scripts/gooey_plot.py for visualisation)
        - no_gaus:         do not take variance but take mean difference to mean. 
                           Better for non-gaussian inputs and much more robust.
        - epsilon:         when dividing, added to the denominator (should not require adjustment)
        '''
        
        super(ScaledGooeyBatchNorm2, self).__init__(**kwargs)
        
        assert viscosity >= 0 and viscosity <= 1.
        assert fluidity_decay >= 0 and fluidity_decay <= 1.
        assert max_viscosity >= viscosity
        assert epsilon > 0.
        
        self.fluidity_decay = fluidity_decay
        self.max_viscosity = max_viscosity
        self.viscosity_init = viscosity
        self.epsilon = epsilon
        self.no_gaus = no_gaus
        
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        if isinstance(input_shapes,list):
            return input_shapes[0]
        return input_shapes
              
    def get_config(self):
        config = {'viscosity': self.viscosity_init,
                  'fluidity_decay': self.fluidity_decay,
                  'max_viscosity': self.max_viscosity,
                  'epsilon': self.epsilon,
                  'no_gaus': self.no_gaus
                  }
        base_config = super(ScaledGooeyBatchNorm2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def build(self, input_shapes):
        
        #shape = (1,)+input_shapes[0][1:]
        if isinstance(input_shapes,list):
            shape = (1,)+input_shapes[0][1:]
        else:
            shape = (1,)+input_shapes[1:]
        
        self.mean = self.add_weight(name = 'mean',shape = shape, 
                                    initializer = 'zeros', trainable =  False) 
        self.den = self.add_weight(name = 'den',shape = shape, 
                                    initializer = 'ones', trainable =  False)
        self.viscosity = tf.Variable(initial_value=self.viscosity_init, 
                                         name='viscosity',
                                         trainable=False,dtype='float32')
        
        self.bias = self.add_weight(name = 'bias',shape = shape, 
                                    initializer = 'zeros', trainable = self.trainable) 
        
        self.gamma = self.add_weight(name = 'gamma',shape = shape, 
                                    initializer = 'ones', trainable = self.trainable) 
            
        super(ScaledGooeyBatchNorm2, self).build(input_shapes)
    
    def _m_mean(self, x, mask):
        x = tf.reduce_sum( x * mask ,axis=0,keepdims=True)
        norm = tf.abs(tf.reduce_sum(mask, axis=0,keepdims=True)) + self.epsilon
        return tf.math.divide_no_nan(x, norm)
                    
    def _calc_mean_and_protect(self, x, mask, default):
        x = self._m_mean(x, mask)
        x = tf.where(tf.math.is_finite(x), x, default)#protect against empty batches
        return x
    
    def _calc_update(self, old, new, training, visc=None):
        if visc is None:
            visc = self.viscosity
        delta = new-old
        update = old + (1. - visc)*delta
        return tf.keras.backend.in_train_phase(update,old,training=training)
    
    def _update_viscosity(self, training):
        if self.fluidity_decay > 0:
            newvisc = self.viscosity + (self.max_viscosity - self.viscosity)*self.fluidity_decay
            newvisc = tf.keras.backend.in_train_phase(newvisc,self.viscosity,training=training)
            tf.keras.backend.update(self.viscosity,newvisc)
    
    def _calc_out(self, x_in, cond):
        
        ngmean = tf.stop_gradient(self.mean)
        ngden = tf.stop_gradient(self.den)
        
        out = (x_in - ngmean) / (tf.abs(ngden) + self.epsilon)
        out = out*self.gamma + self.bias
        return tf.where(cond>0.5,  out, x_in)
    
    def call(self, inputs, training=None):
        if isinstance(inputs,list):
            x_in, cond = inputs
            cond = tf.where(cond > 0.5, tf.ones_like(cond), 0.) #make sure it's ones and zeros
        else:
            x_in = inputs
            cond = tf.ones_like(x_in[...,0:1])
        
        if not self.trainable:
            return self._calc_out(x_in, cond)
        
        x = tf.stop_gradient(x_in) #stop feat gradient
        #x = x_in #maybe don't stop the gradient?
        x_m = self._calc_mean_and_protect(x, cond, self.mean)
        
        diff_to_mean = tf.abs(x - self.mean) #self.mean or x_m
        if not self.no_gaus:
            diff_to_mean = diff_to_mean**2
            
        x_std = self._calc_mean_and_protect(diff_to_mean, cond, self.den) 
        
        if not self.no_gaus:
            x_std = tf.sqrt(x_std + self.epsilon)
            
        
        update = self._calc_update(self.mean,x_m,training)
        tf.keras.backend.update(self.mean, update)
        
        update = self._calc_update(self.den,x_std,training)
        tf.keras.backend.update(self.den, update)
        
        self._update_viscosity(training)
        
        out = self._calc_out(x_in, cond)
        
        return out
    
        
    
class ConditionalScaledGooeyBatchNorm(LayerWithMetrics):
    def __init__(self,**kwargs):
        '''
        Inputs (list):
        - features
        - condition (threshold > 0.5)
        
        Applies normalisation to two different classes within the array.
        The condition (float) selects which is applied to which entry.
        
        Options: see ScaledGooeyBatchNorm2 options, will be passed as kwargs
        '''
        
        super(ConditionalScaledGooeyBatchNorm, self).__init__(**kwargs)
        if 'name' in kwargs.keys():
            kwargs.pop('name')
        
        with tf.name_scope(self.name + "/1/"):
            self.bn_a = ScaledGooeyBatchNorm2(name=self.name+'_bn_a',**kwargs)
        with tf.name_scope(self.name + "/2/"):
            self.bn_b = ScaledGooeyBatchNorm2(name=self.name+'_bn_b',**kwargs)
        
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        return self.bn_a.compute_output_shape(input_shapes)
              
    def build(self, input_shapes):
        
        with tf.name_scope(self.name + "/1/"):
            self.bn_a.build(input_shapes)
        with tf.name_scope(self.name + "/2/"):
            self.bn_b.build(input_shapes)
            
        super(ConditionalScaledGooeyBatchNorm, self).build(input_shapes)
        
    def call(self, inputs, training=None):
        x, cond = inputs
        cond = tf.where(cond > 0.5, tf.ones_like(cond),  0.) #make sure it's ones and zeros
        
        x_a = self.bn_a([x, cond],training = training)
        x_b = self.bn_b([x, 1.-cond], training = training)
        
        return tf.where(cond>0.5, x_a, x_b)
            

class ProcessFeatures(tf.keras.layers.Layer):
    def __init__(self,
                 newformat=True,#compat can be restored but default is new format
                 **kwargs):
        """
        Inputs are: 
         - Features
         
        Call will return:
         - processed features
        
        will apply some simple fixed preprocessing to the standard TrainData_OC features
        
        """
        
        from datastructures import TrainData_NanoML
        self.td=TrainData_NanoML()
        self.newformat = newformat
        super(ProcessFeatures, self).__init__(**kwargs)
        
    
    def get_config(self):
        config = {'newformat': self.newformat}
        base_config = super(ProcessFeatures, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        '''
        'recHitEnergy',
            'recHitEta',
            'isTrack',
            'recHitTheta',
            'recHitR',
            'recHitX',
            'recHitY',
            'recHitZ',
            'recHitTime',
            'recHitHitR'  
        '''
        feat = None
        fdict = self.td.createFeatureDict(inputs, False)
        
        if not self.newformat:
            #please make sure in TrainData_OC that this is consistent
            fdict['recHitR'] /= 100.
            fdict['recHitX'] /= 100.
            fdict['recHitY'] /= 100.
            fdict['recHitZ'] = (tf.abs(fdict['recHitZ'])-400)/100.
            fdict['recHitEta'] = tf.abs(fdict['recHitEta'])
            fdict['recHitTheta'] = 2.*tf.math.atan(tf.exp(-fdict['recHitEta']))
            fdict['recHitTime'] = tf.nn.relu(fdict['recHitTime'])/10. #remove -1 default
            allf = []
            for k in fdict:
                allf.append(fdict[k])
            feat = tf.concat(allf,axis=-1)
            
            mean = tf.constant([[0.0740814656, 2.46156192, 0., 0.207392946, 3.55599976, 0.0609507263, 
                             -0.00401970092, -0.515379727, 0.0874295086]])
            std =  tf.constant([[0.299679846, 0.382687777, 1., 0.0841238275, 0.250777304, 0.485394388, 
                                 0.518072903,     0.240222782, 0.194716245]])
            feat -= mean
            feat /= std
            
            return feat
            
        else: #new std format
            fdict['recHitEta'] = tf.abs(fdict['recHitEta'])
            fdict['recHitZ'] =  tf.abs(fdict['recHitZ'])
            fdict['recHitTheta'] = 2.*tf.math.atan(tf.exp(-fdict['recHitEta']))
            allf = []
            for k in fdict:
                allf.append(fdict[k])
            feat = tf.concat(allf,axis=-1)
        
            mean = tf.constant([[ 3.95022651e-02,  2.46088736e+00,  
                                 0.00000000e+00,
                                  1.56797723e+00,  3.54114793e+02, 
                                 0.,#x
                                 0.,#y  
                                  3.47267313e+02, 
                                 -3.73342582e-01,
                                  7.61663214e-01]])
            std =  tf.constant([[ 0.16587503,  0.36627547,  1.        ,  
                                 1.39035478, 22.55941696,
                                 50., 50., 
                                 21.75297722,  
                                 1.89301789,  
                                 0.14808707]])
            
            feat -= mean
            feat /= std
            
            return feat
        
        ##old format below
        
        
    


class ManualCoordTransform(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(ManualCoordTransform, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def build(self, input_shapes): #pure python
        assert input_shapes[-1] == 3 #only works for x y z
        super(ManualCoordTransform, self).build(input_shapes)
    
    
    def _calc_r(self, x,y,z):
        return tf.math.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-6)
    
    def _calc_abseta(self, x, y, z):
        r = self._calc_r(x,y,0)
        return -1 * tf.math.log(r / tf.abs(z + 1e-3) / 2.)
    
    def _calc_phi(self, x, y):
        return tf.math.atan2(x, y)
    
    def call(self, inputs):
        x,y,z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        z = tf.abs(z) #take absolute
        eta = self._calc_abseta(x,y,z) #-> z
        r = self._calc_r(x, y, z) #-> cylindrical r
        phi = self._calc_phi(x, y)
        newz = eta
        newx = tf.math.cos(phi)*r
        newy = tf.math.sin(phi)*r
        
        coords = tf.concat([newx,newy,newz],axis=-1)
        coords /= tf.constant([[262.897095, 246.292236, 0.422947705]])
        return coords


class DirectedGraphBuilder(tf.keras.layers.Layer):
    def __init__(self,strength=1., cutoff=1., **kwargs):
        '''
        Builds a directed graph by increasing the distances to neighbouring vertices
        locally, if they have smaller score.
        
        input is: distances, neighbour indices, score
        Output is distances, neighbour indices
        
        score is strictly between 0 and 1
        
        strength and cut-off don't do anything yet
        
        '''
        self.strength=strength
        self.cutoff=cutoff
        super(DirectedGraphBuilder, self).__init__(**kwargs)
    
    def get_config(self):
        config = {'strength': self.strength,
                  'cutoff': self.cutoff}
        base_config = super(DirectedGraphBuilder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
        
    def build(self, input_shapes): #pure python
        super(DirectedGraphBuilder, self).build(input_shapes)
        
    def compute_output_shape(self, input_shapes):
        return input_shapes[0],input_shapes[1]
    
    def call(self, inputs):
        assert len(inputs) == 3
        dist, nidx, score = inputs
        neighscores = SelectWithDefault(nidx, score, -1e4)
        
        #between <=1
        diff = score-tf.reduce_max(neighscores[:,1:],axis=1)
        
        noneigh = tf.zeros_like(nidx[:,1:])-1
        noneigh = tf.concat([nidx[:,0:1],noneigh],axis=-1)
        
        #where diff<0
        dist = tf.where(diff<0,0.,dist)
        nidx = tf.where(diff<0,noneigh,nidx)
        
        return dist,nidx

    
class NeighbourCovariance(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        """
        Inputs: 
          - coordinates (Vin x C)
          - distance squared (Vout x K)
          - features (Vin x F)
          - neighbour indices (Vout x K)
          
        Returns concatenated  (Vout x { F*C^2 + F*C})
          - feature weighted covariance matrices (lower triangle) (Vout x F*C^2)
          - feature weighted means (Vout x F*C)
          
        """
        super(NeighbourCovariance, self).__init__(**kwargs)
        self.outshapes=None
    
    
    def build(self, input_shapes): #pure python
        super(NeighbourCovariance, self).build(input_shapes)
        
    @staticmethod
    def raw_get_cov_shapes(input_shapes):
        coordinates_s, features_s = None, None
        if len(input_shapes) == 4:
            coordinates_s, _, features_s, _ = input_shapes
        else:
            coordinates_s, features_s, _ = input_shapes
        nF = features_s[1]
        nC = coordinates_s[1]
        covshape = nF*nC**2 #int(nF*(nC*(nC+1)//2))
        return nF, nC, covshape
    
    @staticmethod
    def raw_call(coordinates, distsq, features, n_idxs):
        cov, means = NeighbourCovarianceOp(coordinates=coordinates, 
                                           distsq=10. * distsq,#same as gravnet scaling
                                         features=features, 
                                         n_idxs=n_idxs)
        return cov, means
    
    def call(self, inputs):
        coordinates, distsq, features, n_idxs = None, None, None, None 
        if len(inputs) == 4:
            coordinates, distsq, features, n_idxs = inputs
        else:
            coordinates, features, n_idxs = inputs
            distsq = tf.zeros_like(n_idxs,dtype='float32')
        
        cov,means = NeighbourCovariance.raw_call(coordinates, distsq, features, n_idxs)
        
        nF, nC, covshape = NeighbourCovariance.raw_get_cov_shapes([s.shape for s in inputs])
        
        cov = tf.reshape(cov, [-1, covshape])
        means = tf.reshape(means, [-1, nF*nC])
        
        return tf.concat([cov,means],axis=-1)
        
class NeighbourApproxPCA(tf.keras.layers.Layer):
    def __init__(self,hidden_nodes=[32,32,32], **kwargs):
        """
        Inputs: 
          - coordinates (Vin x C)
          - distsq (Vin x K)
          - features (Vin x F)
          - neighbour indices (Vin x K)
          
        Returns concatenated:
          - feature weighted covariance matrices (lower triangle) (Vout x F*C**2)
          - feature weighted means (Vout x F*C)
          
        """
        super(NeighbourApproxPCA, self).__init__(**kwargs)
        
        self.hidden_dense=[]
        self.hidden_nodes = hidden_nodes
        
        for i in range(len(hidden_nodes)):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.hidden_dense.append( tf.keras.layers.Dense( hidden_nodes[i],activation='elu'))
                
        self.nF = None
        self.nC = None
        self.covshape = None
        
        
        print('NeighbourApproxPCA: Warning. This layer is still very sensitive to the input normalisations and the learning rates.')
        
        
    def get_config(self):
        config = {'hidden_nodes': self.hidden_nodes}
        base_config = super(NeighbourApproxPCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shapes): #pure python
        nF, nC, covshape = NeighbourCovariance.raw_get_cov_shapes(input_shapes)
        self.nF = nF
        self.nC = nC
        self.covshape = covshape
        
        #build is actually called for this layer
        dshape=(None, None, nC**2)
        for i in range(len(self.hidden_dense)):
            with tf.name_scope(self.name + "/1/" + str(i)):
                self.hidden_dense[i].build(dshape)
                dshape = (None, None, self.hidden_dense[i].units)
                
                
        super(NeighbourApproxPCA, self).build(input_shapes)  
        
    def compute_output_shape(self, input_shapes):
        nF, _,_ = NeighbourCovariance.raw_get_cov_shapes(input_shapes)
        return (None, nF * self.hidden_dense[-1].units)

    def call(self, inputs):
        coordinates, distsq, features, n_idxs = inputs
        
        cov, means = NeighbourCovarianceOp(coordinates=coordinates, 
                                           distsq=10. * distsq,#same as gravnet scaling
                                         features=features, 
                                         n_idxs=n_idxs)
        for d in self.hidden_dense:
            cov = d(cov)
        
        cov = tf.reshape(cov, [-1, self.nF * self.hidden_dense[-1].units ])
        means = tf.reshape(means, [-1, self.nF*self.nC])
        
        return tf.concat([cov,means],axis=-1)
            
        
class ApproxPCA(tf.keras.layers.Layer):
    # New implementation of NeighbourApproxPCA 
    # Old version is kept to not cause unexpected behaviour
    def __init__(self, size='small', 
                 base_path=os.environ.get('HGCALML') + '/HGCalML_data/pca/pretrained/', 
                 empty=False, # If true, weights are not loaded for testing
                 **kwargs):
        """
        Inputs: 
          - coordinates (Vin x C)
          - distsq (Vin x K)
          - features (Vin x F)
          - neighbour indices (Vin x K)
          
        Returns:
          - per feature: Approximated PCA of weighted coordinates' covariance matrices
            (V, F, C**2)
          - if enabled: feature weighted means (Vout x F*C)
          
        """
        super(ApproxPCA, self).__init__(**kwargs)
        assert size.lower() in ['small', 'medium', 'large']\
            , "size must be 'small', 'medium', or 'large'!"
        self.size = size.lower()
        self.base_path = base_path
        self.layers = []
        self.empty = empty
        
        print("This layer uses the pretrained PCA approximation layers found in `pca/pretrained`")
        print("It is still somewhat experimental and subject to change!")
        
        
    def get_config(self):
        base_config = super(ApproxPCA, self).get_config()
        init_config = {'base_path': self.base_path, 'size': self.size}
        if not self.config:
            self.config = {}
        config = dict(list(base_config.items()) + list(init_config.items()) + list(self.config.items()))
        return config

    
    def build(self, input_shapes): #pure python
        nF, nC, _ = NeighbourCovariance.raw_get_cov_shapes(input_shapes)
        self.nF = nF
        self.nC = nC
        self.covshape = nF * nC * nC
        self.counter = 0
        self.train_layers = not self.empty

        self.path = self.base_path + f"{str(self.nC)}D/{self.size}/"
        assert os.path.exists(self.path), f"path: {self.path} not found!"
        with open(self.path + 'config.yaml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
        # Build model and load weights
        inputs = tf.keras.layers.Input(shape=(self.nC**2,))
        x = inputs
        nodes = self.config['nodes']
        for i, node in enumerate(nodes):
            x = tf.keras.layers.Dense(node, activation='elu')(x)
        outputs = tf.keras.layers.Dense(self.nC**2)(x)
        with tf.name_scope(self.name + '/pca/model'):
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        if not self.empty:
            self.model.load_weights(self.path)
            self.model.trainable = self.train_layers
        self.model.summary()

        for i in range(len(nodes) + 1):
            with tf.name_scope(self.name + "/1/" + str(i)):
                # layer = model.layers[i+1]
                if i == 0:
                    # first entry, input layer
                    input_dim = [None, self.nC**2]  # Not sure if I need the batch dimension
                else:
                    input_dim = [None, nodes[i-1]]
                if i == len(nodes):
                    # Last entry, output layer, no activation
                    output_dim = self.nC**2
                    layer = tf.keras.layers.Dense(units=output_dim, trainable=self.train_layers)
                else:
                    output_dim = nodes[i]
                    layer = tf.keras.layers.Dense(units=output_dim, activation='elu', trainable=self.train_layers)
                layer.build(input_dim)
                weights = self.model.layers[i+1].get_weights()[0]
                bias = self.model.layers[i+1].get_weights()[1]
                if not self.empty:
                    layer.set_weights([weights, bias])
                self.layers.append(layer)

        super(ApproxPCA, self).build(input_shapes)  
        del self.model # To avoid confusing error messages about missing gradients
        
        
    def compute_output_shape(self):
        out_shape = (None, self.nF, self.nC * self.nC)
        return out_shape


    def call(self, inputs):
        self.counter += 1
        ReturnMean = False  
        coordinates, distsq, features, n_idxs = inputs
        
        cov, means = NeighbourCovarianceOp(coordinates=coordinates, 
                                           distsq=10. * distsq, #same as gravnet scaling
                                           features=features, 
                                           n_idxs=n_idxs)
        
        means = tf.reshape(means, [-1, self.nF*self.nC])
        cov = tf.reshape(cov, shape=(-1, self.nC**2))

        x = cov
        for i, layer in enumerate(self.layers):
            x = layer(x)
        approxPCA = tf.reshape(x, shape=(-1, self.nF * self.nC**2))

        if ReturnMean:
            return tf.concat([approxPCA, means], axis=-1)
        else:
            return approxPCA
    
    
    
class LocalDistanceScaling (LayerWithMetrics):
    def __init__(self,
                 max_scale = 10,
                 **kwargs):
        """
        Inputs: 
        - distances (V x N)
        - scaling (V x 1): (linear activation)
        
        Returns:
        distances * scaling : V x N x 1
        scaling is bound to be within 1/max_scale and max_scale.
        """
        super(LocalDistanceScaling, self).__init__(**kwargs)
        self.max_scale = float(max_scale)
        #some helpers
        self.c = 1. - 1./max_scale
        self.b = max_scale/(2.*self.c)
        
        
        #derivative sigmoid = sigmoid(x) (1- sigmoid(x))
        #der sig|x=0 = 1/4
        #derivate of sigmoid (ax) = a (der sig)(ax)
        #
        
        
    def get_config(self):
        config = {'max_scale': self.max_scale}
        base_config = super(LocalDistanceScaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[1]
    
    @staticmethod
    def raw_call(dist,scale,a,b,c):
        #the derivative is continuous at 0
        scale_pos = a*tf.math.sigmoid(scale) +  1. - a/2.
        scale_neg = 2. * c * tf.math.sigmoid(b*scale) + 1. - c
        scale = tf.where(scale>=0,scale_pos, scale_neg)
        return dist*scale, scale
    
    def call(self, inputs):
        dist,scale = inputs
        newdist,scale = LocalDistanceScaling.raw_call(dist,scale,self.max_scale,self.b,self.c)
        self.add_prompt_metric(tf.reduce_mean(scale), self.name+'_dist_scale')
        self.add_prompt_metric(tf.math.reduce_std(scale), self.name+'_var_dist_scale')
        return newdist
    

 
        
class CreateGlobalIndices(tf.keras.layers.Layer):
    def __init__(self, **kwargs):    
        """
        
        This layer just simply creates global vertex indices per batch.
        These can e.g. be used in conjunction with the selection given by LocalClustering
        to determine which vertices out of all global ones were selected.
        
        Inputs are:
         - a tensor to determine the total dimensionality in the first dimension
         
        """  
        if 'dynamic' in kwargs:
            super(CreateGlobalIndices, self).__init__(**kwargs)
        else:
            super(CreateGlobalIndices, self).__init__(dynamic=False,**kwargs)
        
    def compute_output_shape(self, input_shape):
        s = (input_shape[0],1)
        return s
    
    
    def compute_output_signature(self, input_signature):
        print('>>>>>CreateGlobalIndices input_signature',input_signature)
        input_shape = tf.nest.map_structure(check_type_return_shape, input_signature)
        output_shape = self.compute_output_shape(input_shape)
        return [tf.TensorSpec(dtype=tf.int32, shape=output_shape[i]) for i in range(len(output_shape))]
    
    def build(self, input_shapes):
        super(CreateGlobalIndices, self).build(input_shapes)
    
    def call(self, inputs):
        ins = tf.cast(inputs*0.,dtype='int32')[:,0:1]
        add = tf.expand_dims(tf.range(tf.shape(inputs)[0],dtype='int32'), axis=1)
        return ins+add
    
    
class WeightedNeighbourMeans(tf.keras.layers.Layer): 
    def __init__(self, 
                 distweight: bool=False,
                 **kwargs):
        """
        Options:
        - distweight: weight by the usual GravNet distance weighting (exp(-10 dsq))
        
        Input:
        - features/coords to be weighted meaned ;)
        - weights (V x 1) only (this increases mem consumption)
        - distances (only used if distweight=True)
        - neighbour indices
        
        Output:
        - difference between initial and weighted neighbour means
        
        """
        
        super(WeightedNeighbourMeans, self).__init__(**kwargs) 
        self.distweight = distweight 
        
    def get_config(self):
        config = {'distweight': self.distweight}
        base_config = super(WeightedNeighbourMeans, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def call(self, inputs):
        assert len(inputs)==4
        feat, weights, dist, nidx = inputs
        
        nweights = SelectWithDefault(nidx, weights, 0.)[:,:,0]#remove last 1 dim
        
        if self.distweight:
            nweights*=tf.exp(-10.*dist)#needs to be before log
        
        nweights = tf.nn.relu(nweights)+1e-6#secure
        
        f,_ = AccumulateLinKnnSumw(nweights, feat, nidx)
        out = f-feat
        return out
        
class WeightFeatures(tf.keras.layers.Layer): 
    def __init__(self, 
                 **kwargs):
        """
        Input:
        - features
        - features to compute weights
        
        Output:
        - weighted features. Weights receive an activation of 2*tanh and a bias starting at weight=1
        
        """    
        super(WeightFeatures, self).__init__(**kwargs) 
        
        with tf.name_scope(self.name + "/1/"):
            self.weight_dense = tf.keras.layers.Dense(1,activation='tanh',bias_initializer = 'ones')

    def build(self, input_shapes):
        input_shape = input_shapes[1]

        with tf.name_scope(self.name + "/1/"):
            self.weight_dense.build(input_shape)
    
    def call(self,inputs):
        features, wfeat = inputs
        weight = self.weight_dense(wfeat)/0.761594156 #normliase such that with chosen bias initializer weights are 1
        return features*weight
        
class RecalcDistances(tf.keras.layers.Layer):         
    def __init__(self, 
                 **kwargs):
        """
        
        Careful: This class expands in V x K x C
        
        Input:
        - coordinates
        - neighbour indices
        
        Output:
        - new distances
        
        """    
        super(RecalcDistances, self).__init__(**kwargs) 
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[1] #same as nidx
        
    def call(self,inputs):
        assert len(inputs)==2
        coords, nidx = inputs
        
        ncoords = SelectWithDefault(nidx, coords, 0.) # V x K x C
        dist = tf.reduce_sum( (ncoords - tf.expand_dims(coords,axis=1))**2, axis=2 )
        return dist

class SelectFromIndicesWithPad(tf.keras.layers.Layer):
    
    '''
    SelectWithDefault wrapper
    
    inputs: 
    - indices
    - to be selected
    
    options:
    - default value
    '''
    def __init__(self, default = 0., **kwargs):  
        self.default = default
        if 'dynamic' in kwargs:
            super(SelectFromIndicesWithPad, self).__init__(**kwargs)
        else:
            super(SelectFromIndicesWithPad, self).__init__(dynamic=False,**kwargs)
            
    def get_config(self):
        config = {'default': self.default}
        base_config = super(SelectFromIndicesWithPad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):# 
        return input_shapes[0][:-1]+input_shapes[1][-1:]
    
    def call(self, inputs):
        assert len(inputs)==2
        out = SelectWithDefault(inputs[0], inputs[1], self.default)
        return out
    
    
class SelectFromIndices(tf.keras.layers.Layer): 
    def __init__(self, **kwargs):    
        """
        
        This layer selects a set of vertices.
        
        Inputs are:
         - the selection indices
         - a list of tensors the selection should be applied to (extending the indices)
         
        This layer is useful in combination with e.g. LocalClustering, to apply the clustering selection
        to other tensors (e.g. the output of a GravNet layer, or a SoftPixelCNN layer)
        
         
        """  
        if 'dynamic' in kwargs:
            super(SelectFromIndices, self).__init__(**kwargs)
        else:
            super(SelectFromIndices, self).__init__(dynamic=False,**kwargs)
        
    def compute_output_shape(self, input_shapes):#these are tensors shapes
        #ts = tf.python.framework.tensor_shape.TensorShape
        outshapes = [(None, ) +tuple(s[1:]) for s in input_shapes][1:]
        return outshapes #all but first (indices)
    
    
    def compute_output_signature(self, input_signature):
        print('>>>>>SelectFromIndices input_signature',input_signature)
        input_shape = tf.nest.map_structure(check_type_return_shape, input_signature)
        output_shape = self.compute_output_shape(input_shape)
        input_dtypes=[i.dtype for i in input_signature]
        return [tf.TensorSpec(dtype=input_dtypes[i+1], shape=output_shape[i]) for i in range(0,len(output_shape))]
    
    def build(self, input_shapes): #pure python
        super(SelectFromIndices, self).build(input_shapes)
        outshapes = self.compute_output_shape(input_shapes)
        
        self.outshapes = [[-1,] + list(s[1:]) for s in outshapes] 
          
    @staticmethod  
    def raw_call(indices, inputs, outshapes=None):
        if outshapes is None:
            outshapes =  [[-1,] + list(s.shape[1:]) for s in inputs] 
        outs=[]
        for i in range(0,len(inputs)):
            g = tf.gather_nd( inputs[i], indices)
            g = tf.reshape(g, outshapes[i]) #[-1]+inputs[i].shape[1:])
            outs.append(g) 
        if len(outs)==1:
            return outs[0]
        return outs
        
    def call(self, inputs):
        indices = inputs[0]
        outshapes = self.outshapes # self.compute_output_shape([tf.shape(i) for i in inputs])
        return SelectFromIndices.raw_call(indices, inputs[1:], outshapes)
        
        
class MultiBackGather(tf.keras.layers.Layer):  
    def __init__(self, **kwargs):    
        """
        
        This layer gathers back vertices that were previously clustered using the output of LocalClustering.
        E.g. if vertices 0,1,2, and 3 ended up in the same cluster with 0 being the cluster centre, and 4,5,6 ended
        up in a cluster with 4 being the cluster centre, there will be two clusters: A and B.
        This layer will create a vector of the previous dimensionality containing:
        [A,A,A,A,B,B,B] so that the cluster properties of A and B are gathered back to the positions of their
        constituents.
        
        If multiple clusterings were performed, the layer can operate on a list of backgather indices.
        (internally the order of this list will be inverted) 
        
        Inputs are:
         - The data to gather back to larger dimensionality by repetition
         - A list of backgather indices (one index tensor for each repetition).
         
        """  
        if 'dynamic' in kwargs:
            super(MultiBackGather, self).__init__(**kwargs) 
        else:
            super(MultiBackGather, self).__init__(dynamic=False,**kwargs) 
        
    def compute_output_shape(self, input_shape):
        return input_shape #batch dim is None anyway
    
    
    @staticmethod 
    def raw_call(x, gathers):
        for k in range(len(gathers)):
            l = len(gathers) - k - 1
            #cast is needed because keras layer out dtypes are not really working
            x = SelectFromIndices.raw_call(tf.cast(gathers[l],tf.int32), 
                                           [x], [ [-1]+list(x.shape[1:]) ])[0]
        return x
        
    def call(self, inputs):
        x, gathers = inputs
        return MultiBackGather.raw_call(x,gathers)
        
class MultiBackScatter(tf.keras.layers.Layer):  
    def __init__(self, **kwargs):    
        """
        
        This layer scatters back vertices that were previously clustered using the output of LocalClustering.
        E.g. if vertices 0,1,2, and 3 ended up in the same cluster with 0 being the cluster centre, and 4,5,6 ended
        up in a cluster with 4 being the cluster centre, there will be two clusters: A and B.
        This layer will create a vector of the previous dimensionality containing:
        [A,0,0,0,B,0,0] so that the cluster properties of A and B are scattered back to the positions of their
        initial points.
        
        If multiple clusterings were performed, the layer can operate on a list of scatter indices.
        (internally the order of this list will be inverted) 
        
        Inputs are:
         - The data to scatter back to larger dimensionality by repetition
         - A list of list of type: [backscatterV, backscatter indices ] (one index tensor for each repetition).
         
        """  
        if 'dynamic' in kwargs:
            super(MultiBackScatter, self).__init__(**kwargs) 
        else:
            super(MultiBackScatter, self).__init__(dynamic=False,**kwargs) 
        
    def compute_output_shape(self, input_shape):
        return input_shape #batch dim is None anyway
    
    
    @staticmethod 
    def raw_call(x, scatters):
        xin=x
        for k in range(len(scatters)):
            l = len(scatters) - k - 1
            V, scidx = scatters[l]
            #print('scatters[l]',scatters[l])
            #cast is needed because keras layer out dtypes are not really working
            shape = tf.concat([tf.expand_dims(V,axis=0), tf.shape(x)[1:]],axis=0)
            x = tf.scatter_nd(scidx, x, shape)
            
        return tf.reshape(x,[-1,xin.shape[1]])
        
    def call(self, inputs):
        x, scatters = inputs
        if x.shape[0] is None:
            return tf.reshape(x,[-1 ,x.shape[1]])
        xnew = MultiBackScatter.raw_call(x,scatters)
        return xnew
        
class MultiBackScatterOrGather(tf.keras.layers.Layer):  
    def __init__(self, default: float=0., **kwargs):    
        """
        
        Either applies a scattering or gathering operation depending on the inputs
        (compatible with the outputs of NoiseFilter and NeighbourGroups).
        
        In general preferred
        
        Inputs:
        - data
        - scatter or gather indices
        
        """  
        self.default = default
        if 'dynamic' in kwargs:
            super(MultiBackScatterOrGather, self).__init__(**kwargs) 
        else:
            super(MultiBackScatterOrGather, self).__init__(dynamic=False,**kwargs) 
        
    def get_config(self):
        config = {'default': self.default}
        base_config = super(MultiBackScatterOrGather, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    @staticmethod 
    def raw_call(x, scatters, default=0.):
        xin=x
        for k in range(len(scatters)):
            l = len(scatters) - k - 1
            if isinstance(scatters[l], list):
                V, scidx = scatters[l]
                #print('scatters[l]',scatters[l])
                #cast is needed because keras layer out dtypes are not really working
                shape = tf.concat([tf.expand_dims(V,axis=0), tf.shape(x)[1:]],axis=0)
                if default:
                    x = tf.tensor_scatter_nd_update(tf.zeros(shape, x.dtype)+default, scidx, x)
                else:
                    x = tf.scatter_nd(scidx, x, shape)
            else:
                x = SelectFromIndices.raw_call(tf.cast(scatters[l],tf.int32), 
                                           [x], [ [-1]+list(x.shape[1:]) ])
                                     
        return tf.reshape(x,[-1,xin.shape[1]])
        
    def call(self, inputs):
        x, scatters = inputs
        if x.shape[0] is None:
            return tf.reshape(x,[-1 ,x.shape[1]])
        xnew = MultiBackScatterOrGather.raw_call(x,scatters,self.default)
        return xnew    
        
        
############# Local clustering section ends


class KNN(LayerWithMetrics):
    def __init__(self,K: int, radius=-1., 
                 use_approximate_knn=False,
                 min_bins=None,
                 tf_distance=False,
                 **kwargs):
        """
        
        Select self+K nearest neighbours, with possible radius constraint.
        
        Call will return 
         - self + K neighbour indices of K neighbours within max radius
         - distances to self+K neighbours
        
        Inputs: coordinates, row_splits
        
        :param K: number of nearest neighbours
        :param radius: maximum distance of nearest neighbours,
                       can also contain the keyword 'dynamic'
        :param use_approximate_knn: use approximate kNN method (SlicingKnn) instead of exact method (SelectKnn)
        """
        super(KNN, self).__init__(**kwargs) 
        self.K = K
        
        self.use_approximate_knn = use_approximate_knn
        self.min_bins = min_bins
        self.tf_distance = tf_distance
        
        if isinstance(radius,int):
            radius=float(radius)
        self.radius = radius
        assert (isinstance(radius,str) and radius=='dynamic') or isinstance(radius,float)
        assert not(radius=='dynamic' and not use_approximate_knn)
        self.dynamic_radius = None
        if radius == 'dynamic':
            radius=1.
            with tf.name_scope(self.name + "/1/"):
                self.dynamic_radius = tf.Variable(initial_value=radius, 
                                         trainable=False,dtype='float32')
        
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius,
                  'min_bins':self.min_bins,
                  'use_approximate_knn': self.use_approximate_knn,
                  'tf_distance': self.tf_distance}
        base_config = super(KNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return (None, self.K+1),(None, self.K+1)

    @staticmethod 
    def raw_call(coordinates, row_splits, K, radius, use_approximate_knn, min_bins, tfdist, myself):
        nbins=None
        if use_approximate_knn:
            bin_width = radius # default value for SlicingKnn kernel
            idx,dist,nbins = SlicingKnn(K+1, coordinates,  row_splits,
                                  features_to_bin_on = (0,1),
                                  bin_width=(bin_width,bin_width),
                                  return_n_bins=True,
                                  min_bins = min_bins)
        else:
            idx,dist = BinnedSelectKnn(K+1, coordinates,  row_splits,
                                       n_bins=min_bins,
                                 max_radius= radius, tf_compatible=False,
                                 name = myself.name)

        
        if tfdist:
            ncoords = SelectWithDefault(idx, coordinates,0.)
            distsq = tf.reduce_sum( (ncoords[:,0:1,:]-ncoords)**2, axis=-1)
            distsq = tf.where(idx<0, 0., distsq)
            
        idx = tf.reshape(idx, [-1,K+1])
        dist = tf.reshape(dist, [-1,K+1])
        
        
        return idx,dist,nbins

    def update_dynamic_radius(self, dist, training=None):
        if self.dynamic_radius is None or not self.trainable:
            return
        #update slowly, with safety margin
        update = tf.math.reduce_max(tf.sqrt(dist))*1.05 #can be inverted for performance TBI
        update = self.dynamic_radius + 0.1*(update-self.dynamic_radius)
        updated_radius = tf.keras.backend.in_train_phase(update,self.dynamic_radius,training=training)
        tf.keras.backend.update(self.dynamic_radius,updated_radius)
        
    def call(self, inputs, training=None):
        coordinates, row_splits = inputs
        import time
        
        idx,dist = None, None
        if self.dynamic_radius is None:
            idx,dist,nbins = KNN.raw_call(coordinates, row_splits, self.K, 
                                          self.radius, self.use_approximate_knn,
                                          self.min_bins, self.tf_distance,self)
        else:
            idx,dist,nbins = KNN.raw_call(coordinates, row_splits, self.K, 
                                          self.dynamic_radius, self.use_approximate_knn,
                                          self.min_bins, self.tf_distance,self)
            self.update_dynamic_radius(dist,training)
            
        if self.use_approximate_knn:
            self.add_prompt_metric(nbins,self.name+'_slicing_bins')
            
        return idx,dist
        

        
    
class AddIdentity2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddIdentity2D, self).__init__(**kwargs) 
        
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, inputs):
        diag = tf.expand_dims(tf.eye(inputs.shape[-1]), axis=0)
        return inputs + diag
        
class WarpedSpaceKNN(tf.keras.layers.Layer):
    def __init__(self,K: int, radius: float=-1., **kwargs):
        """

        Select K nearest neighbours, with possible radius constraint in warped space
        Warning: the time consumption increases with space_dim**2
        About factor 2 slower than standard kNN for 3 dimensions, then increasing

        Call will return
         - self + K neighbour indices of K neighbours within max radius
         - distances to self+K neighbours

        Inputs: coordinates, warp tensor, row_splits

        :param K: number of nearest neighbours
        :param radius: maximum distance of nearest neighbours
        """
        super(WarpedSpaceKNN, self).__init__(**kwargs)
        self.K = K
        self.radius = radius


    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius}
        base_config = super(WarpedSpaceKNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return (None, self.K+1),(None, self.K+1)

    @staticmethod
    def raw_call(coordinates, row_splits, warp, K, radius):
        idx,dist = SelectModKnn(K+1, coordinates,  warp, row_splits,
                             max_radius= radius, tf_compatible=False)

        idx = tf.reshape(idx, [-1,K+1])
        dist = tf.reshape(dist, [-1,K+1])
        return idx,dist

    def call(self, inputs):
        coordinates, warp, row_splits = inputs
        return WarpedSpaceKNN.raw_call(coordinates, row_splits, warp, self.K, self.radius)


class SortAndSelectNeighbours(tf.keras.layers.Layer):
    def __init__(self,K: int, radius: float=-1., sort=True, descending=False, **kwargs):
        """
        
        This layer will sort neighbour indices by distance and possibly select neighbours
        within a radius, or the closest ones up to K neighbours.
        
        If a sorting score is given the sorting will be based on that (will still return the same)
        
        Inputs: distances, neighbour indices, sorting_score (opt)
        
        Call will return 
         - neighbour distances sorted by distance 
         - neighbour indices sorted by distance 
        
        
        :param K: number of nearest neighbours, will do no selection if K<1
        :param radius: maximum distance of nearest neighbours (no effect if < 0)
        :param descending: use descending order
        
        """
        super(SortAndSelectNeighbours, self).__init__(**kwargs) 
        self.K = K
        self.radius = radius
        self.sort=sort
        self.descending = descending
        
        
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius,
                  'sort': self.sort,
                  'descending': self.descending}
        base_config = super(SortAndSelectNeighbours, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        if self.K > 0:
            return (None, self.K),(None, self.K)
        else:
            return input_shapes

    def compute_output_signature(self, input_signature):
        
        input_shapes = [x.shape for x in input_signature]
        input_dtypes = [x.dtype for x in input_signature]
        output_shapes = self.compute_output_shape(input_shapes)
        
        return [tf.TensorSpec(dtype=input_dtypes[i], shape=output_shapes[i]) for i in range(len(output_shapes))]
        
    @staticmethod 
    def raw_call(distances, nidx, K, radius, sort, incr_sorting_score, keep_self=True):
        
        K = K if K>0 else distances.shape[1]
        if not sort:
            return distances[:,:K],nidx[:,:K]
        
        if tf.shape(incr_sorting_score)[1] is not None and tf.shape(incr_sorting_score)[1]==1:
            incr_sorting_score = SelectWithDefault(nidx, incr_sorting_score, 0.)[:,0]
        
        tfssc = tf.where(nidx<0, 1e9, incr_sorting_score) #make sure the -1 end up at the end
        if keep_self:
            tfssc = tf.concat([tf.reduce_min(tfssc[:,1:],axis=1,keepdims=True)-1.,tfssc[:,1:]  ],axis=1) #make sure 'self' remains in first place

        sorting = tf.argsort(tfssc, axis=1)
        
        snidx = tf.gather(nidx,sorting,batch_dims=1) #_nd(nidx,sorting,batch_dims=1)
        sdist = tf.gather(distances,sorting,batch_dims=1)
        if K > 0:
            snidx = snidx[:,:K]
            sdist = sdist[:,:K]
            
        if radius > 0:
            snidx = tf.where(sdist > radius, -1, snidx)
            sdist = tf.where(sdist > radius, 0. , sdist)
            
        #fix the shapes
        sdist = tf.reshape(sdist, [-1, K])
        snidx = tf.reshape(snidx, [-1, K])
            
        return sdist, tf.cast(snidx, tf.int32) #just to avoid keras not knowing the dtype


        
    def call(self, inputs):
        distances, nidx, ssc = None, None, None
        if len(inputs)==2:
            distances, nidx = inputs
            ssc = distances
            
        elif len(inputs)==3:
            distances, nidx,ssc = inputs
            
        if self.descending:
            ssc = -1.* ssc
        
        return SortAndSelectNeighbours.raw_call(distances,nidx,self.K,self.radius,self.sort, ssc)
        #make TF compatible
        
        


class NoiseFilter(LayerWithMetrics):
    def __init__(self, 
                 threshold = 0.1, 
                 print_reduction=False,
                 return_backscatter=True,
                 **kwargs):
        
        '''
        This layer will leave at least one noise hit per row split intact
        
        threshold: note, noise will have low score values.
        
        The loss corrects for non-equal class balance
        
        Inputs:
         - noise score (linear activation), high means not noise
         - row splits
         
        Outputs:
         - selection indices
         - row splits
         - backgather/(bsnumber backscatter indices)
        '''

        
        if 'dynamic' in kwargs:
            super(NoiseFilter, self).__init__(**kwargs)
        else:
            super(NoiseFilter, self).__init__(dynamic=False,**kwargs)
            
        assert threshold>0 and threshold < 1
        self.threshold = threshold
        self.print_reduction = print_reduction
        self.return_backscatter = return_backscatter
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'print_reduction': self.print_reduction,
                  'return_backscatter': self.return_backscatter}
        
        base_config = super(NoiseFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shape):
        if self.return_backscatter:
            return (None,1), (None,1), [(None, 1), (None,1)]
        else:
            return (None,1), (None,1), (None, 1)
        
    
    def call(self, inputs):
        
        score, row_splits = inputs
        
        defbg = tf.expand_dims(tf.range(tf.shape(score)[0]),axis=1)
        if row_splits.shape[0] is None: #dummy execution
            bg = defbg
            if self.return_backscatter:
                bg = [tf.shape(score)[0], bg]
                
            return tf.range(tf.shape(score)[0],dtype='int32'),row_splits, bg
        
        sel, allbackgather, newrs = None,None,None
        if self.return_backscatter:
            allidxs = tf.expand_dims(tf.range(tf.shape(score)[0]),axis=1)
            sel=[]
            rs=[0]
            for i in range(len(row_splits)-1):
                thissel = allidxs[row_splits[i]:row_splits[i+1]][score[row_splits[i]:row_splits[i+1]]>self.threshold]
                rs.append(thissel.shape[0])
                sel.append(thissel)
            
            newrs = tf.cumsum(tf.concat(rs,axis=0),axis=0)
            sel = tf.expand_dims(tf.concat(sel,axis=0),axis=1)
            allbackgather = [tf.shape(score)[0], sel]
        else:
            sel, allbackgather, newrs = select_threshold_with_backgather(score, self.threshold, row_splits)
            
        if self.print_reduction:
            print(self.name,' reduction from ', int(row_splits[-1]), ' to ', int(newrs[-1]), ': ', float(row_splits[-1])/float(newrs[-1]))
        
        self.add_prompt_metric(tf.cast(newrs[-1],dtype='float32')/tf.cast(row_splits[-1],dtype='float32'),self.name+'_reduction')
        
        return sel, newrs, allbackgather
        
        

class EdgeCreator(tf.keras.layers.Layer):
    def __init__(self, addself=False,**kwargs):
        '''
        Be careful! this blows up the space to V x K-1 x F !
        
        Inputs:
        - neighbour indices (assumes V x 0 index is probe vertex index)
        - features
        
        returns:
        - edges (V x K -1 x F) (difference to K=0)
        '''
        if 'dynamic' in kwargs:
            super(EdgeCreator, self).__init__(**kwargs)
        else:
            super(EdgeCreator, self).__init__(dynamic=False,**kwargs)
            
        self.addself=addself
        
    def get_config(self):
        config = {'addself': self.addself}
        
        base_config = super(EdgeCreator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))     
    
    def compute_output_shape(self, input_shapes): 
        if self.addself:
            return (None, input_shapes[0][-1], input_shapes[1][-1]) 
        return (None, input_shapes[0][-1]-1, input_shapes[1][-1]) # K x F
    
    def call(self, inputs):
        selffeat = tf.expand_dims(inputs[1],axis=1)
        if self.addself:
            edges = selffeat - SelectWithDefault(inputs[0][:,:], inputs[1], -1.)
        else:
            edges = selffeat - SelectWithDefault(inputs[0][:,1:], inputs[1], -1.)
        return edges

class EdgeContractAndMix(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        '''
        reduce_mean and max over edges and concat to original edge features.
        
        Inputs:
        - tensor: V x E x F
        
        Outputs:
        - tensor: V x E x 2F (mean and max expanded to edges)
        '''
        super(EdgeContractAndMix, self).__init__(**kwargs)
    
    def call(self, inputs):
        x_e = inputs
        x_sum = tf.reduce_sum(x_e, axis=1, keepdims=True)
        x_max = tf.reduce_max(x_e, axis=1, keepdims=True)
        x_out = tf.concat([x_sum,x_max], axis=-1)
        x_out = tf.tile(x_out, [1, x_e.shape[1], 1])
        return x_out
    

class EdgeSelector(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.5, 
                 **kwargs):
        '''
        Inputs:  edge score (V x K-1), neighbour indices (V x K )
        Outputs: selected neighbour indices ('-1 masked')
        
        This layer violates the standard '-1'-padded neighbour sorting.
        Use a sorting layer afterwards!
        
        '''
        assert threshold<1 and threshold>=0
        self.threshold = threshold        
        if 'dynamic' in kwargs:
            super(EdgeSelector, self).__init__(**kwargs)
        else:
            super(EdgeSelector, self).__init__(dynamic=False,**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[1]
    
    def get_config(self):
        config = {'threshold': self.threshold}
        base_config = super(EdgeSelector, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
       
    def call(self, inputs):   
        assert len(inputs) == 2
        score, nidx = inputs
        assert len(score.shape) == 3 and len(nidx.shape) == 2
        
        indices = tf.where(score[:,:,0] < self.threshold, -1, nidx[:,1:])
        indices = tf.sort(indices, axis=1, direction = "DESCENDING")#put the -1 at the end
        sindxs = tf.concat([nidx[:,0:1], indices],axis=1)
        sindxs = tf.reshape(sindxs, [-1,nidx.shape[1]]) #to get the output shapes defined
        #old
        score = tf.concat([tf.ones_like(score[:,0:1,:]),score],axis=1)#add self score always 1
        nsidxs = tf.where(score[:,:,0] < self.threshold, -1, nidx)
        
        return sindxs
        
    

class DampenGradient(tf.keras.layers.Layer):
    def __init__(self, strength=0.5, **kwargs):
        '''
        Dampens gradient during back propagation.
        strength = 0. means normal gradient, strength = 1. means no gradient
        '''
        assert strength>=0 and strength<=1
        super(DampenGradient, self).__init__(**kwargs)
        self.strength = strength
        
    def get_config(self):
        config = {'strength': self.strength}
        base_config = super(DampenGradient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return self.strength*tf.stop_gradient(inputs)+ (1.-self.strength)*inputs
            
class GroupScoreFromEdgeScores(tf.keras.layers.Layer):
    def __init__(self, den_offset=0., **kwargs):
        '''
        Input: 
        - edge scores (V x K-1 x 1)
        - neighbour indices (V x K)
        
        Output:
        - group score (V x 1)
        
        '''
        self.den_offset = den_offset
        if 'dynamic' in kwargs:
            super(GroupScoreFromEdgeScores, self).__init__(**kwargs)
        else:
            super(GroupScoreFromEdgeScores, self).__init__(dynamic=False,**kwargs)
            
    def get_config(self):
        base_config = super(GroupScoreFromEdgeScores, self).get_config()
        return dict(list(base_config.items()) + list({'den_offset': self.den_offset}.items()))
    
    def compute_output_shape(self, input_shapes): 
        return (None,1)
            
        #no config
    def call(self, inputs): 
        if len(inputs) == 3:
            score, nidx, energy = inputs
            energy = SelectWithDefault(nidx,energy, 0.)
            energy = tf.reduce_sum(energy, axis=1)
            eweight = (1.+tf.sqrt(energy))[:,0]# V 
        else:
            score, nidx = inputs
            eweight = 1.
        #take mean
        active = tf.cast(nidx>-1, 'float32')
        
        n_neigh = tf.math.count_nonzero(active, axis=1)# V 
        n_neigh = tf.cast(n_neigh,dtype='float32') - 1.
        groupscore = tf.reduce_sum(active[:,1:]*score[:,:,0], axis=1)# V 
        
        #give slight priority to larger groups
        n_neigh += self.den_offset
        n_neigh = tf.where(n_neigh<=0, 1e-2, n_neigh)
        
        groupscore = tf.math.divide_no_nan(eweight * groupscore,n_neigh)
        return tf.expand_dims(groupscore,axis=1)


### soft pixel section
class NeighbourGroups(LayerWithMetrics):
    def __init__(self, 
                 threshold = None, 
                 initial_threshold = None, #compatibility
                 purity_min_target = None,
                 efficiency_min_target=None,
                 thresh_viscosity = 1e-2,
                 print_reduction=False, 
                 return_backscatter=True,
                **kwargs):
        '''
        
        Param:
            - threshold: hierarchy discriminator cut off (if None it is automatically adjusted)
            - purity_min_target: minimum purity target in case threshold is automatically adjusted
            - efficiency_min_target: minimum efficiency target in case threshold is automatically adjusted
        
        Inputs:
            - neighbourhood classifier (sigmoid activation must be applied)
            - neighbour indices  (V x K) <- must contain "self"
            - row splits
            
        Outputs: 
             - neighbour indices for directed graph accumulation (V x K), '-1' padded
             - selection indices (V' x 1)! (Use with Accumulate/Select layers)
             - backgather/backscatter indices (V x K)
             - new row splits
        '''
        
        
        
        super(NeighbourGroups, self).__init__(**kwargs) 
        self.print_reduction = print_reduction
        self.return_backscatter = return_backscatter
        self.purity_min_target = purity_min_target
        self.efficiency_min_target = efficiency_min_target
        self.thresh_viscosity = thresh_viscosity
        self.initial_threshold = threshold
        
        if threshold is None:
            #assert purity_min_target is not None and efficiency_min_target is not None
            self.initial_threshold = 0.5
        else:
            assert purity_min_target is None and efficiency_min_target is None #if threshold is not None
        #make this a variable
        with tf.name_scope(self.name + "/1/"):
            self.threshold = tf.Variable(initial_value=self.initial_threshold, trainable=False,dtype='float32')
        
            
    def get_config(self):
        config = {'purity_min_target': self.purity_min_target,
                  'efficiency_min_target': self.efficiency_min_target,
                  'thresh_viscosity': self.thresh_viscosity,
                  'print_reduction': self.print_reduction,
                  'threshold': self.initial_threshold,
                  'return_backscatter': self.return_backscatter}
        
        base_config = super(NeighbourGroups, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def _update_threshold(self, purity, efficiency, training=None):
        if self.purity_min_target is None or training is None or training is False:
            return
        
        #calc update
        puradd = tf.nn.relu(self.purity_min_target-purity)
        effadd = -tf.nn.relu(self.efficiency_min_target-efficiency)
        add = puradd+effadd
        #add some gooeyness to it
        update = (1. + self.thresh_viscosity*add)*self.threshold
        update = tf.clip_by_value(update, 1e-3, 1.-1e-3)
        
        updated_thresh = tf.keras.backend.in_train_phase(update,self.threshold,training=training)
        tf.keras.backend.update(self.threshold,updated_thresh)
    


    def call(self, inputs,  training=None):
        assert len(inputs)==3
        score, nidxs, row_splits = inputs
        score = tf.clip_by_value(score, 0., 1.)
    
        ######### model build phase        
        if row_splits.shape[0] is None:
            sel = tf.expand_dims(tf.range(tf.shape(nidxs)[0]),axis=1)
            dirnidx = nidxs
            ggather = tf.zeros_like(score, dtype='int32')
            rs = row_splits
        
        else:
            hierarchy_idxs=[]
            for i in range(tf.shape(row_splits)[0] - 1):
                a = tf.argsort(score[row_splits[i]:row_splits[i+1]],axis=0, direction='DESCENDING')
                hierarchy_idxs.append(a+row_splits[i])
            hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
            
            #rewrite Localgroup
            rs,dirnidx,sel,ggather = LocalGroup(nidxs, hierarchy_idxs, score, row_splits, 
                                        score_threshold=self.threshold.numpy())#force eager
            preN = dirnidx.shape[0]
            postN = sel.shape[0]
            if self.print_reduction:
                print(self.name,'reduced from', preN ,'to',postN)
            
            self.add_prompt_metric(postN/(preN+1e-3),self.name+'_reduction')
        
        back = ggather
        if self.return_backscatter:
            back = [tf.shape(nidxs)[0], sel]
            
        dirnidx = tf.reshape(dirnidx, tf.shape(nidxs))#to make shape clear
            
        return dirnidx, sel, back, rs

    def compute_output_shape(self, input_shapes):
        score, nidxs, row_splits = input_shapes
        if self.return_backscatter:
            return nidxs, score, [(None, 1), score], row_splits #first dim omitted
        else:
            return nidxs, score, score, row_splits
        

class AccumulateNeighbours(tf.keras.layers.Layer):
    def __init__(self, mode='meanmax' , **kwargs):
        '''
        Inputs: feat, nidx, weights (opt. and strict > 0)
        
        Outputs: accumulated features
        
        param:
        - mode: meanmax, gnlike, sum, mean, min, max, minmeanmax
        '''
        assert mode == 'meanmax' or mode == 'mean' or mode=='gnlike' or mode =='sum' or\
         mode == 'minmeanmax' or mode == 'min' or mode == 'max'
        
        super(AccumulateNeighbours, self).__init__(**kwargs) 
        self.mode = mode
        
    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(AccumulateNeighbours, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shapes):
        super(AccumulateNeighbours, self).build(input_shapes)
        
    def compute_output_shape(self, input_shapes):
        fshape = input_shapes[0]
        if self.mode == 'mean' or self.mode == 'sum' or self.mode == 'min' or  self.mode == 'max':
            return fshape
        elif self.mode == 'minmeanmax':
            return (None, 3*fshape[1])
        else:
            return (None, 2*fshape[1])

    def get_min(self,ndix,feat):
        out,_ = AccumulateLinKnn(tf.zeros_like(ndix,dtype='float32')+1., 
                          -feat, ndix,mean_and_max=True)
        out=out[:,feat.shape[1]:]
        return -tf.reshape(out,[-1, feat.shape[1]])  
          
    def call(self, inputs, training=None):
        assert len(inputs)==2 or len(inputs)==3
        feat,ndix,w = None,None,None
        if len(inputs)==2:
            feat,ndix = inputs
            w = tf.cast(tf.zeros_like(ndix),dtype='float32')+1.
        else:
            feat,ndix,w = inputs
            w = SelectWithDefault(ndix, w, 0.)[:,:,0]
            
        K = tf.cast(ndix.shape[1],dtype='float32')
        #K = tf.expand_dims(tf.expand_dims(K,axis=0),axis=0)
        if self.mode == 'mean' or self.mode == 'meanmax':
            out,_ = AccumulateLinKnnSumw(w, 
                          feat, ndix,mean_and_max=self.mode == 'meanmax')
            return out
        if self.mode=='gnlike':
            out,_ = AccumulateLinKnn(w, 
                          feat, ndix)
            return tf.reshape(out,[-1, 2*feat.shape[1]])
        if self.mode=='sum':
            out,_ = AccumulateLinKnn(w,feat, ndix,mean_and_max=False)
            out*=K
            return tf.reshape(out,[-1, feat.shape[1]])              
        if self.mode == 'max':
            out,_ = AccumulateLinKnn(w, 
                          feat, ndix,mean_and_max=True)
            out=out[:,feat.shape[1]:]
            return tf.reshape(out,[-1, feat.shape[1]])
        if self.mode == 'min':
            return self.get_min(ndix,feat)
        if self.mode == 'minmeanmax':
            meanmax,_ = AccumulateLinKnnSumw(w, 
                          feat, ndix,mean_and_max=True)
            meanmax = tf.reshape(meanmax,[-1, 2*feat.shape[1]])
            minvals = self.get_min(ndix,feat)
            return tf.concat([minvals,meanmax],axis=1)
        return feat #just backup, should not happen



class SoftPixelCNN(tf.keras.layers.Layer):
    def __init__(self, length_scale_momentum=0.01, mode: str='onlyaxes', subdivisions: int=3 , **kwargs):
        """
        Inputs: 
        - coordinates
        - features
        - distances squared
        - neighour_indices
        
        This layer is strongly inspired by 1611.08097 (even though this is only presented for 2D manifolds).
        It implements "soft pixels" for each direction given by the input coordinates plus one central "pixel".
        Each "pixel" is represented by a Gaussian weighting of the inputs.
        For D coordinate dimensions, 2*D + 1 "pixels" are formed (e.g. one for position x direction one for neg x etc).
        The pixels beyond the length scale are removed.
        
        Call will return 
         - soft pixel CNN outputs:  (V x npixels * features)
        
        :param length_scale: scale of the distances that are expected. This scale is dynamically adjusted during training,
                             so this is just a first guess (1 is usually good)
                             
        :param mode: mode to build the soft pixels
                     - full: a full grid is spanned over all dimensions
                     - oneless: only grid points with at least one axis coordinate being zero are used
                       (removes one dimension from output, but leaves the 0 point)
                     - onlyaxes: only the points along the axes are used to define the pixel centres, including the 0 point
                     
        :param subdivisions: number of subdivisions for each dimension (odd number preserves centre pixel)
        
        """
        super(SoftPixelCNN, self).__init__(**kwargs) 
        assert length_scale_momentum > 0
        assert subdivisions > 1
        assert mode=='onlyaxes' or mode=='full' or mode=='oneless'
        
        self.length_scale_momentum=length_scale_momentum
        with tf.name_scope(self.name + "/1/"):
            self.length_scale = tf.Variable(initial_value=1., trainable=False,dtype='float32')
            
        self.mode=mode
        self.subdivisions=subdivisions
        self.ndim = None 
        self.offsets = None
        
        
    def get_config(self):
        config = {'length_scale_momentum' : self.length_scale_momentum,
                  'mode': self.mode,
                  'subdivisions': self.subdivisions
                  }
        base_config = super(SoftPixelCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def create_offsets(D, subdivision):
        length_scale=1
        temp = [np.linspace(-length_scale , length_scale, subdivision) for _ in range(D)]
        res_to_unpack = np.meshgrid(*temp, indexing='xy')
        
        a = [np.expand_dims(b,axis=D) for b in res_to_unpack]
        a = np.concatenate(a,axis=-1)
        a = np.reshape(a,[-1,D])
        a = np.array(a,dtype='float32')
        a = a[ np.sqrt(np.sum(a**2, axis=-1)) <= length_scale  ]
        
        b = a[np.prod(a,axis=-1)==0]
        #all but one 0
        onlyaxis = b[np.count_nonzero(b,axis=-1)==0]
        onlyaxis = np.concatenate([onlyaxis, b[np.count_nonzero(b,axis=-1)==1]],axis=0)
        return a,b,onlyaxis
    
    def build(self, input_shapes):
        self.ndim = input_shapes[0][1]
        self.nfeat = input_shapes[1][1]
        self.offsets=[]
        a,b,c = SoftPixelCNN.create_offsets(self.ndim,self.subdivisions)
        if self.mode == 'full':
            self.offsets = a
        elif self.mode == 'oneless':
            self.offsets = b
        elif self.mode == 'onlyaxes':
            self.offsets = c
            
        super(SoftPixelCNN, self).build(input_shapes)
        
    def compute_output_shape(self, input_shapes):
        noutvert = input_shapes[2][0]
        #ncoords = input_shapes[0][1]
        nfeat = input_shapes[1][1]
        return (noutvert, nfeat*self.offsets)

    def call(self, inputs, training=None):
        
        coordinates, features, distsq, neighbour_indices = inputs
        #create coordinate offsets
        out=[]
        
        #adjust to 2 sigma continuously
        distsq = tf.stop_gradient(distsq)
        if self.trainable:
            tf.keras.backend.update(self.length_scale, 
                                    tf.keras.backend.in_train_phase(
                self.length_scale*(1.-self.length_scale_momentum) + self.length_scale_momentum*2.*tf.reduce_mean(tf.sqrt(distsq+1e-6)),
                                self.length_scale,
                                training=training)
                                    )
        
        for o in self.offsets:
            distancesq = LocalDistance(coordinates+o, neighbour_indices)
            f,_ = AccumulateKnn(self.length_scale*distancesq,  features, neighbour_indices) # V x F
            f = f[:,:self.nfeat]
            f = tf.reshape(f, [-1, self.nfeat])
            out.append(f) #only mean
        out = tf.concat(out,axis=-1)
        return out



######## generic neighbours

class RaggedGravNet(LayerWithMetrics):
    def __init__(self,
                 n_neighbours: int,
                 n_dimensions: int,
                 n_filters : int,
                 n_propagate : int,
                 return_self=True,
                 sumwnorm=False,
                 feature_activation='relu',
                 use_approximate_knn=False,
                 coord_initialiser_noise=1e-2,
                 use_dynamic_knn=True,
                 debug = False,
                 n_knn_bins=None,
                 **kwargs):
        """
        Call will return output features, coordinates, neighbor indices and squared distances from neighbors
        Inputs:
        - features
        - row splits

        :param n_neighbours: neighbors to do gravnet pass over
        :param n_dimensions: number of dimensions in spatial transformations
        :param n_filters:  number of dimensions in output feature transformation, could be list if multiple output
        features transformations (minimum 1)

        :param n_propagate: how much to propagate in feature tranformation, could be a list in case of multiple
        :param return_self: for the neighbour indices and distances, switch whether to return the 'self' index and distance (0)
        :param sumwnorm: normalise distance weights such that their sum is 1. (default False)
        :param feature_activation: activation to be applied to feature creation (F_LR) (default relu)
        :param use_approximate_knn: use approximate kNN method (SlicingKnn) instead of exact method (SelectKnn)
        :param use_dynamic_knn: uses dynamic adjustment of kNN binning derived from previous batches (only in effect together with use_approximate_knn)
        :param debug: switches on debug output, is not persistent when saving
        :param n_knn_bins: number of bins for included kNN (default: None=dynamic)
        :param kwargs:
        """
        super(RaggedGravNet, self).__init__(**kwargs)

        #n_neighbours += 1  # includes the 'self' vertex
        assert n_neighbours > 1
        assert not use_approximate_knn #not needed anymore. Exact one is faster by now

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.return_self = return_self
        self.sumwnorm = sumwnorm
        self.feature_activation = feature_activation
        self.use_approximate_knn = use_approximate_knn
        self.use_dynamic_knn = use_dynamic_knn
        self.debug = debug
        self.n_knn_bins = n_knn_bins
        
        self.n_propagate = n_propagate
        self.n_prop_total = 2 * self.n_propagate

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform = tf.keras.layers.Dense(n_propagate, activation=feature_activation)

        with tf.name_scope(self.name + "/2/"):
            self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions,
                                                                 #very slow turn on
                                                                 kernel_initializer=EyeInitializer(mean=0, stddev=coord_initialiser_noise),
                                                                 use_bias=False)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(self.n_filters, activation='relu')#changed to relu
            
        with tf.name_scope(self.name + "/4/"):
            self.dynamic_radius = tf.Variable(initial_value=1.,trainable=False,dtype='float32')

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform.build(input_shape)

        with tf.name_scope(self.name + "/2/"):
            self.input_spatial_transform.build(input_shape)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform.build((input_shape[0], self.n_prop_total + input_shape[1]))

        super(RaggedGravNet, self).build(input_shape)

    def update_dynamic_radius(self, dist, training):
        if not self.use_dynamic_knn or not self.trainable:
            return
        #update slowly, with safety margin
        lindist = tf.sqrt(dist)
        update = tf.reduce_max(lindist)*1.05 #can be inverted for performance TBI
        mean_dist = tf.reduce_mean(lindist)
        low_update = tf.where(update>2.,2.,update)#receptive field ends at 1.
        update = tf.where(low_update>2.*mean_dist,low_update,2.*mean_dist)#safety setting to not loose all neighbours
        update += 1e-3
        update = self.dynamic_radius + 0.05*(update-self.dynamic_radius)
        updated_radius = tf.keras.backend.in_train_phase(update,self.dynamic_radius,training=training)
        #print('updated_radius',updated_radius)
        tf.keras.backend.update(self.dynamic_radius,updated_radius)
        
    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x

        features = self.input_feature_transform(features)
        prev_feat = features
        features = self.collect_neighbours(features, neighbour_indices, distancesq)
        features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
        features -= tf.tile(prev_feat, [1, 2])
        allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return self.output_feature_transform(features)

    def priv_call(self, inputs, training=None):
        x = inputs[0]
        row_splits = inputs[1]
        tf.assert_equal(x.shape.ndims, 2)
        tf.assert_equal(row_splits.shape.ndims, 1)
        if row_splits.shape[0] is not None:
            tf.assert_equal(row_splits[-1], x.shape[0])
        
        
        coordinates = self.input_spatial_transform(x)
        neighbour_indices, distancesq, sidx, sdist = self.compute_neighbours_and_distancesq(coordinates, row_splits, training)
        neighbour_indices = tf.reshape(neighbour_indices, [-1, self.n_neighbours]) #for proper output shape for keras
        distancesq = tf.reshape(distancesq, [-1, self.n_neighbours])

        outfeats = self.create_output_features(x, neighbour_indices, distancesq)
        if self.return_self:
            neighbour_indices, distancesq = sidx, sdist
        return outfeats, coordinates, neighbour_indices, distancesq

    def call(self, inputs, training):
        return self.priv_call(inputs, training)

    def compute_output_shape(self, input_shapes):
        if self.return_self:
            return (input_shapes[0][0], 2*self.n_filters),\
               (input_shapes[0][0], self.n_dimensions),\
               (input_shapes[0][0], self.n_neighbours+1),\
               (input_shapes[0][0], self.n_neighbours+1)
        else:
            return (input_shapes[0][0], 2*self.n_filters),\
               (input_shapes[0][0], self.n_dimensions),\
               (input_shapes[0][0], self.n_neighbours),\
               (input_shapes[0][0], self.n_neighbours)
              
    

    def compute_neighbours_and_distancesq(self, coordinates, row_splits, training):
        
        idx,dist = BinnedSelectKnn(self.n_neighbours+1, coordinates,  row_splits,
                                 max_radius= -1.0, tf_compatible=False,
                                 n_bins=self.n_knn_bins, name=self.name
                                 
                                 )
        idx = tf.reshape(idx, [-1, self.n_neighbours+1])
        dist = tf.reshape(dist, [-1, self.n_neighbours+1])
        
        dist = tf.where(idx<0,0.,dist)
        
        self.update_dynamic_radius(dist,training)
        
        if self.return_self:
            return idx[:, 1:], dist[:, 1:], idx, dist
        return idx[:, 1:], dist[:, 1:], None, None


    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f = None
        if self.sumwnorm:
            f,_ = AccumulateKnnSumw(10.*distancesq,  features, neighbour_indices)
        else:
            f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices)
        return f

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours,
                  'n_dimensions': self.n_dimensions,
                  'n_filters': self.n_filters,
                  'n_propagate': self.n_propagate,
                  'return_self': self.return_self,
                  'sumwnorm': self.sumwnorm,
                  'feature_activation': self.feature_activation,
                  'use_approximate_knn':self.use_approximate_knn,
                  'n_knn_bins': self.n_knn_bins}
        base_config = super(RaggedGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SelfAttention(tf.keras.layers.Layer):
    
    def __init__(self,**kwargs):
        
        super(SelfAttention, self).__init__(**kwargs)
        with tf.name_scope(self.name + "/1/"):
            self.att_dense = tf.keras.layers.Dense(1, activation = 'softmax')
            
    def build(self, input_shapes): 
        with tf.name_scope(self.name + "/1/"):
            self.att_dense.build(input_shapes)
        super(SelfAttention, self).build(input_shapes)
        
    def compute_output_shape(self, input_shapes): 
        return input_shapes
    
    def call(self, input):
        att = 2.* self.att_dense(input)
        return input * att
    

class MultiAttentionGravNetAdd(LayerWithMetrics):
    def __init__(self,
                 n_attention_kernels :int,
                 **kwargs):
        '''
        To be used as an afterburned to GravNet or if neighbours have been 
        built some other way and the distances normalised around 1.
        
        Creates (per vertex) learnable attention kernels that can 'look' somewhere else
        within the neighbour groups than just the centre
        
        Inputs:
        - features used to derive the coordinate offsets (can be larger)
        - features that are accumulated (should be smaller as it grows with number of kernels)
        - coordinates
        - neighbour indices
        
        Parameters:
        
        :param n_attention_kernels: number of learnable attention kernels
        
        '''
        assert n_attention_kernels>0
        
        super(MultiAttentionGravNetAdd, self).__init__(**kwargs)
        self.n_attention_kernels = n_attention_kernels
        self.kernel_coord_dense = []
    
    
    def get_config(self):
        config = {'n_attention_kernels': self.n_attention_kernels}
        base_config = super(MultiAttentionGravNetAdd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes): # data, idxs
        featshape = input_shapes[1]
        featshape = (featshape[0], featshape[-1]*(self.n_attention_kernels))
        return featshape
        
    def build(self, input_shapes): 
        assert len(input_shapes)==4
        featshape = input_shapes[0]
        ncoords = input_shapes[2][-1]
        
        for i in range(self.n_attention_kernels):
            with tf.name_scope(self.name + "/"+str(i)+"/"):
                self.kernel_coord_dense.append( 
                     tf.keras.layers.Dense(ncoords, 
                                           kernel_initializer='zeros',
                                           bias_initializer='glorot_uniform',
                                           activation='tanh')#restrict to -1 1
                     )
                self.kernel_coord_dense[-1].build(featshape)
    
    def call(self, inputs):
        assert len(inputs)==4
        feat, passfeat, coord, nidx = inputs
        #coord = tf.stop_gradient(coord) #avoid coordinate gradient
        outfeat = []
        for di in range(len(self.kernel_coord_dense)):
            refcadd = self.kernel_coord_dense[di](feat)
            
            for i in range(coord.shape[-1]):
                meancoord = tf.reduce_mean(refcadd[:,i])
                self.add_prompt_metric(meancoord, self.name+'_coord_add_mean_'+str(di)+'_'+str(i))
                self.add_prompt_metric(tf.math.reduce_std(refcadd[:,i]-meancoord), self.name+'_coord_add_var_'+str(di)+'_'+str(i))
            
            refcoord = refcadd + coord
            refcoord = tf.expand_dims(refcoord,axis=1)#V x 1 x C
            othercoord = SelectWithDefault(nidx, coord, 0.)#V x K x C
            dist = tf.reduce_sum((othercoord - refcoord)**2,axis=2)
            acc,_ = AccumulateKnn(10.*dist, passfeat, nidx, mean_and_max=False)
            acc = tf.reshape(acc, tf.shape(passfeat))
            outfeat.append(acc)
        return tf.concat(outfeat,axis=1)    
        

class DynamicDistanceMessagePassing(tf.keras.layers.Layer):
    '''
    allows distances after each passing operation to be dynamically adjusted.
    this similar to FusedRaggedGravNetAggAtt, but incorporates the scaling in the message passing loop
    '''

    def __init__(self, n_feature_transformation,
                 **kwargs):
        super(DynamicDistanceMessagePassing, self).__init__(**kwargs)

        self.dist_mod_dense = []
        self.n_feature_transformation = n_feature_transformation
        self.feature_tranformation_dense = []
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.dist_mod_dense.append(tf.keras.layers.Dense(1, activation='sigmoid'))  # restrict variations a bit
            with tf.name_scope(self.name + "/6/" + str(i)):
                self.feature_tranformation_dense.append(tf.keras.layers.Dense(self.n_feature_transformation[i], activation='relu'))

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.dist_mod_dense[0].build((input_shape[0], input_shape[1]))
        for i in range(1, len(self.dist_mod_dense)):
            with tf.name_scope(self.name+"/5/"+str(i)):
                self.dist_mod_dense[i].build((input_shape[0],input_shape[1]+self.n_feature_transformation[i-1]*2))

        with tf.name_scope(self.name + "/6/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)
        for i in range(1, len(self.dist_mod_dense)):
            with tf.name_scope(self.name + "/6/" + str(i)):
                self.feature_tranformation_dense[i].build((input_shape[0], self.n_feature_transformation[i-1] * 2))


        super(DynamicDistanceMessagePassing, self).build(input_shapes)


    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x

        for i in range(len(self.n_feature_transformation)):
            if i == 0:
                scale = 10. * self.dist_mod_dense[0](x)
            else:
                scale = 10. * self.dist_mod_dense[i](tf.concat([x, features], axis=-1))
            distancesq *= scale
            t = self.feature_tranformation_dense[i]
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices, distancesq)

            features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
            features -= tf.tile(prev_feat, [1, 2])

            allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return features

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices)
        return f

    def call(self, inputs):
        x, neighbor_indices, distancesq = inputs
        return self.create_output_features(x, neighbor_indices, distancesq)


    def get_config(self):
        config = {
                  'n_feature_transformation': self.n_feature_transformation}
        base_config = super(DynamicDistanceMessagePassing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CollectNeighbourAverageAndMax(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        '''
        Simply accumulates all neighbour index information (including self if in the neighbour indices)
        Output will be divded by K, but not explicitly averaged if the number of neighbours is <K for
        particular vertices
        
        Inputs:  data, idxs
        '''
        super(CollectNeighbourAverageAndMax, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes): # data, idxs
        return (input_shapes[0][0],2*input_shapes[0][-1])
    
    def call(self, inputs):
        x, idxs = inputs
        f,_ = AccumulateKnn(tf.cast(idxs*0, tf.float32),  x, idxs)
        return tf.reshape(f, [-1,2*x.shape[-1]])
    

class MessagePassing(tf.keras.layers.Layer):
    '''
    Inputs: x, neighbor_indices
    
    
    '''

    def __init__(self, n_feature_transformation,
                 **kwargs):
        super(MessagePassing, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.feature_tranformation_dense = []
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense.append(tf.keras.layers.Dense(self.n_feature_transformation[i], activation='relu'))  # restrict variations a bit

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense[i].build((input_shape[0], self.n_feature_transformation[i-1] * 2))

        super(MessagePassing, self).build(input_shapes)

    def create_output_features(self, x, neighbour_indices):
        allfeat = []
        features = x


        for i in range(len(self.n_feature_transformation)):
            t = self.feature_tranformation_dense[i]
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices)
            features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
            features -= tf.tile(prev_feat, [1, 2])
            allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return features

    def collect_neighbours(self, features, neighbour_indices):
        neighbour_indices = tf.stop_gradient(neighbour_indices)
        f,_ = AccumulateLinKnn(tf.cast(neighbour_indices*0, tf.float32)+1.,  features, neighbour_indices)
        return f

    def call(self, inputs):
        x, neighbor_indices = inputs
        return self.create_output_features(x, neighbor_indices)

    def get_config(self):
        config = {'n_feature_transformation': self.n_feature_transformation,
                  }
        base_config = super(MessagePassing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistanceWeightedMessagePassing(tf.keras.layers.Layer):
    '''
    Inputs: x, neighbor_indices, distancesq

    '''
    def __init__(self, 
                 n_feature_transformation, #=[32, 32, 32, 32, 4, 4],
                 sumwnorm=False,
                 activation='relu',
                 **kwargs):
        super(DistanceWeightedMessagePassing, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.sumwnorm = sumwnorm
        self.feature_tranformation_dense = []
        self.activation = activation
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense.append(tf.keras.layers.Dense(self.n_feature_transformation[i],
                                                                              activation=activation))  # restrict variations a bit

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense[i].build((input_shape[0], self.n_feature_transformation[i - 1] * 2))

        super(DistanceWeightedMessagePassing, self).build(input_shapes)

    def compute_output_shape(self, inputs_shapes):
        fshape = inputs_shapes[0][-1]
        return (None, fshape + 2*sum(self.n_feature_transformation))
        
    def get_config(self):
        config = {'n_feature_transformation': self.n_feature_transformation,
                  'activation': self.activation,
                  'sumwnorm':self.sumwnorm
        }
        base_config = super(DistanceWeightedMessagePassing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x

        for i in range(len(self.n_feature_transformation)):
            t = self.feature_tranformation_dense[i]
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices, distancesq)
            features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
            features -= tf.tile(prev_feat, [1, 2])
            allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return features

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f=None
        if self.sumwnorm:
            f,_ = AccumulateKnnSumw(10.*distancesq,  features, neighbour_indices, mean_and_max=True)
        else:
            f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices)
        return f

    def call(self, inputs):
        x, neighbor_indices, distancesq = inputs
        return self.create_output_features(x, neighbor_indices, distancesq)


class DistanceWeightedAttentionMP(DistanceWeightedMessagePassing):
    def __init__(self, 
                 *args, #dummy
                 K : int, #needs to know KNN,
                 pos_embedding : int,
                 activation='elu',
                 **kwargs):
        
        #force it
        super(DistanceWeightedAttentionMP, self).__init__(
            *args,activation=activation,**kwargs)
        
        assert K>0
        
        self.K = K
        self.pos_embedding = pos_embedding
        self.attention_dense = []
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/6/" + str(i)):
                self.attention_dense.append(tf.keras.layers.Dense(K,activation='sigmoid')) 
                
        with tf.name_scope(self.name + "/7/"):
            self.pos_emb = tf.keras.layers.Dense(pos_embedding, activation=activation)

    def get_config(self):
        config = {'K': self.K, 'pos_embedding': self.pos_embedding}
        base_config = super(DistanceWeightedAttentionMP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shapes):
        input_shape = input_shapes[0]
        
        total_feat = input_shape[-1] + self.K * self.pos_emb.units
        with tf.name_scope(self.name + "/6/" + str(0)):
            self.attention_dense[0].build((input_shape[0], total_feat))

        for i in range(1, len(self.feature_tranformation_dense)):
            total_feat = 2*self.n_feature_transformation[i - 1] + self.K * self.pos_emb.units
            with tf.name_scope(self.name + "/6/" + str(i)):
                self.attention_dense[i].build((input_shape[0], total_feat))

        with tf.name_scope(self.name + "/7/"):
            self.pos_emb.build(input_shape)
            
        super(DistanceWeightedAttentionMP, self).build(input_shapes)

        
    def call(self, inputs):
        
        x, nidx, distancesq = inputs
        pos_emb = self.pos_emb(x)
        pos_emb = SelectWithDefault(nidx, pos_emb, -2.)
        pos_emb = tf.reshape(pos_emb, [-1, self.K*self.pos_emb.units])
        total_feat = [x]
        
        features = x
        norm = tf.cast(self.K, 'float32')
        #print('x',x.shape)
        
        for i in range(len(self.n_feature_transformation)):
            #this is quite light-weight, so use them all
            emb = tf.concat([features,pos_emb],axis=-1)
            w = self.attention_dense[i](emb) * tf.exp(-10. * distancesq)
            
            #print('w',w.shape)
            #print('features',features.shape)
            
            features = self.feature_tranformation_dense[i](features)
            prev_feat = features
            
            features = norm * AccumulateLinKnn(w,  features, nidx)[0]
            features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
            features -= tf.tile(prev_feat, [1, 2])
            total_feat.append(features)

        out = tf.concat(total_feat,axis=-1)
        return out

class AttentionMP(DistanceWeightedAttentionMP):
    
    def __init__(self, *args,**kwargs):
        super(AttentionMP, self).__init__(*args,**kwargs)
        
    def call(self, inputs):
        x, nidx = inputs
        distancesq = tf.zeros_like(nidx, dtype='float32')
        return super(AttentionMP, self).call([x, nidx, distancesq])
        
class EdgeConvStatic(tf.keras.layers.Layer):
    '''
    just like edgeconv but with static edges
    
    Input:
    - features
    - neighbour indices
    
    output:
    - new features
    
    :param n_feature_transformation: transformations on the edges (list)
    :param select_neighbours: if False, input is expected to be features (V x K x F) only (default: True)
    '''

    def __init__(self, 
                 n_feature_transformation,
                 select_neighbours=True,
                 add_mean=False,
                 **kwargs):
        super(EdgeConvStatic, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.select_neighbours = select_neighbours
        self.feature_tranformation_dense = []
        self.add_mean = add_mean
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense.append(tf.keras.layers.Dense(self.n_feature_transformation[i],
                                                                              activation='elu')) 
    
    def get_config(self):
        config = {'n_feature_transformation': self.n_feature_transformation,
                  'add_mean': self.add_mean,
                  'select_neighbours': self.select_neighbours,
        }
        base_config = super(EdgeConvStatic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shapes):
        input_shape = input_shapes
        if self.select_neighbours:
            input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.feature_tranformation_dense[0].build((input_shape[0],None,input_shape[-1]))

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense[i].build((input_shape[0], self.n_feature_transformation[i - 1]))

        super(EdgeConvStatic, self).build(input_shapes)

    
    def compute_output_shape(self, input_shapes): # data, idxs
        if self.add_mean:
            return (None, 2*self.feature_tranformation_dense[-1].units)
        else:
            return (None, self.feature_tranformation_dense[-1].units)
    
    
    def call(self, inputs):
        feat = inputs
        neighfeat = feat
        
        if self.select_neighbours:
            feat, nidx = inputs
            
            nF = feat.shape[-1]
            nN = nidx.shape[-1]
            
            TFnidx = tf.expand_dims(tf.where(nidx<0,0,nidx),axis=2)
            neighfeat = tf.gather_nd(feat, TFnidx)
            neighfeat = tf.where(tf.expand_dims(nidx,axis=2)<0, 0., neighfeat)#zero pad
            neighfeat = tf.reshape(neighfeat, [-1, nN, nF])
            
            neighfeat = neighfeat - feat[:, tf.newaxis, :]
        else:
            neighfeat = neighfeat - neighfeat[:,0:1,:]
        
        for t in self.feature_tranformation_dense:
            neighfeat = t(neighfeat)
            
        out = tf.reduce_max(neighfeat,axis=1)
        if self.add_mean:
            out = tf.concat([out, tf.reduce_mean(neighfeat,axis=1) ],axis=-1)
        return out
        
        
class XYZtoXYZPrime(tf.keras.layers.Layer):
    
    def call(self, inputs):
        x = inputs[...,0:1] 
        y = inputs[...,1:2] 
        z = inputs[...,2:3] 
        r = tf.sqrt(tf.reduce_sum(inputs**2, axis=-1, keepdims=True) + 1e-2)
        
        #also adjust scale a bit
        xprime = x / tf.where(z == 0., tf.sign(z)*1., z *10.)
        yprime = y / tf.where(z == 0., tf.sign(z)*1., z *10.)
        zprime = r / 100.
        
        return tf.concat([xprime,yprime,zprime], axis=-1)
        


        
class SingleLocalGravNetAttention(tf.keras.layers.Layer):
    
    def __init__(self, 
                 n_keys : int,
                 n_values : int,
                 n_pre_keys : int = -1,
                 **kwargs):
        '''
        One head
        
        Inputs:
        - features
        - nidx
        - distsq
        '''
        
        super(SingleLocalGravNetAttention, self).__init__(**kwargs)
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_pre_keys = n_pre_keys
        
        with tf.name_scope(self.name + "/1/"):
            self.key_dense = tf.keras.layers.Dense(self.n_keys,activation='tanh')#better behaved
        with tf.name_scope(self.name + "/2/"):
            self.query_dense = tf.keras.layers.Dense(self.n_keys,activation='tanh')#better behaved
        with tf.name_scope(self.name + "/3/"):
            self.value_dense = tf.keras.layers.Dense(self.n_values)
        
        if self.n_pre_keys > 0:
            with tf.name_scope(self.name + "/4/"):
                self.pre_key_dense = tf.keras.layers.Dense(self.n_pre_keys,activation='elu')
        
    
    def get_config(self):
        config = {'n_keys': self.n_keys,
                  'n_values': self.n_values,
                  'n_pre_keys': self.n_pre_keys}
        base_config = super(SingleLocalGravNetAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
        
    
    def build(self, input_shapes):
        input_shape = input_shapes[0]
        
        with tf.name_scope(self.name + "/1/"):
            if self.n_pre_keys > 0:
                self.key_dense.build((input_shape[0],1,self.n_pre_keys+1))
            else:
                self.key_dense.build(input_shape)
        with tf.name_scope(self.name + "/2/"):
            self.query_dense.build(input_shape)
        with tf.name_scope(self.name + "/3/"):
            self.value_dense.build(input_shape)
            
        if self.n_pre_keys > 0:
            with tf.name_scope(self.name + "/4/"):
                self.pre_key_dense.build(input_shape)
            
        super(SingleLocalGravNetAttention, self).build(input_shapes)
    
    def call(self, inputs):
        feat, nidx, distsq = inputs
        
        d_k = tf.cast(nidx.shape[1], 'float32') 
        
        #take into account that these are relative properties between neighbours
        if self.n_pre_keys > 0:
            keys = self.pre_key_dense(feat)
            n_keys = SelectWithDefault(nidx,keys,0.) # [V x K x F]
            #add distance info but no gradient
            n_keys = tf.concat([n_keys, tf.stop_gradient(distsq[...,tf.newaxis]) ],axis=-1)
            n_keys = n_keys - n_keys[:,0:1]
            n_keys = self.key_dense(n_keys)
        else:
            keys = self.key_dense(feat)
            n_keys = SelectWithDefault(nidx,keys,0.) # [V x K x F]
        
        queries = self.query_dense(feat)
        values = self.value_dense(feat)
        
        # now only create attention for the neighbours
        att = tf.reduce_sum(tf.expand_dims(queries, axis = 1) * n_keys, axis=-1)
        att = tf.nn.softmax(att / tf.sqrt(d_k), axis=-1) # [V x K ]
        
        #multiply with gravnet-style attention, but less strongly weightedg
        att = att * tf.exp(-2. * distsq)
        att = att * d_k #accumulateknn includes 1/d_k, so make the weights O(1) again
        
        acc = AccumulateLinKnn(att, values, nidx, force_tf=False, mean_and_max=True)[0] #[ V x 2F] -> one head
        acc = tf.reshape(acc, [-1, 2*values.shape[-1]])
        return acc

        
class LocalGravNetAttention(tf.keras.layers.Layer):
    '''
    
    Input:
    - features
    - neighbour indices
    - distance_sq
    
    output:
    - new features
    
    :param n_heads: number of heads
    :param n_keys: number of key features == n_queue *per* head
    :param n_values: number of values *per* head
    '''

    def __init__(self, 
                 n_heads : int, 
                 n_keys : int,
                 n_values : int,
                 n_pre_keys : int = -1,
                 **kwargs):
        
        super(LocalGravNetAttention, self).__init__(**kwargs)
        
        self.n_heads = n_heads
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_pre_keys = n_pre_keys
        
        self.heads=[]
        
        for i in range(self.n_heads):
            with tf.name_scope(self.name + "/1/"+str(i)):
                self.heads.append( SingleLocalGravNetAttention(n_keys, n_values, n_pre_keys=n_pre_keys) )
                
        
    def get_config(self):
        config = {'n_heads': self.n_heads,
                  'n_keys': self.n_keys,
                  'n_values': self.n_values,
                  'n_pre_keys':self.n_pre_keys}
        base_config = super(LocalGravNetAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
        
    
    def build(self, input_shapes):
        for i in range(self.n_heads):
            with tf.name_scope(self.name + "/1/"+str(i)):
                self.heads[i].build(input_shapes)
        
        super(LocalGravNetAttention, self).build(input_shapes)
    
    def call(self, inputs):
        assert len(inputs) == 3
        x, _, _ = inputs
        out=[]
        for h in self.heads:
            out.append(h(inputs))
        out = tf.concat(out, axis=-1)
        return out
    
    
class FlatNeighbourFeatures(tf.keras.layers.Layer):  
    '''
    
    Input:
    - features
    - neighbour indices
    
    output:
    - flattened, zero padded neighbour features (big)
    
    '''

    def __init__(self,**kwargs):
        super(FlatNeighbourFeatures, self).__init__(**kwargs)
        
    def call(self, inputs):
        feat,nidx = inputs
        
        n_feat = SelectWithDefault(nidx,feat,0.) # [V x K x F]
        return tf.reshape(n_feat, [-1, nidx.shape[1] * feat.shape[1]])
    
    
