import tensorflow as tf
import pdb
import yaml
import os
from select_knn_op import SelectKnn
from slicing_knn_op import SlicingKnn
from select_mod_knn_op import SelectModKnn
from accknn_op import AccumulateKnn
from local_cluster_op import LocalCluster
from local_group_op import LocalGroup
from local_distance_op import LocalDistance
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
import numpy as np
#just for the moment
#### helper###
from datastructures import TrainData_OC,TrainData_NanoML
from initializers import EyeInitializer

from oc_helper_ops import SelectWithDefault

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
        assert len(inputs.shape)==2
        return tf.cast(inputs[:,0],dtype='int32')
        
    

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
    def __init__(self,**kwargs):
        super(PrintMeanAndStd, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shapes):
        #return input_shapes[0]
        return input_shapes
        
    def call(self, inputs):
        tf.print(self.name,'mean',tf.reduce_mean(inputs,axis=0),summarize=100)
        tf.print(self.name,'std',tf.math.reduce_std(inputs,axis=0),summarize=100)
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
        
    
class GooeyBatchNorm(tf.keras.layers.Layer):
    def __init__(self,
                 viscosity=0.2,
                 fluidity_decay=1e-4,
                 max_viscosity=1.,
                 epsilon=1e-4,
                 print_viscosity=False,
                 soften_update: float = 0.,
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
        
    def get_config(self):
        config = {'viscosity': self.viscosity_init,
                  'fluidity_decay': self.fluidity_decay,
                  'max_viscosity': self.max_viscosity,
                  'epsilon': self.epsilon,
                  'print_viscosity': self.print_viscosity,
                  'soften_update': self.soften_update
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
                                    initializer = 'zeros', trainable = False) 
        
        self.variance = self.add_weight(name = 'variance',shape = shape, 
                                    initializer = 'ones', trainable = False) 
        
        self.viscosity = tf.Variable(initial_value=self.viscosity_init, 
                                         name='viscosity',
                                         trainable=False,dtype='float32')
            
        super(GooeyBatchNorm, self).build(input_shapes)
    
    def _calc_soft_update(self, old, new, training):
        delta = new-old
        #soft update, avoid too strong jumps
        if self.soften_update>0:
            #scale down relative change
            delta = tf.nn.softsign(tf.math.divide_no_nan(self.soften_update*delta,old))*old
            delta /= self.soften_update
        update = old + (1. - self.viscosity)*delta
        return tf.keras.backend.in_train_phase(update,old,training=training)

    def call(self, inputs, training=None):
        #x, _ = inputs
        x = inputs
        
        #update only if trainable flag is set, AND in training mode
        if self.trainable:
            newmean = tf.reduce_mean(x,axis=0,keepdims=True) #FIXME
            update = self._calc_soft_update(self.mean,newmean,training)
            tf.keras.backend.update(self.mean,update)
            
            newvar = tf.math.reduce_std(x-self.mean,axis=0,keepdims=True) #FIXME
            update = self._calc_soft_update(self.variance,newvar,training)
            tf.keras.backend.update(self.variance,update)
            
            #increase viscosity
            if self.fluidity_decay > 0:
                newvisc = self.viscosity + (self.max_viscosity - self.viscosity)*self.fluidity_decay
                newvisc = tf.keras.backend.in_train_phase(newvisc,self.viscosity,training=training)
                tf.keras.backend.update(self.viscosity,newvisc)
                if self.print_viscosity:
                    tf.print(self.name, 'viscosity',newvisc)
        
        #apply
        x -= self.mean
        x = tf.math.divide_no_nan(x, self.variance + self.epsilon)
        
        #print(self.name,'corr x mean', tf.reduce_mean(x))
        #print(self.name,'corr x var', tf.math.reduce_std(x))
        
        return x

        
    
    

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
        self.model.load_weights(self.path)
        self.model.trainable = False
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
                    layer = tf.keras.layers.Dense(units=output_dim, trainable=False)
                else:
                    output_dim = nodes[i]
                    layer = tf.keras.layers.Dense(units=output_dim, activation='elu', trainable=False)
                layer.build(input_dim)
                layer.set_weights(self.model.layers[i+1].get_weights())
                self.layers.append(layer)

        super(ApproxPCA, self).build(input_shapes)  
        
        
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
    
    
    
class LocalDistanceScaling (tf.keras.layers.Layer):
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
        return dist*tf.where(scale>=0,scale_pos, scale_neg)
    
    def call(self, inputs):
        dist,scale = inputs
        return LocalDistanceScaling.raw_call(dist,scale,self.max_scale,self.b,self.c)
    

 
        
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
        
        nweights = tf.nn.relu(nweights)#secure
        nweights = -tf.math.log(nweights+1e-6) #accumulateKnn has a minus sign
        
        f,_ = AccumulateKnnSumw(nweights, feat, nidx)
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


class KNN(tf.keras.layers.Layer):
    def __init__(self,K: int, radius=-1., 
                 use_approximate_knn=True,
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
                  'use_approximate_knn': self.use_approximate_knn}
        base_config = super(KNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return (None, self.K+1),(None, self.K+1)

    @staticmethod 
    def raw_call(coordinates, row_splits, K, radius, use_approximate_knn):
        if use_approximate_knn:
            bin_width = radius # default value for SlicingKnn kernel
            idx,dist = SlicingKnn(K+1, coordinates,  row_splits,
                                  features_to_bin_on = (0,1),
                                  bin_width=(bin_width,bin_width))
        else:
            idx,dist = SelectKnn(K+1, coordinates,  row_splits,
                                 max_radius= radius, tf_compatible=False)

        idx = tf.reshape(idx, [-1,K+1])
        dist = tf.reshape(dist, [-1,K+1])
        return idx,dist

    def update_dynamic_radius(self, dist, training=None):
        if self.dynamic_radius is None or not self.trainable:
            return
        #update slowly, with safety margin
        update = tf.reduce_max(dist)*1.2
        update = self.dynamic_radius + 0.1*(update-self.dynamic_radius)
        updated_radius = tf.keras.backend.in_train_phase(update,self.dynamic_radius,training=training)
        tf.keras.backend.update(self.dynamic_radius,updated_radius)
        
    def call(self, inputs, training=None):
        coordinates, row_splits = inputs
        if self.dynamic_radius is None:
            return KNN.raw_call(coordinates, row_splits, self.K, self.radius, self.use_approximate_knn)
        else:
            idx,dist = KNN.raw_call(coordinates, row_splits, self.K, self.dynamic_radius, self.use_approximate_knn)
            self.update_dynamic_radius(dist,training)
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
    def __init__(self,K: int, radius: float=-1., sort=True, **kwargs):
        """
        
        This layer will sort neighbour indices by distance and possibly select neighbours
        within a radius, or the closest ones up to K neighbours.
        
        Inputs: distances, neighbour indices
        
        Call will return 
         - neighbour distances sorted by distance (increasing)
         - neighbour indices sorted by distance (increasing)
        
        
        :param K: number of nearest neighbours, will do no selection if K<1
        :param radius: maximum distance of nearest neighbours (no effect if < 0)
        
        """
        super(SortAndSelectNeighbours, self).__init__(**kwargs) 
        self.K = K
        self.radius = radius
        self.sort=sort
        
        
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius,
                  'sort': self.sort}
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
    def raw_call(distances, nidx, K, radius, sort):
        
        if not sort:
            distances[:,:K],nidx[:,:K]
        
        tfdist = tf.where(nidx<0, 1e9, distances) #make sure the -1 end up at the end
        tfdist = tf.concat([tf.zeros_like(tfdist[:,0:1])-1.,tfdist[:,1:]  ],axis=1) #make sure 'self' remains

        sorting = tf.argsort(tfdist, axis=1)
        snidx = tf.gather(nidx,sorting,batch_dims=1) #_nd(nidx,sorting,batch_dims=1)
        sdist = tf.gather(distances,sorting,batch_dims=1)
        if K > 0:
            snidx = snidx[:,:K]
            sdist = sdist[:,:K]
            
        if radius > 0:
            snidx = tf.where(sdist > radius, -1, snidx)
            sdist = tf.where(sdist > radius, 0. , sdist)
            
        K = K if K>0 else distances.shape[1]
        #fix the shapes
        sdist = tf.reshape(sdist, [-1, K])
        snidx = tf.reshape(snidx, [-1, K])
            
        return sdist, tf.cast(snidx, tf.int32) #just to avoid keras not knowing the dtype


        
    def call(self, inputs):
        distances, nidx = inputs
        return SortAndSelectNeighbours.raw_call(distances,nidx,self.K,self.radius,self.sort)
        #make TF compatible
        
        


class NoiseFilter(tf.keras.layers.Layer):
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
            return (input_shapes[0][-1], input_shapes[1][-1]) 
        return (input_shapes[0][-1]-1, input_shapes[1][-1]) # K x F
    
    def call(self, inputs):
        selffeat = tf.expand_dims(inputs[1],axis=1)
        if self.addself:
            edges = selffeat - SelectWithDefault(inputs[0][:,:], inputs[1], -1.)
        else:
            edges = selffeat - SelectWithDefault(inputs[0][:,1:], inputs[1], -1.)
        return edges
    

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
    
    def get_config(self):
        config = {'threshold': self.threshold}
        base_config = super(EdgeSelector, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
       
    def call(self, inputs):   
        assert len(inputs) == 2
        score, nidx = inputs
        assert len(score.shape) == 3 and len(nidx.shape) == 2
        
        score = tf.concat([tf.ones_like(score[:,0:1,:]),score],axis=1)#add self score always 1
        return tf.where(score[:,:,0] < self.threshold, -1, nidx)
        
    

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
    def __init__(self, **kwargs):
        '''
        Input: 
        - edge scores (V x K-1 x 1)
        - neighbour indices (V x K)
        
        Output:
        - group score (V x 1)
        
        '''
        
        if 'dynamic' in kwargs:
            super(GroupScoreFromEdgeScores, self).__init__(**kwargs)
        else:
            super(GroupScoreFromEdgeScores, self).__init__(dynamic=False,**kwargs)
    
    def compute_output_shape(self, input_shapes): 
        return (1,)
            
        #no config
    def call(self, inputs): 
        score, nidx = inputs
        #take mean
        active = tf.cast(nidx>-1, 'float32')
        n_neigh = tf.math.count_nonzero(active, axis=1)# V 
        n_neigh = tf.cast(n_neigh,dtype='float32') - 1.
        groupscore = tf.reduce_sum(active[:,1:]*score[:,:,0], axis=1)
        #give slight priority to larger groups
        groupscore = tf.math.divide_no_nan(groupscore,n_neigh+1.)#prefer larger groups
        return tf.expand_dims(groupscore,axis=1)


### soft pixel section
class NeighbourGroups(tf.keras.layers.Layer):
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
            if self.print_reduction:
                print(self.name,'reduced from', dirnidx.shape[0] ,'to',sel.shape[0])
        
        back = ggather
        if self.return_backscatter:
            back = [tf.shape(nidxs)[0], sel]
            
        dirnidx = tf.reshape(dirnidx, tf.shape(nidxs))#to make shape clear
            
        return dirnidx, sel, back, rs

    def compute_output_shape(self, input_shapes):
        score, nidxs, row_splits = input_shapes
        if self.return_backscatter:
            return nidxs, score, [(1, ), score], row_splits #first dim omitted
        else:
            return nidxs, score, score, row_splits
        

class AccumulateNeighbours(tf.keras.layers.Layer):
    def __init__(self, mode='meanmax' , **kwargs):
        '''
        Inputs: feat, nidx
        
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
            return (3*fshape[1], )
        else:
            return (2*fshape[1], )

    def get_min(self,ndix,feat):
        out,_ = AccumulateKnn(tf.zeros_like(ndix,dtype='float32'), 
                          -feat, ndix,mean_and_max=True)
        out=out[:,feat.shape[1]:]
        return -tf.reshape(out,[-1, feat.shape[1]])  
          
    def call(self, inputs, training=None):
        assert len(inputs)==2
        feat,ndix = inputs
        
        zeros = tf.cast(tf.zeros_like(ndix),dtype='float32')
        K = tf.cast(ndix.shape[1],dtype='float32')
        #K = tf.expand_dims(tf.expand_dims(K,axis=0),axis=0)
        if self.mode == 'mean' or self.mode == 'meanmax':
            out,_ = AccumulateKnnSumw(zeros, 
                          feat, ndix,mean_and_max=self.mode == 'meanmax')
            return out
        if self.mode=='gnlike':
            out,_ = AccumulateKnn(zeros, 
                          feat, ndix)
            return tf.reshape(out,[-1, 2*feat.shape[1]])
        if self.mode=='sum':
            out,_ = AccumulateKnn(zeros,feat, ndix,mean_and_max=False)
            out*=K
            return tf.reshape(out,[-1, feat.shape[1]])              
        if self.mode == 'max':
            out,_ = AccumulateKnn(zeros, 
                          feat, ndix,mean_and_max=True)
            out=out[:,feat.shape[1]:]
            return tf.reshape(out,[-1, feat.shape[1]])
        if self.mode == 'min':
            return self.get_min(ndix,feat)
        if self.mode == 'minmeanmax':
            meanmax,_ = AccumulateKnnSumw(zeros, 
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

class RaggedGravNet(tf.keras.layers.Layer):
    def __init__(self,
                 n_neighbours: int,
                 n_dimensions: int,
                 n_filters : int,
                 n_propagate : int,
                 return_self=True,
                 sumwnorm=False,
                 feature_activation='relu',
                 use_approximate_knn=True,
                 coord_initialiser_noise=1e-2,
                 use_dynamic_knn=True,
                 **kwargs):
        """
        Call will return output features, coordinates, neighbor indices and squared distances from neighbors

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
        :param kwargs:
        """
        super(RaggedGravNet, self).__init__(**kwargs)

        n_neighbours += 1  # includes the 'self' vertex
        assert n_neighbours > 1

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.return_self = return_self
        self.sumwnorm = sumwnorm
        self.feature_activation = feature_activation
        self.use_approximate_knn = use_approximate_knn
        self.use_dynamic_knn = use_dynamic_knn
        
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
        update = tf.reduce_max(dist)*1.2
        mean_dist = tf.reduce_mean(dist)
        low_update = tf.where(update>2.,2.,update)#receptive field ends at 1.
        update = tf.where(low_update>2.*mean_dist,low_update,2.*mean_dist)#safety setting to not loose all neighbours
        update += 1e-3
        update = self.dynamic_radius + 0.1*(update-self.dynamic_radius)
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
        
        coordinates = self.input_spatial_transform(x)
        neighbour_indices, distancesq, sidx, sdist = self.compute_neighbours_and_distancesq(coordinates, row_splits, training)
        neighbour_indices = tf.reshape(neighbour_indices, [-1, self.n_neighbours-1]) #for proper output shape for keras
        distancesq = tf.reshape(distancesq, [-1, self.n_neighbours-1])

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
               (input_shapes[0][0], self.n_neighbours),\
               (input_shapes[0][0], self.n_neighbours)
        else:
            return (input_shapes[0][0], 2*self.n_filters),\
               (input_shapes[0][0], self.n_dimensions),\
               (input_shapes[0][0], self.n_neighbours-1),\
               (input_shapes[0][0], self.n_neighbours-1)
              
    

    def compute_neighbours_and_distancesq(self, coordinates, row_splits, training):
        if self.use_approximate_knn:
            bin_width = self.dynamic_radius # default value for SlicingKnn kernel
            idx,dist = SlicingKnn(self.n_neighbours, coordinates,  row_splits,
                                  features_to_bin_on = (0,1),
                                  bin_width=(bin_width,bin_width))
        else:
            idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                                 max_radius= -1.0, tf_compatible=False)
        idx = tf.reshape(idx, [-1, self.n_neighbours])
        dist = tf.reshape(dist, [-1, self.n_neighbours])
        
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
                  'use_approximate_knn':self.use_approximate_knn}
        base_config = super(RaggedGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



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
        f,_ = AccumulateKnn(tf.cast(neighbour_indices*0, tf.float32),  features, neighbour_indices)
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

    def __init__(self, n_feature_transformation,
                 sumwnorm=False,
                 **kwargs):
        super(DistanceWeightedMessagePassing, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.sumwnorm = sumwnorm
        self.feature_tranformation_dense = []
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense.append(tf.keras.layers.Dense(self.n_feature_transformation[i],
                                                                              activation='relu'))  # restrict variations a bit

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense[i].build((input_shape[0], self.n_feature_transformation[i - 1] * 2))

        super(DistanceWeightedMessagePassing, self).build(input_shapes)


    def get_config(self):
        config = {'n_feature_transformation': self.n_feature_transformation,
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



class EdgeConvStatic(tf.keras.layers.Layer):
    '''
    just like edgeconv but with static edges
    
    Input:
    - features
    - neighbour indices
    
    output:
    - new features
    
    :param n_feature_transformation: transformations on the edges (list)
    :param select_neighbours: if False, input is expected to be features (V x K x F) only
    '''

    def __init__(self, 
                 n_feature_transformation,
                 select_neighbours=False,
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
            
        neighfeat = neighfeat - neighfeat[:,:,0:1] 
        
        for t in self.feature_tranformation_dense:
            neighfeat = t(neighfeat)
            
        out = tf.reduce_max(neighfeat,axis=1)
        if self.add_mean:
            out = tf.concat([out, tf.reduce_mean(neighfeat,axis=1) ],axis=-1)
        return out
        
        
        
        

        
        
        
        
        
