import tensorflow as tf
from select_knn_op import SelectKnn
from select_mod_knn_op import SelectModKnn
from accknn_op import AccumulateKnn
from local_cluster_op import LocalCluster
from local_group_op import LocalGroup
from local_distance_op import LocalDistance
from lossLayers import LLClusterCoordinates, LLLocalClusterCoordinates
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
                 **kwargs):
        super(GooeyBatchNorm, self).__init__(**kwargs)
        
        assert viscosity >= 0 and viscosity <= 1.
        assert fluidity_decay >= 0 and fluidity_decay <= 1.
        assert max_viscosity >= viscosity
        
        self.fluidity_decay = fluidity_decay
        self.max_viscosity = max_viscosity
        self.viscosity_init = viscosity
        self.epsilon = epsilon
        self.print_viscosity = print_viscosity
        
    def get_config(self):
        config = {'viscosity': self.viscosity_init,
                  'fluidity_decay': self.fluidity_decay,
                  'max_viscosity': self.max_viscosity,
                  'epsilon': self.epsilon,
                  'print_viscosity': self.print_viscosity
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
    

    def call(self, inputs, training=None):
        #x, _ = inputs
        x = inputs
        
        #update only if trainable flag is set, AND in training mode
        if self.trainable:
            newmean = tf.reduce_mean(x,axis=0,keepdims=True) #FIXME
            newmean = (1. - self.viscosity)*newmean + self.viscosity*self.mean
            updated_mean = tf.keras.backend.in_train_phase(newmean,self.mean,training=training)
            tf.keras.backend.update(self.mean,updated_mean)
            
            newvar = tf.math.reduce_std(x-self.mean,axis=0,keepdims=True) #FIXME
            newvar = (1. - self.viscosity)*newvar + self.viscosity*self.variance
            updated_var = tf.keras.backend.in_train_phase(newvar,self.variance,training=training)
            tf.keras.backend.update(self.variance,updated_var)
            
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
        
        return x


class NormalizeInputShapes(tf.keras.layers.Layer):
    def __init__(self,
                 **kwargs):
        super(NormalizeInputShapes, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shapes):
        outshapes=[]
        for s in input_shapes:
            if len(s)<2:
                outshapes.append((None, 1))
            else:
                outshapes.append(s)
            
        return outshapes
    
    def call(self, inputs):
        outs=[]
        for i in inputs:
            if len(i.shape)<2:
                outs.append(tf.reshape(i,[-1,1]))
            else:
                outs.append(i)
        return outs
            
    
    

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
    
    
    
class LocalDistanceScaling (tf.keras.layers.Layer):
    def __init__(self,
                 max_scale = 10,
                 **kwargs):
        """
        Inputs: 
        - distances (V x N)
        - scaling (V x 1)
        
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
    

class LocalClustering(tf.keras.layers.Layer):
    def __init__(self,
                 print_reduction=False,
                 **kwargs):
        """
        
        This layer performs a local clustering (a bit similar to object condensation):
        
        Inputs are: 
         - neighbour indices (V x K)
         - hierarchy tensor (V x 1) to determine the clustering hierarchy
         - row splits
         
        The layer will select all neighbours and assign them to the vertex with the highest hierarchy score.
        These vertices cannot give rise to a new cluster anymore, nor can be part of any other cluster.
        The process gets repeated following the hierarchy score (decreasing) until there are no vertices left.
        
         
        Call will return:
         - indices to select the cluster centres, 
         - updated row splits for after applying the selection
         - indices to gather back the original dimensionality by repetion
        
        This layer does *not* introduce any new gradients on anything (e.g. the hierarchy score),
        so that needs to be done by hand. (e.g. using LLLocalClusterCoordinates).
        
        Since this layers functionality is inherently sequential, it will run on CPU. Right now, 
        the implementation (local_cluster_xx) can still be optimised for using multiple CPU cores.
        A simple way of (mild) parallelising would be given by the row splits.
        
        """
        if 'dynamic' in kwargs:
            super(LocalClustering, self).__init__(**kwargs)
        else:
            super(LocalClustering, self).__init__(dynamic=False,**kwargs)
        self.print_reduction=print_reduction
        
    def get_config(self):
        config = {'print_reduction': self.print_reduction}
        base_config = super(LocalClustering, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shapes):
        return (None,1), (None,), (None,1)
    
    def build(self, input_shapes):
        super(LocalClustering, self).build(input_shapes)
        
    @staticmethod
    def raw_call(neighs, hier, row_splits,print_reduction,name, threshold=-1):
        if row_splits.shape[0] is None:
            return tf.zeros_like(hier, dtype='int32'), row_splits, tf.zeros_like(hier, dtype='int32')
        
        if hier.shape[1] > 1:
            raise ValueError(name+' received wrong hierarchy shape')
        
        hierarchy_idxs=[]
        for i in range(row_splits.shape[0] - 1):
            a = tf.argsort(hier[row_splits[i]:row_splits[i+1]],axis=0, direction='DESCENDING')
            hierarchy_idxs.append(a+row_splits[i])
        hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
        if threshold > 0 and threshold < 1:
            hierarchy_idxs = tf.where( hier < threshold, -(hierarchy_idxs+1), hierarchy_idxs )
        
        
        rs,sel,ggather = LocalCluster(neighs, hierarchy_idxs, row_splits)
        
        #keras does not like gather_nd outputs
        sel = tf.reshape(sel, [-1,1])
        rs = tf.reshape(rs, [-1])
        rs = tf.cast(rs, tf.int32) #just so keras knows
        ggather = tf.reshape(ggather, [-1,1])
        if print_reduction:
            tf.print(name,'reduction',round(float(sel.shape[0])/float(ggather.shape[0]),2),'to',sel.shape[0])
        return sel, rs, ggather
        
    def call(self, inputs):
        neighs, hier, row_splits = inputs
        return LocalClustering.raw_call(neighs, hier, row_splits,self.print_reduction,self.name)
        
        
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
    def __init__(self, **kwargs):    
        """
        
        Either applies a scattering or gathering operation depending on the inputs
        (compatible with the outputs of NoiseFilter and NeighbourGroups).
        
        In general preferred
        
        """  
        self.gathers=[]
        if 'dynamic' in kwargs:
            super(MultiBackScatterOrGather, self).__init__(**kwargs) 
        else:
            super(MultiBackScatterOrGather, self).__init__(dynamic=False,**kwargs) 
        
        
    @staticmethod 
    def raw_call(x, scatters):
        xin=x
        for k in range(len(scatters)):
            l = len(scatters) - k - 1
            if scatters[l] is list:
                V, scidx = scatters[l]
                #print('scatters[l]',scatters[l])
                #cast is needed because keras layer out dtypes are not really working
                shape = tf.concat([tf.expand_dims(V,axis=0), tf.shape(x)[1:]],axis=0)
                x = tf.scatter_nd(scidx, x, shape)
            else:
                x = SelectFromIndices.raw_call(tf.cast(scatters[l],tf.int32), 
                                           [x], [ [-1]+list(x.shape[1:]) ])[0]
                
        return tf.reshape(x,[-1,xin.shape[1]])
        
    def call(self, inputs):
        x, scatters = inputs
        if x.shape[0] is None:
            return tf.reshape(x,[-1 ,x.shape[1]])
        xnew = MultiBackScatterOrGather.raw_call(x,scatters)
        return xnew    
        
        
############# Local clustering section ends


class KNN(tf.keras.layers.Layer):
    def __init__(self,K: int, radius: float=-1., **kwargs):
        """
        
        Select self+K nearest neighbours, with possible radius constraint.
        
        Call will return 
         - self + K neighbour indices of K neighbours within max radius
         - distances to self+K neighbours
        
        Inputs: coordinates, row_splits
        
        :param K: number of nearest neighbours
        :param radius: maximum distance of nearest neighbours
        """
        super(KNN, self).__init__(**kwargs) 
        self.K = K
        self.radius = radius
        
        
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius}
        base_config = super(KNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return (None, self.K+1),(None, self.K+1)

    @staticmethod 
    def raw_call(coordinates, row_splits, K, radius):
        idx,dist = SelectKnn(K+1, coordinates,  row_splits,
                             max_radius= radius, tf_compatible=False)

        idx = tf.reshape(idx, [-1,K+1])
        dist = tf.reshape(dist, [-1,K+1])
        return idx,dist

    def call(self, inputs):
        coordinates, row_splits = inputs
        return KNN.raw_call(coordinates, row_splits, self.K, self.radius)
        

        
    
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
        
        


class GraphClusterReshape(tf.keras.layers.Layer):
    '''
    
    This layer implements a graph reshaping. 
    
    Inputs:
    - features
    - neighbour indices
    
    The first dimention of neighbour indices can be smaller than of features 
    (e.g. as selected clusters from LocalCluster).
    The features of all neighbours are flattenend to the new features of the whole
    cluster. In case there are less neighbours than the second dimension of neighbour indices,
    the corresponding entries are zero-padded.
    (technically, <K neighbours are implemented as -1 indices in this all other layers)
    
    (There is no information loss, therefore 'reshaping')
    
    Output:
    - reshaped features
    
    
    '''
    def __init__(self, **kwargs):
        super(GraphClusterReshape, self).__init__(dynamic=True,**kwargs) 
    #has no init
    def build(self, input_shapes):
        super(GraphClusterReshape, self).build(input_shapes)
        
    def compute_output_shape(self, input_shapes): #features, nidx = inputs
        return (None, input_shapes[0][1]*input_shapes[1][1])
        
        
    @staticmethod 
    def raw_call(features, nidx, reshape=True):
        nidx = tf.cast(nidx,tf.int32)#just because of keras build
        TFnidx = tf.expand_dims(tf.where(nidx<0,0,nidx),axis=2)
        
        gfeat = tf.gather_nd(features, TFnidx)
        gfeat = tf.reshape(gfeat, [-1, tf.shape(nidx)[1], tf.shape(features)[1]])#same shape
        
        out = tf.where(tf.expand_dims(nidx,axis=2)<0, 0., gfeat)
        if reshape:
            out = tf.reshape(out, [-1, tf.shape(features)[1]*tf.shape(nidx)[1]])
        else:
            out = tf.reshape(out, [-1, tf.shape(nidx)[1], tf.shape(features)[1]])
        return out
    
        
    def call(self, inputs):
        features, nidx = inputs
        return GraphClusterReshape.raw_call(features, nidx)
        
        
### compound layers for convenience


class LocalClusterReshapeFromNeighbours(tf.keras.layers.Layer):
    def __init__(self, K=-1, 
                 radius=-1, 
                 print_reduction=False, 
                 loss_enabled=False, 
                 loss_scale = 1., 
                 loss_repulsion=0.5,
                 print_loss=False,
                 **kwargs):
        
        raise ValueError("DEPRECATED, use LNC instead")
        
class LocalClusterReshapeFromNeighbours2(tf.keras.layers.Layer):
    def __init__(self, K=-1, 
                 radius=-1, 
                 print_reduction=False, 
                 loss_enabled=False, 
                 hier_transforms=[],
                 loss_scale = 1., 
                 loss_repulsion=0.5,
                 print_loss=False,
                 **kwargs):
        
        raise ValueError("DEPRECATED, use LNC instead")


class NoiseFilter(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.1, 
                 loss_scale = 1., 
                 loss_enabled=False,
                 print_loss=False,
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
         - [] a list of all tensors to be filtered accordingly (at least one)
         - truth index
         
        Outputs:
         - row splits
         - backgather/(bsnumber backscatter indices)
         - [] the list of all other tensors to be filtered
        '''

        
        if 'dynamic' in kwargs:
            super(NoiseFilter, self).__init__(**kwargs)
        else:
            super(NoiseFilter, self).__init__(dynamic=False,**kwargs)
            
        self.threshold = threshold
        self.loss_scale = loss_scale
        self.loss_enabled = loss_enabled
        self.print_loss = print_loss
        self.print_reduction = print_reduction
        self.return_backscatter = return_backscatter
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'loss_scale': self.loss_scale,
                  'loss_enabled': self.loss_enabled,
                  'print_loss': self.print_loss,
                  'print_reduction': self.print_reduction,
                  'return_backscatter': self.return_backscatter}
        
        base_config = super(NoiseFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shapes):
        return input_shapes[1:-1] #all but score itself and truth indices
    
    
    def call(self, inputs):
        
        score, row_splits, *other, tidxs = inputs
        
        defbg = tf.expand_dims(tf.range(tf.shape(score)[0]),axis=1)
        if row_splits.shape[0] is None: #dummy execution
            bg = defbg
            if self.return_backscatter:
                bg = [tf.shape(score)[0], bg]
            return [row_splits, bg] + other
            
        #score loss
        if self.loss_enabled:
            notnoise = tf.where(tidxs>=0, tf.ones_like(score), 0.)
            
            Nnotnoise = tf.cast(tf.math.count_nonzero(notnoise,axis=0),dtype='float32')
            Nnoise = tf.cast(tf.math.count_nonzero(1.-notnoise,axis=0),dtype='float32')
            
            classloss = tf.keras.losses.binary_crossentropy(notnoise, score)
            
            notnoiseloss = tf.math.divide_no_nan(tf.reduce_sum(classloss*notnoise[:,0]),tf.squeeze(Nnotnoise))
            noiseloss = tf.math.divide_no_nan(tf.reduce_sum(classloss*(1.-notnoise[:,0])),tf.squeeze(Nnoise))
            
            classloss = notnoiseloss+noiseloss
            self.add_loss(classloss)
            if self.print_loss:
                accuracy = tf.where(score>0.5, notnoise, 0.)
                accuracy = tf.reduce_sum(accuracy) / tf.squeeze(Nnotnoise)
                print(self.name, ' loss ', classloss, ' ; accuracy', accuracy)
        
        if self.threshold<=0: #does nothing
            bg=defbg
            if self.return_backscatter:
                bg = [tf.shape(score)[0], defbg]
            return [row_splits, bg]+ other
        
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
            
        other = SelectFromIndices.raw_call(sel, other)
        
        if self.print_reduction:
            print(self.name,' reduction from ', int(row_splits[-1]), ' to ', int(newrs[-1]), ': ', float(row_splits[-1])/float(newrs[-1]))
        
        return [newrs, allbackgather] + other
        
        

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
                 threshold = 0.9, 
                 loss_scale = 1., 
                 loss_enabled = False,
                 print_loss=False,
                 use_truth=False,
                 **kwargs):
        '''
        Inputs: neighbour indices (V x K ) , edge score (V x K-1) , spectator weights, truth index
        Outputs: selected neighbour indices ('-1 masked')
        
        '''
        assert threshold<1 and threshold>=0
        self.threshold = threshold
        self.loss_scale = loss_scale
        self.loss_enabled = loss_enabled
        self.print_loss = print_loss
        self.use_truth = use_truth
        
        if 'dynamic' in kwargs:
            super(EdgeSelector, self).__init__(**kwargs)
        else:
            super(EdgeSelector, self).__init__(dynamic=False,**kwargs)
    
    def get_config(self):
        config = {'threshold': self.threshold,
                  'loss_enabled': self.loss_enabled,
                  'loss_scale': self.loss_scale}
        
        base_config = super(EdgeSelector, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
       
    def call(self, inputs):   
        
        nidx, score, specweights, tidxs = inputs
        #loss part
        if self.loss_enabled: #revised Nov 2021
            sel_tidxs = SelectWithDefault(nidx, tidxs, -1)
            #sanity check
            tf.assert_equal(tf.expand_dims(tidxs,axis=1),sel_tidxs[:,0:1])
            #make sure noise disappears
            sel_tidxs = tf.where(sel_tidxs<0,-2,sel_tidxs)
            
            sel_spec  = SelectWithDefault(nidx, specweights, 1.)[:,1:] #don't count -1 entries anyway
            
            #just downweight spectators slightly (don't let them build bridges!
            notspecmask = tf.where(sel_spec[:,:,0]>0, .9, tf.ones_like(score[:,:,0]))
            
            active = notspecmask*tf.where(nidx[:,1:]>=0, tf.ones_like(score[:,:,0]), 0.)
            #Nactive = tf.reduce_sum(active,axis=1)
            #mask spectators
            sameasprobe = tf.cast(tf.expand_dims(tidxs,axis=1) == sel_tidxs,dtype='float32')[:,1:]
            
                    
            edgeloss = tf.keras.losses.binary_crossentropy(sameasprobe, score)
            
            #print('>>>',sameasprobe[0], edgeloss[0], score[0])
            
            edgeloss *= active #includes spec 
            
            samesum = tf.reduce_sum(active*sameasprobe[:,:,0])
            notsamesum = tf.reduce_sum(active*(1.-sameasprobe[:,:,0]))
            
            
            abovethresh = tf.cast(score[:,:,0]>self.threshold,dtype='float32')
            rightpred = abovethresh * active * sameasprobe[:,:,0]
            efficiency = tf.reduce_sum(rightpred) / (tf.reduce_sum(active*sameasprobe[:,:,0])+1e-3)
            purity = tf.reduce_sum(rightpred) / (tf.reduce_sum(active*abovethresh)+1e-3)
            fakes = tf.reduce_sum(abovethresh * active*(1.-sameasprobe[:,:,0])) / (tf.reduce_sum(active*abovethresh)+1e-3)
            
            edgelosssame  = tf.math.divide_no_nan(tf.reduce_sum(sameasprobe[:,:,0]*edgeloss),samesum)
            edgelossnotsame = tf.math.divide_no_nan(tf.reduce_sum((1.-sameasprobe[:,:,0])*edgeloss),notsamesum)
            edgeloss = self.loss_scale * tf.reduce_mean(edgelosssame+edgelossnotsame)
            
            if hasattr(edgeloss, "numpy"):
                print(self.name,
                      'loss', edgeloss.numpy(), 
                      'same', edgelosssame.numpy(), 
                      'notsame', edgelossnotsame.numpy(), 
                      'purity', purity.numpy(),
                      'efficiency', efficiency.numpy(), 
                      'fakes', fakes.numpy(), 
                      'avg score', tf.reduce_mean(tf.reduce_sum(score[:,:,0],axis=1)/tf.reduce_sum(active,axis=1)),
                      'avg active', tf.reduce_mean(tf.reduce_sum(active,axis=1)).numpy(), 
                      'avg truth', tf.reduce_mean(tidxs).numpy(),
                      'max truth', tf.reduce_max(tidxs).numpy())
            
            self.add_loss( edgeloss )
            
        score = tf.concat([tf.ones_like(score[:,0:1,:]),score],axis=1)#add self score always 1
        
        #the scores are still training, but we provide truth to the following layers to train them
        #simultaneously with the edge selector. Later this must be switched off
        if self.use_truth:
            print("WARNING: EdgeSelector",self.name, "uses truth! Can be useful in the first training rounds, but MUST BE SWITCHED OFF LATER")
            sel_tidxs = SelectWithDefault(nidx, tidxs, -1)
            sel_tidxs = tf.where(sel_tidxs<0,-2,sel_tidxs)
            tscore = tf.cast(tf.expand_dims(tidxs,axis=1) == sel_tidxs,dtype='float32')[:,1:]
            #add one real prediction to add some non-truth noise
            score = tf.concat([tf.ones_like(score[:,0:1]),tscore[:,:-1], score[:,-1:] ],axis=1)#self always true
        
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
        - edge scores (V x K x 1)
        - neighbour indices
        
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
        groupscore = tf.math.divide_no_nan(groupscore,n_neigh+.1)
        return tf.expand_dims(groupscore,axis=1)
        
        
                
class LNC(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.9, 
                 sum_other=[],
                 select_other=[],
                 loss_scale = 1., 
                 distance_loss_scale = 1., 
                 
                 print_reduction=False, 
                 loss_enabled=False, 
                 print_loss=False,
                 use_spectators=False,
                 return_neighbours=False,
                 noise_loss_scale : float = 0.1,
                 **kwargs):
        '''
        Local Neighbour Clustering
        This should be an improvement over LocalClusterReshapeFromNeighbours2 with fewer hyperparameters
        and actually per-point exclusive clustering (not per group)
        
        Options:
        - threshold: minimum group-classifier threshold to cluster (larger means fewer higher purity clusters)
        - sum_other (list of ints): list of indices for the 'other' tensors (see inputs) that should be summed by feature instead of just selected
        - loss_scale: loss scale
        - return_neighbours: returns the collected neighbour features as zero-padded V x K x F' rather than mean and max
                             If this is set to True, the network will also return the number of neighbours per vertex
        - ... should be self explanatory
        
        Inputs:
         - features
         - neighbourhood classifier (linear activation applied only)
         - coordinates (V x C): only used for loss evaluation
         - neighbour indices  <- must contain "self"
         - row splits
         
         - +[] of other tensors to be selected or feature-summed (see options) according to clustering
         
         - spectator weights (only if options: use_spectators=True)
         - truth idx (can be dummy if loss is disabled)
         
        
        Outputs:
         - output features (V x 2F) if return_neighbours False, (V x K x F) otherwise
         - only if return_neighbours True: number of neighbours (cast to float32)
         - new row splits
         - backgather indices
         - +[] other tensors with selection applied
         - selected truth index
        
        '''
        self.threshold = threshold
        self.print_reduction = print_reduction
        self.loss_enabled = loss_enabled
        self.loss_scale = loss_scale
        self.distance_loss_scale = distance_loss_scale
        self.print_loss = print_loss
        self.noise_loss_scale = noise_loss_scale
        self.use_spectators = use_spectators
        self.sum_other = sum_other
        self.select_other = select_other
        self.return_neighbours = return_neighbours
        
        assert self.noise_loss_scale >= 0
        
        if 'dynamic' in kwargs:
            super(LNC, self).__init__(**kwargs)
        else:
            super(LNC, self).__init__(dynamic=False,**kwargs)#eager
        
    
    def get_config(self):
        config = {'threshold': self.threshold,
                  'print_reduction': self.print_reduction,
                  'loss_enabled': self.loss_enabled,
                  'loss_scale': self.loss_scale,
                  'distance_loss_scale': self.distance_loss_scale,
                  'print_loss': self.print_loss,
                  'noise_loss_scale': self.noise_loss_scale,
                  'use_spectators': self.use_spectators,
                  'sum_other': self.sum_other,
                  'select_other': self.select_other,
                  'return_neighbours': self.return_neighbours}
        
        base_config = super(LNC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _sel_pass_shape(self, input_shape):
        shapes =  input_shape[5:-1] 
        return [(None, s[1:]) for s in shapes]

    def compute_output_shape(self, input_shapes): #features, nidx = inputs
        K = input_shapes[3][-1]#neighbour indices
        
        directoutshape = [(input_shapes[0][0], input_shapes[0][1]*2)]
        if self.return_neighbours:
            directoutshape = [(input_shapes[0][0], K, input_shapes[0][1]),(None,K)]
        
        if len(input_shapes) > 6:
            return directoutshape+[(None, 1), (None, 1)] + self._sel_pass_shape(input_shapes) + [(None,)]
        else:
            return directoutshape+[(None, 1), (None, 1)] + [(None,)]

    
    #def __compute_output_signature(self, input_signature):
    #    
    #    input_shapes = [x.shape for x in input_signature]
    #    input_dtypes = [x.dtype for x in input_signature]
    #    output_shapes = self.compute_output_shape(input_shapes)
    #
    #    lenin = len(input_signature)
    #    # out, rs, backgather
    #    if lenin > 6:
    #        return  [tf.TensorSpec(dtype=input_dtypes[0], shape=output_shapes[0]), \
    #                tf.TensorSpec(dtype=tf.int32, shape=output_shapes[1]), \
    #                tf.TensorSpec(dtype=tf.int32, shape=output_shapes[2])] + \
    #                [tf.TensorSpec(dtype=input_dtypes[i], shape=output_shapes[i-2]) for i in range(5,lenin)]
    #    else:
    #        return  tf.TensorSpec(dtype=input_dtypes[0], shape=output_shapes[0]), \
    #                tf.TensorSpec(dtype=tf.int32, shape=output_shapes[1]), \
    #                tf.TensorSpec(dtype=tf.int32, shape=output_shapes[2])
            
    

    def build(self, input_shape):
        super(LNC, self).build(input_shape)


    def call(self, inputs):
        
        
        features, score, coords, nidxs, row_splits, other, specweight, tidxs = 8*[None]
        
        if len(inputs) > 6:
            features, score, coords, nidxs, row_splits, *other, tidxs = inputs
        else:
            features, score, coords, nidxs, row_splits, tidxs = inputs
            other=[]
        
        if self.use_spectators:
            specweight = other[-1]
            other = other[0:-1]
            
        for i in self.sum_other:
            if i >= len(other):
                raise ValueError(self.name+" sum_other: indices don't match input")
            
        for i in self.select_other:
            if i >= len(other):
                raise ValueError(self.name+" select_other: indices don't match input")
            
        score = tf.nn.sigmoid(score)
        tnidxs = nidxs #used for truth
        if self.loss_enabled:
            if specweight is not None:
                specweight  = SelectWithDefault(nidxs, specweight, 0.)
                tnidxs = tf.where(specweight[:,:,0]>0, -1 , nidxs) #remove spectators from loss
        #generate loss
        if self.loss_enabled and self.distance_loss_scale > 0:
            
            lossval = tf.zeros_like(score[0,0])
            if row_splits.shape[0] is not None:
            #some headroom for radius
                lossval = self.distance_loss_scale * self.loss_scale * \
                    LLClusterCoordinates.raw_loss([coords, tidxs, row_splits],
                                          repulsion_contrib=0.5, 
                                          print_loss=self.print_loss, 
                                          name=self.name
                                          )
            
            self.add_loss(tf.reduce_mean(lossval))
            
        if self.loss_enabled:
            
            sel_tidxs = SelectWithDefault(tnidxs, tidxs, tf.expand_dims(tidxs,axis=1))
            
            #just in case no "self" index
            same = tf.reduce_min(sel_tidxs,axis=1) == tf.reduce_max(sel_tidxs,axis=1)
            sameobject = tf.where(same, 1.,tf.zeros_like(same,dtype='float32'))
            
            classloss = tf.keras.losses.binary_crossentropy(sameobject, score)

            #this should be kept independent on the relation of objects versus noise in the event

            N_tot    = tf.cast(tf.shape(tidxs)[0],dtype='float32')
            N_noise  = tf.cast(tf.math.count_nonzero(tidxs<0),dtype='float32')
            N_assobj = N_tot-N_noise
            
            nweight = tf.where(tidxs<0, self.noise_loss_scale, tf.zeros_like(score))
            nclassloss = classloss * nweight[:,0]#remove last "1" dimension
            nclassloss = tf.math.divide_no_nan(tf.reduce_sum(nclassloss),N_noise)
            
            oweight = tf.where(tidxs>=0, 1., tf.zeros_like(score))
            oclassloss = classloss * oweight[:,0]
            oclassloss = tf.math.divide_no_nan(tf.reduce_sum(oclassloss),N_assobj)
            
            classloss = self.loss_scale  * (nclassloss+oclassloss)
            if self.print_loss:
                print(self.name, "classifier loss",classloss,"mean score", tf.reduce_mean(score))
            self.add_loss(classloss)
            
            #now make a classifier loss 
            
        # do the reshaping
        
        
        hierarchy_idxs=[]
        if row_splits.shape[0] is None:
            sel = tf.expand_dims(tf.zeros_like(nidxs, dtype='int32'),axis=2)
            ggather = tf.zeros_like(score, dtype='int32')
            npg = tf.zeros_like(score, dtype='int32') + 1
            rs = row_splits
        
        else:
            for i in range(tf.shape(row_splits)[0] - 1):
                a = tf.argsort(score[row_splits[i]:row_splits[i+1]],axis=0, direction='DESCENDING')
                hierarchy_idxs.append(a+row_splits[i])
            hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
            
            #neighbour_idxs, hierarchy_idxs, hierarchy_score,  row_splits,     score_threshold
            
            rs,sel,ggather,npg = LocalGroup(nidxs, hierarchy_idxs, score, row_splits, 
                                        score_threshold=self.threshold)
            
        #keras does not like gather_nd outputs
        #sel = tf.reshape(sel, [-1,K]) #
        #rs = tf.reshape(rs, [-1])
        rs = tf.cast(rs, tf.int32) #just so keras knows
        backgather = tf.reshape(ggather, [-1,1])
        

        #print("sel",sel)
        rs = tf.cast(rs, tf.int32)#just so keras knows
        #be explicit because of keras
        #backgather = tf.cast(backgather, tf.int32)
        seloutshapes =  [[-1,] + list(s.shape[1:]) for s in other] 
        otherout = SelectFromIndices.raw_call(sel[:,0], other, seloutshapes)
        #change those that should be summed
        
        for i in self.sum_other:
            otherout[i] = SelectWithDefault(sel[:,:,0], other[i], 0.)
            otherout[i] = tf.reduce_sum(otherout[i],axis=1)#sum over neighbours
        
        for i in self.select_other:
            otherout[i] = SelectWithDefault(sel[:,0,0], other[i], 0.)
        
        if self.print_reduction:
            tf.print(self.name,'reduction',tf.cast(tf.shape(sel)[0],dtype='float')/tf.cast(tf.shape(nidxs)[0],dtype='float'),'to',tf.shape(sel)[0])
        
        #expanding is done within SelectWithDefault
        #the latter could also be done with AccumulateKnn if it got generalised from V to V'
        out_padded = SelectWithDefault(sel[:,:,0], features, 0.)
        npg = tf.cast(npg,dtype='float32')
        
        sel_tidxs = SelectWithDefault(sel[:,0], tidxs, -1)
        sel_tidxs = tf.squeeze(sel_tidxs, axis=-1)
        
        if self.return_neighbours:
            return [out_padded, npg, rs, backgather] + otherout + [sel_tidxs]
        
        out_max  = SelectWithDefault(sel[:,:,0], features, -1000.)
        
        
        out = tf.concat([tf.reduce_sum(out_padded, axis=1)/(npg+1e-6),  
                         tf.reduce_max(out_max, axis=1)],axis=-1) 
        
        return [out, rs, backgather] + otherout + [sel_tidxs]
        
        
                
class LNC2(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.9, 
                 
                 loss_scale = 1., 
                 distance_loss_scale = 1., 
                 
                 print_reduction=False, 
                 loss_enabled=False, 
                 print_loss=False,
                 
                 assign_empty: int=-10,
                 
                 print_output_shape=False,
                 
                 noise_loss_scale : float = 0.1,
                 
                 return_backscatter=True,
                 **kwargs):
        '''
        Local Neighbour Clustering
        This should be an improvement over LocalClusterReshapeFromNeighbours2 with fewer hyperparameters
        and actually per-point exclusive clustering (not per group)
        
        Options:
        - threshold: minimum group-classifier threshold to cluster (larger means fewer higher purity clusters)
        - loss_scale: loss scale
        - assign_empty: truth index to assign to empty spaces when flattening truth

        
        Inputs are lists themselves:
         IA) A list of
            - neighbourhood classifier (linear activation applied only)
            - distances (V x K): only used for loss evaluation
            - neighbour indices  (V x K) <- must contain "self"
            - row splits
            - spectator weights (ignored/can be None if loss disabled)
            - truth index (ignored/can be None if loss disabled)
            - a flat list of all previous truth indices
            
         IB) A list of all tensors that should be flattened+zero padded (in K) for selected neighbourhoods
         IC) A list of all tensors that should be summed (in K) for selected neighbourhoods
         ID) A list of all tensors that should be mean and maxed (in K) for selected neighbourhoods
         IE) A list of all tensors where the first 'self' index should be selected for selected neighbourshoods
         
         
        
        Outputs are lists:
         OA) A list of
             - The number of neighbours merged
             - new row splits
             - backgather indices
             - an extended flat list of all truth indices, empty space are replaced by <assign_empty>
             
         OB) everything that was in IB processed
         OC) everything that was in IC processed
         OD) everything that was in ID processed
         OE) everything that was in IE processed
         
        
        '''
        self.threshold = threshold
        self.print_reduction = print_reduction
        self.loss_enabled = loss_enabled
        self.loss_scale = loss_scale
        self.distance_loss_scale = distance_loss_scale
        self.print_loss = print_loss
        self.noise_loss_scale = noise_loss_scale
        self.assign_empty = assign_empty
        self.print_output_shape = print_output_shape
        self.return_backscatter = return_backscatter
        
        assert self.noise_loss_scale >= 0
        
        if 'dynamic' in kwargs:
            super(LNC2, self).__init__(**kwargs)
        else:
            super(LNC2, self).__init__(dynamic=False,**kwargs)#eager
        
    
    def get_config(self):
        config = {'threshold': self.threshold,
                  'print_reduction': self.print_reduction,
                  'loss_enabled': self.loss_enabled,
                  'loss_scale': self.loss_scale,
                  'distance_loss_scale': self.distance_loss_scale,
                  'print_loss': self.print_loss,
                  'noise_loss_scale': self.noise_loss_scale,
                  'assign_empty': self.assign_empty,
                  'print_output_shape': self.print_output_shape,
                  'return_backscatter': self.return_backscatter}
        
        base_config = super(LNC2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    def build(self, input_shape):
        super(LNC2, self).build(input_shape)


    def call(self, inputs):
        
        IA, IB, IC, ID, IE = inputs
        
        score, dist, nidxs, row_splits, specweight, tidxs = IA
        score = tf.nn.sigmoid(score)
        
        #build truth loss first
        if self.loss_enabled:
            #distance loss
            if self.distance_loss_scale > 0:  #revised Nov 2021
                lossval = tf.zeros_like(score[0,0])
                nneigh = tf.zeros_like(score)
                if row_splits.shape[0] is not None:
                    lossval = LLLocalClusterCoordinates.raw_loss(dist, nidxs, tidxs, specweight, 
                                                                 print_loss=False, name=self.name)
                lossval = tf.reduce_mean(lossval)
                if self.print_loss:
                    print(self.name, "distance loss",lossval,'average neighbours',tf.reduce_mean(nneigh))
                    
                self.add_loss(self.distance_loss_scale*lossval)
            
            sel_tidxs = SelectWithDefault(nidxs, tidxs, tf.expand_dims(tidxs,axis=1))
            
            #just in case no "self" index
            same = tf.reduce_min(sel_tidxs,axis=1) == tf.reduce_max(sel_tidxs,axis=1)
            sameobject = tf.where(same, 1.,tf.zeros_like(same,dtype='float32'))
            
            classloss = tf.keras.losses.binary_crossentropy(sameobject, score)

            #this should be kept independent on the relation of objects versus noise in the event

            N_tot    = tf.cast(tf.shape(tidxs)[0],dtype='float32')
            N_noise  = tf.cast(tf.math.count_nonzero(tidxs<0),dtype='float32')
            N_assobj = N_tot-N_noise
            
            nweight = tf.where(tidxs<0, self.noise_loss_scale, tf.zeros_like(score))
            nclassloss = classloss * nweight[:,0]#remove last "1" dimension
            nclassloss = tf.math.divide_no_nan(tf.reduce_sum(nclassloss),N_noise)
            
            oweight = tf.where(tidxs>=0, (1.-specweight), tf.zeros_like(score))
            oclassloss = classloss * oweight[:,0]
            oclassloss = tf.math.divide_no_nan(tf.reduce_sum(oclassloss),N_assobj)
            
            classloss = self.loss_scale  * (nclassloss+oclassloss)
            if self.print_loss:
                print(self.name, "classifier loss",classloss,"mean score", tf.reduce_mean(score))
            self.add_loss(classloss)
            
        ################## loss done    
        
        ######## create selectors
        hierarchy_idxs=[]
        if row_splits.shape[0] is None:
            sel = tf.expand_dims(tf.zeros_like(nidxs, dtype='int32'),axis=2)
            ggather = tf.zeros_like(score, dtype='int32')
            npg = tf.zeros_like(score, dtype='int32') + 1
            rs = row_splits
        
        else:
            for i in range(tf.shape(row_splits)[0] - 1):
                a = tf.argsort(score[row_splits[i]:row_splits[i+1]],axis=0, direction='DESCENDING')
                hierarchy_idxs.append(a+row_splits[i])
            hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
            rs,sel,ggather,npg = LocalGroup(nidxs, hierarchy_idxs, score, row_splits, 
                                        score_threshold=self.threshold)
            #print('selmin',tf.reduce_min(sel[:,0]))
            #np.savetxt("test",sel[:,0].numpy())

        rs = tf.cast(rs, tf.int32) #just so keras knows
        backgather = tf.reshape(ggather, [-1,1])
        
        if self.print_reduction:
            tf.print(self.name,'reduction',tf.cast(tf.shape(sel)[0],dtype='float')/tf.cast(tf.shape(nidxs)[0],dtype='float'),'to',tf.shape(sel)[0])
        
        if self.return_backscatter:
            backgather = [tf.shape(score)[0], sel[:,0]]
        #[tf.shape(score)[0], sel]]
        OA = [npg,rs,backgather]
        OB, OC, OD, OE = [], [], [], []
        ######### now work with rs, sel, ggather and npg
        
        #IB, IC, ID, IE
        '''
         IB) A list of all tensors that should be flattened+zero padded (in K) for selected neighbourhoods
         IC) A list of all tensors that should be summed (in K) for selected neighbourhoods
         ID) A list of all tensors that should be mean and maxed (in K) for selected neighbourhoods
         IE) A list of all tensors where the first 'self' index should be selected for selected neighbourshoods
         
        '''
        
        #the stuff below takes a lot of memory. accumulatekNN could be used here, but needs check
        
        for ip in IB: #flattened+zero padded
            toflat = SelectWithDefault(sel[:,:,0], ip, tf.cast(0., ip.dtype))
            toflat = tf.reshape(toflat, [-1, sel.shape[1]*ip.shape[-1]])
            if self.print_output_shape:
                print(self.name,'flat out shape',toflat.shape)
            OB.append(toflat)
        
        for ip in IC: #summed
            ocsel = SelectWithDefault(sel[:,:,0], ip, tf.cast(0., ip.dtype))
            ocsel = tf.reduce_sum(ocsel, axis=1)
            if self.print_output_shape:
                print(self.name,'summed out shape',ocsel.shape)
            OC.append(ocsel)
        
        for ip in ID:
            odsel = SelectWithDefault(sel[:,:,0], ip, tf.cast(0., ip.dtype))
            odsel = tf.reduce_sum(odsel, axis=1)/(tf.cast(npg, ip.dtype)+1e-6)
            odselmax = SelectWithDefault(sel[:,:,0], ip, tf.cast(-1000., ip.dtype))
            odselmax = tf.reduce_max(odselmax, axis=1)
            if self.print_output_shape:
                print(self.name,'meanmax out shape',odsel.shape,odselmax.shape)
            OD.append(tf.concat([odsel,odselmax],axis=-1) )
            
        for ip in IE:
            tosel = SelectWithDefault(sel[:,0:1,0], ip, tf.cast(0., ip.dtype))
            tosel = tf.squeeze(tosel, axis=1)
            if self.print_output_shape:
                print(self.name,'sel out shape',tosel.shape)
            OE.append(tosel)
        
            
        return OA, OB, OC, OD, OE
             
### soft pixel section

class NeighbourGroups(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = None, 
                 purity_min_target = None,
                 efficiency_min_target=None,
                 thresh_viscosity = 1e-2,
                 
                 loss_scale = 1., 
                 
                 print_reduction=False, 
                 loss_enabled=False, 
                 print_loss=False,
                 return_backscatter=True,
                **kwargs):
        '''
        
        Param:
            - threshold: hierarchy discriminator cut off (if None it is automatically adjusted)
            - purity_min_target: minimum purity target in case threshold is automatically adjusted
            - efficiency_min_target: minimum efficiency target in case threshold is automatically adjusted
        
        Inputs:
            - neighbourhood classifier (linear activation applied only)
            - neighbour indices  (V x K) <- must contain "self"
            - row splits
            - truth index (ignored/can be None if loss disabled)
            
        Outputs: 
             - neighbour indices for directed graph accumulation (V x K)
             - selection indices (V' x 1)! (Use with Accumulate/Select layers)
             - backgather/backscatter indices (V x K)
             - new row splits
        '''
        
        
        
        super(NeighbourGroups, self).__init__(**kwargs) 
        self.print_reduction = print_reduction
        self.loss_enabled = loss_enabled
        self.loss_scale = loss_scale
        self.print_loss = print_loss
        self.return_backscatter = return_backscatter
        self.purity_min_target = purity_min_target
        self.efficiency_min_target = efficiency_min_target
        self.thresh_viscosity = thresh_viscosity
        
        if threshold is None:
            assert purity_min_target is not None and efficiency_min_target is not None
            threshold = 0.5
        else:
            assert purity_min_target is None and efficiency_min_target is None #if threshold is not None
        #make this a variable
        with tf.name_scope(self.name + "/1/"):
            self.threshold = tf.Variable(initial_value=threshold, trainable=False,dtype='float32')
        
            
    def get_config(self):
        config = {'purity_min_target': self.purity_min_target,
                  'efficiency_min_target': self.efficiency_min_target,
                  'thresh_viscosity': self.thresh_viscosity,
                  'print_reduction': self.print_reduction,
                  'loss_enabled': self.loss_enabled,
                  'loss_scale': self.loss_scale,
                  'print_loss': self.print_loss,
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
        
        updated_thresh = tf.keras.backend.in_train_phase(update,self.threshold,training=training)
        tf.keras.backend.update(self.threshold,updated_thresh)
    


    def call(self, inputs,  training=None):
        assert len(inputs)==4
        score, nidxs, row_splits, tidxs = inputs
        score = tf.nn.sigmoid(score)
        
        if self.loss_enabled:
            
            sel_tidxs = SelectWithDefault(nidxs, tidxs, -200)
            nntidxs = tf.where(tidxs<0, -20, tidxs)
            
            sameasprobe = tf.cast( tf.reduce_all(nntidxs == sel_tidxs[:,:,0],axis=1,keepdims=True), dtype='float32')#V x 1
            
            true_pos = tf.reduce_sum(tf.where(score>self.threshold, sameasprobe, 0.))
            pos = tf.reduce_sum(tf.where(score>self.threshold, tf.ones_like(sameasprobe), 0.))
            all_true = tf.reduce_sum(sameasprobe)
            
            purity = tf.math.divide_no_nan(true_pos,pos)
            efficiency = tf.math.divide_no_nan(true_pos,all_true)
            
            self._update_threshold(purity, efficiency, training)
            
            classloss = tf.keras.losses.binary_crossentropy(sameasprobe, score)
            classloss = tf.reduce_mean(classloss)
            
            self.add_loss(classloss)
            if self.print_loss and hasattr(efficiency, 'numpy'):
                print(self.name, ' loss ', classloss.numpy(), ', purity', purity.numpy(), 'efficiency', efficiency.numpy(),
                      'threshold', self.threshold.numpy())
            elif self.print_loss:
                tf.print(self.name, ' loss ', classloss, ', purity', purity, 'efficiency', efficiency)
                
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
        
        back = ggather
        if self.return_backscatter:
            back = [tf.shape(nidxs)[0], sel]
            
        dirnidx = tf.reshape(dirnidx, tf.shape(nidxs))#to make shape clear
            
        return dirnidx, sel, back, rs

    def compute_output_shape(self, input_shapes):
        score, nidxs, row_splits, _ = input_shapes
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
        K = tf.cast(tf.reduce_sum(tf.ones_like(ndix[0:1,:])),dtype='float32')
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
            out,_ = AccumulateKnn(zeros,feat, ndix,mean_and_max=False)#*K
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




class SoftPixelRadiusCNN(tf.keras.layers.Layer):
    def __init__(self, length_scale_momentum=0.01, subdivisions: int=3 ,**kwargs):
        """
        Inputs: 
        - features
        - distances squared
        - neighbour_indices
        
        This layer is strongly inspired by 1611.08097 (even though this is only presented for 2D manifolds).
        It implements "soft pixels" for each direction given by the input coordinates plus one central "pixel".
        Each "pixel" is represented by a Gaussian weighting of the inputs.
        For D coordinate dimensions, 2*D + 1 "pixels" are formed (e.g. one for position x direction one for neg x etc).
        The pixels beyond the length scale are removed.
        
        Call will return 
         - soft pixel CNN outputs:  (V x npixels * features)
        
        :param length_scale_momentum: Momentum of the automatically adjusted length scale
                             
        :param subdivisions: number of subdivisions along radius
        
        """
        raise ValueError("SoftPixelRadiusCNN: not implemented yet")
        super(SoftPixelRadiusCNN, self).__init__(**kwargs) 
        assert length_scale_momentum > 0
        assert subdivisions > 1
        
        with tf.name_scope(self.name + "/1/"):
            self.length_scale = tf.Variable(initial_value=1., trainable=False,dtype='float32')

        
    def get_config(self):
        config = {'length_scale_momentum' : self.length_scale_momentum,
                  'subdivisions': self.subdivisions
                  }
        base_config = super(SoftPixelRadiusCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        noutvert = input_shapes[2][0]
        #ncoords = input_shapes[0][1]
        nfeat = input_shapes[1][1]
        return (noutvert, nfeat*self.subdivisions)
    
    
    def build(self, input_shapes):
        super(SoftPixelRadiusCNN, self).build(input_shapes)
        
    def call(self, inputs, training=None):
        
        features, distsq, neighbour_indices = inputs
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
            
        scaler = 10.*self.length_scale*float(self.subdivisions)
        
        for i in range(self.subdivisions):
            offset=float(i)/self.length_scale
            #up to here
            
        #    distancesq = LocalDistance(coordinates+o, neighbour_indices)
        #    f,_ = AccumulateKnn(self.length_scale*distancesq,  features, neighbour_indices) # V x F
        #    f = f[:,:self.nfeat]
        #    f = tf.reshape(f, [-1, self.nfeat])
        #    out.append(f) #only mean
        #out = tf.concat(out,axis=-1)
        #return out

######## generic neighbours

class RaggedGravNet(tf.keras.layers.Layer):
    def __init__(self,
                 n_neighbours: int,
                 n_dimensions: int,
                 n_filters : int,
                 n_propagate : int,
                 return_self=False,
                 sumwnorm=False,
                 feature_activation='relu',
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

        self.n_propagate = n_propagate
        self.n_prop_total = 2 * self.n_propagate

        with tf.name_scope(self.name + "/1/"):
                self.input_feature_transform = tf.keras.layers.Dense(n_propagate, activation=feature_activation)

        with tf.name_scope(self.name + "/2/"):
            self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions,
                                                                 kernel_initializer=EyeInitializer(mean=0, stddev=0.01),
                                                                 use_bias=False)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(self.n_filters, activation='relu')#changed to relu

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform.build(input_shape)

        with tf.name_scope(self.name + "/2/"):
            self.input_spatial_transform.build(input_shape)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform.build((input_shape[0], self.n_prop_total + input_shape[1]))

        super(RaggedGravNet, self).build(input_shape)

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

    def priv_call(self, inputs):
        x = inputs[0]
        row_splits = inputs[1]
        
        coordinates = self.input_spatial_transform(x)
        neighbour_indices, distancesq, sidx, sdist = self.compute_neighbours_and_distancesq(coordinates, row_splits)
        neighbour_indices = tf.reshape(neighbour_indices, [-1, self.n_neighbours-1]) #for proper output shape for keras
        distancesq = tf.reshape(distancesq, [-1, self.n_neighbours-1])

        outfeats = self.create_output_features(x, neighbour_indices, distancesq)
        if self.return_self:
            neighbour_indices, distancesq = sidx, sdist
        return outfeats, coordinates, neighbour_indices, distancesq

    def call(self, inputs):
        return self.priv_call(inputs)

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
              
    

    def compute_neighbours_and_distancesq(self, coordinates, row_splits):
        idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius= -1.0, tf_compatible=False)
        idx = tf.reshape(idx, [-1, self.n_neighbours])
        dist = tf.reshape(dist, [-1, self.n_neighbours])
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
                  'feature_activation': self.feature_activation}
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
        
        
        
        

        
        
        
        
        
