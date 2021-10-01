import tensorflow as tf
from select_knn_op import SelectKnn
from select_mod_knn_op import SelectModKnn
from accknn_op import AccumulateKnn
from local_cluster_op import LocalCluster
from local_group_op import LocalGroup
from local_distance_op import LocalDistance
from lossLayers import LLClusterCoordinates
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
import numpy as np
#just for the moment
#### helper###
from datastructures import TrainData_OC,TrainData_NanoML
from initializers import EyeInitializer

from oc_helper_ops import SelectWithDefault

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
    
    return sel, allbackgather, newrs



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
                 **kwargs):
        """
        Inputs are: 
         - Features
         
        Call will return:
         - processed features
        
        will apply some simple fixed preprocessing to the standard TrainData_OC features
        
        """
        self.td=TrainData_NanoML()
        super(ProcessFeatures, self).__init__(**kwargs)
        
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        '''
        'recHitEnergy'
        'recHitEta'   
        'recHitID'    
        'recHitTheta' 
        'recHitR'     
        'recHitX'     
        'recHitY'     
        'recHitZ'     
        'recHitTime'  
        '''
        fdict = self.td.createFeatureDict(inputs, False)
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
        self.gathers=[]
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
        
    
############# Local clustering section ends


class KNN(tf.keras.layers.Layer):
    def __init__(self,K: int, radius: float=-1., **kwargs):
        """
        
        Select K nearest neighbours, with possible radius constraint.
        
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
        


class WarpRegularizer(tf.keras.layers.Layer):
    def __init__(self, strength : float = 0.1, **kwargs):
        super(WarpRegularizer, self).__init__(**kwargs) 
        self.strength = strength
        
    def get_config(self):
        config = {'strength': self.strength}
        base_config = super(WarpRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
     
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, inputs):
        warp = inputs
        diag = tf.expand_dims(tf.eye(warp.shape[-1]), axis=0)
        
        loss = diag*warp - warp #penalise non-diag elements
        loss *= loss
        loss = self.strength * tf.reduce_mean(loss)
        self.add_loss(loss)
        
        print(self.name, loss)
        
        return inputs
    
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
    def __init__(self,K: int, radius: float=-1., **kwargs):
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
        
        
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius}
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
    def raw_call(distances, nidx, K, radius):
        
        tfdist = tf.where(nidx<0, 1e9, distances) #make sure the -1 end up at the end

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
        return SortAndSelectNeighbours.raw_call(distances,nidx,self.K,self.radius)
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
         - backgather indices
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
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'loss_scale': self.loss_scale,
                  'loss_enabled': self.loss_enabled,
                  'print_loss': self.print_loss,
                  'print_reduction': self.print_reduction}
        
        base_config = super(NoiseFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def compute_output_shape(self, input_shapes):
        return input_shapes[1:-1] #all but score itself and truth indices
    
    
    def call(self, inputs):
        
        score, row_splits, *other, tidxs = inputs
        
        if row_splits.shape[0] is None: #dummy execution
            return [row_splits, tf.range(tf.shape(score)[0])] + other
            
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
        
        
        sel, allbackgather, newrs = select_threshold_with_backgather(score, self.threshold, row_splits)
        
        seloutshapes =  [[-1,] + list(s.shape[1:]) for s in other] 
        other = SelectFromIndices.raw_call(sel, other, seloutshapes)
        
        if self.print_reduction:
            print(self.name,' reduction from ', int(row_splits[-1]), ' to ', int(newrs[-1]), ': ', float(row_splits[-1])/float(newrs[-1]))
        
        return [newrs, allbackgather] + other
        
        

class EdgeCreator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
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
    
    def compute_output_shape(self, input_shapes): 
        return (input_shapes[0][-1], input_shapes[1][-1]) # K x F
    
    def call(self, inputs):
        selffeat = tf.expand_dims(inputs[1],axis=1)
        return selffeat - SelectWithDefault(inputs[0][:,1:], inputs[1], 0.)
    

class EdgeSelector(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.9, 
                 loss_scale = 1., 
                 loss_enabled = False,
                 print_loss=False,
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
        if self.loss_enabled:
            sel_tidxs = SelectWithDefault(nidx, tidxs, tf.expand_dims(tidxs,axis=1))
            sel_spec = SelectWithDefault(nidx, specweights, 1.)[:,1:] #don't count -1 entries anyway
            specmask = tf.where(sel_spec[:,:,0]>0, 0., tf.ones_like(score[:,:,0]))
            #mask spectators
            sameasprobe = tf.cast(sel_tidxs[:,0:1] == sel_tidxs,dtype='float32')[:,1:] #remove self ref
            
            edgeloss = specmask * tf.keras.losses.binary_crossentropy(sameasprobe, score)
            edgeloss = self.loss_scale * tf.reduce_mean(edgeloss)
            if self.print_loss:
                print(self.name,'loss ', edgeloss)
            
            self.add_loss( edgeloss )
            
        
        score = tf.concat([tf.ones_like(score[:,0:1,:]),score],axis=1)#add self score always 1
        
        return tf.where(score[:,:,0] < self.threshold, -1, nidx)
        
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
        #take almost mean
        n_neigh = tf.math.count_nonzero(nidx+1, axis=1, keepdims=True)# V x 1
        n_neigh = 1. + (tf.cast(n_neigh,dtype='float32')-1.)*0.5 #give higher score to larger groups
        groupscore = tf.reduce_sum(score, axis=1)*0.5 #make sure max is bound to one
        groupscore /= n_neigh
        return groupscore
        
        
                
class LNC(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold = 0.9, 
                 sum_other=[],
                 
                 loss_scale = 1., 
                 distance_loss_scale = 1., 
                 
                 print_reduction=False, 
                 loss_enabled=False, 
                 print_loss=False,
                 use_spectators=False,
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
         - reshaped features (not 'clustered' yet)
         - new row splits
         - backgather indices
         
         - +[] other tensors with selection applied
        
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
                  'sum_other': self.sum_other}
        
        base_config = super(LNC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _sel_pass_shape(self, input_shape):
        shapes =  input_shape[5:-1] 
        return [(None, s[1:]) for s in shapes]

    def compute_output_shape(self, input_shapes): #features, nidx = inputs
        #K = input_shapes[3][-1]#neighbour indices
        
        if len(input_shapes) > 6:
            return [(input_shapes[0][0], input_shapes[0][1]*2), (None, 1), (None, 1)] + self._sel_pass_shape(input_shapes)
        else:
            return (input_shapes[0][0], input_shapes[0][1]*2), (None, 1), (None, 1)

    
    def __compute_output_signature(self, input_signature):
        
        input_shapes = [x.shape for x in input_signature]
        input_dtypes = [x.dtype for x in input_signature]
        output_shapes = self.compute_output_shape(input_shapes)

        lenin = len(input_signature)
        # out, rs, backgather
        if lenin > 6:
            return  [tf.TensorSpec(dtype=input_dtypes[0], shape=output_shapes[0]), \
                    tf.TensorSpec(dtype=tf.int32, shape=output_shapes[1]), \
                    tf.TensorSpec(dtype=tf.int32, shape=output_shapes[2])] + \
                    [tf.TensorSpec(dtype=input_dtypes[i], shape=output_shapes[i-2]) for i in range(5,lenin)]
        else:
            return  tf.TensorSpec(dtype=input_dtypes[0], shape=output_shapes[0]), \
                    tf.TensorSpec(dtype=tf.int32, shape=output_shapes[1]), \
                    tf.TensorSpec(dtype=tf.int32, shape=output_shapes[2])
            
    

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
        
        
        #expanding is done within SelectWithDefault
        #the latter could also be done with AccumulateKnn if it got generalised from V to V'
        out_mean = SelectWithDefault(sel[:,:,0], features, 0.)
        out_max  = SelectWithDefault(sel[:,:,0], features, -1000.)
        
        npg = tf.cast(npg,dtype='float32')
        
        out = tf.concat([tf.reduce_sum(out_mean, axis=1)/(npg+1e-6),  
                         tf.reduce_max(out_max, axis=1)],axis=-1) 
        
        
        if self.print_reduction:
            tf.print(self.name,'reduction',tf.cast(tf.shape(sel)[0],dtype='float')/tf.cast(tf.shape(nidxs)[0],dtype='float'),'to',tf.shape(sel)[0])
        
        return [out, rs, backgather] + otherout
        
        



### soft pixel section


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
                 **kwargs):
        """
        Call will return output features, coordinates, neighbor indices and squared distances from neighbors

        :param n_neighbours: neighbors to do gravnet pass over
        :param n_dimensions: number of dimensions in spatial transformations
        :param n_filters:  number of dimensions in output feature transformation, could be list if multiple output
        features transformations (minimum 1)

        :param n_propagate: how much to propagate in feature tranformation, could be a list in case of multiple
        :param return_self: for the neighbour indices and distances, switch whether to return the 'self' index and distance (0)
        :param kwargs:
        """
        super(RaggedGravNet, self).__init__(**kwargs)

        n_neighbours += 1  # includes the 'self' vertex
        assert n_neighbours > 1

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.return_self = return_self

        self.n_propagate = n_propagate
        self.n_prop_total = 2 * self.n_propagate

        with tf.name_scope(self.name + "/1/"):
                self.input_feature_transform = tf.keras.layers.Dense(n_propagate, activation='relu')

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

        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices)
        return f

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours,
                  'n_dimensions': self.n_dimensions,
                  'n_filters': self.n_filters,
                  'n_propagate': self.n_propagate,
                  'return_self': self.return_self}
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
                 **kwargs):
        super(DistanceWeightedMessagePassing, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
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
        
        
        
        

        
        
        
        
        
