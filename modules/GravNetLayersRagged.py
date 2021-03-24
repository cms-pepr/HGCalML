import tensorflow as tf
from select_knn_op import SelectKnn
from accknn_op import AccumulateKnn
from local_cluster_op import LocalCluster

from local_distance_op import LocalDistance
from lossLayers import LLLocalClusterCoordinates
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
import numpy as np
#just for the moment
#### helper###
from datastructures import TrainData_OC,TrainData_NanoML

def check_type_return_shape(s):
    if not isinstance(s, tf.TensorSpec):
        raise TypeError('Only TensorSpec signature types are supported, '
                      'but saw signature entry: {}.'.format(s))
    return s.shape


############# Some layers for convenience ############


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
        self.td=TrainData_OC()
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
        fdict['recHitTime'] = tf.nn.relu(fdict['recHitTime'])/10. #remove -1 default
        allf = []
        for k in fdict:
            allf.append(fdict[k])
        return tf.concat(allf,axis=-1)
    
    
class NeighbourCovariance(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        """
        Inputs: 
          - coordinates (Vin x C)
          - features (Vin x F)
          - neighbour indices (Vout x K)
          
        Returns concatenated  (Vout x { F*(C*(C+1)/2 + F*C})
          - feature weighted covariance matrices (lower triangle) (Vout x F*(C*(C+1)/2)
          - feature weighted means (Vout x F*C)
          
        THIS OPERATION HAS NO GRADIENT SO FAR!!
        """
        super(NeighbourCovariance, self).__init__(**kwargs)
        self.outshapes=None
    
    
    def build(self, input_shapes): #pure python
        super(NeighbourCovariance, self).build(input_shapes)
        
    @staticmethod
    def raw_call(coordinates, features, n_idxs):
        nF = features.shape[1]
        nC = coordinates.shape[1]
        nVout = -1 #n_idxs.shape[0]
        cov, means = NeighbourCovarianceOp(coordinates=coordinates, 
                                         features=features, 
                                         n_idxs=n_idxs)
        # Vout x F x C(C+1)/2 , Vout x F x C
        covshape = int(nF*(nC*(nC+1)//2))
        cov = tf.reshape(cov, [nVout, covshape])
        means = tf.reshape(means, [nVout, nF*nC])
        
        return cov, means
    
    def call(self, inputs):
        coordinates, features, n_idxs = inputs
        cov,means = NeighbourCovariance.raw_call(coordinates, features, n_idxs)
        return tf.concat([cov,means],axis=-1)
        
    
class LocalDistanceScaling (tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        """
        Inputs: 
        - distances (V x N)
        - scaling (V x 1)
        
        Returns:
        distances * scaling : V x N x 1
        scaling is bound to be within 0 and 2.
        """
        super(LocalDistanceScaling, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[1]
    
    @staticmethod
    def raw_call(dist,scale):
        scale = tf.nn.softsign(scale)+1
        return dist*scale
    
    def call(self, inputs):
        dist,scale = inputs
        return LocalDistanceScaling.raw_call(dist,scale)
    
############# PCA like section

class NeighbourPCA   (tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        """
        This layer performas a simple exact PCA* of the inputs fiven by:
        
        - the features (V x F)
        - the neighbour indices to consider
        
        The output shape is the F^2
        
        Too resource intense for full application
        
        """
        super(NeighbourPCA, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0][:-1]+input_shapes[0][-1]**2
    
    @staticmethod
    def raw_call(nidx, feat):
        nfeat = tf.gather_nd(feat, tf.expand_dims(nidx,axis=2))
        s,_,v = tf.linalg.svd(nfeat)  # V x F, V x F x F
        s = tf.expand_dims(s, axis=2)  # V x F x 1 
        s = tf.sqrt(s + 1e-6)
        scaled = s*v  # V x F x F
        return tf.reshape(scaled, [-1,feat.shape[1]**2])
        
    def call(self, inputs):
        feat, neighs = inputs
        return NeighbourPCA.raw_call(neighs,feat)

############# Local clustering section


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
    
    
    def compute_output_signature(self, input_signature):
        
        input_shapes = [x.shape for x in input_spec]
        output_shapes = self.compute_output_shape(input_shapes)

        return [tf.TensorSpec(dtype=tf.int32, shape=output_shapes[i]) for i in range(len(output_shape))]
    
   
    def build(self, input_shapes):
        super(LocalClustering, self).build(input_shapes)
        
    @staticmethod
    def raw_call(neighs, hier, row_splits,print_reduction,name):
        if row_splits.shape[0] is None:
            return tf.zeros_like(hier, dtype='int32'), row_splits, tf.zeros_like(hier, dtype='int32')
        
        if hier.shape[1] > 1:
            raise ValueError(name+' received wrong hierarchy shape')
        
        hierarchy_idxs=[]
        for i in range(row_splits.shape[0] - 1):
            a = tf.argsort(hier[row_splits[i]:row_splits[i+1]],axis=0, direction='DESCENDING')
            hierarchy_idxs.append(a+row_splits[i])
        hierarchy_idxs = tf.concat(hierarchy_idxs,axis=0)
        
        
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
    def raw_call(indices, inputs, outshapes):
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
        


class SortAndSelectNeighbours(tf.keras.layers.Layer):
    def __init__(self,K: int, radius: float, **kwargs):
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
    def raw_call(features, nidx):
        nidx = tf.cast(nidx,tf.int32)#just because of keras build
        TFnidx = tf.expand_dims(tf.where(nidx<0,0,nidx),axis=2)
        
        gfeat = tf.gather_nd(features, TFnidx)
        gfeat = tf.reshape(gfeat, [-1, tf.shape(nidx)[1], tf.shape(features)[1]])#same shape
        
        out = tf.where(tf.expand_dims(nidx,axis=2)<0, 0., gfeat)
        out = tf.reshape(out, [-1, tf.shape(features)[1]*tf.shape(nidx)[1]])
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
        '''
        
        This layer is a simple, but handy combination of: 
        - SortAndSelectNeighbours
        - LocalClustering
        - SelectFromIndices
        - GraphClusterReshape
        
        and comes with it's own loss layer (LLLocalClusterCoordinates) to implement
        a gradient on all operations and inputs.
        
        Inputs:
         - features
         - distances
         - hierarchy feature
         - neighbour indices
         - row splits
         - +[] of other tensors to be selected according to clustering
         
         - truth idx (can be dummy if loss is disabled
         
        
        When applied, the layer graph reshapes a max. selected number of neighbours (K),
        within a maximum radius. At the same time, other features of the new found cluster
        centres can be selected in one go.
        
        The included loss function implements a gradient on the input distances, and the hierarchy feature
        following a similar approach as object condensation (but applied only locally to selected neighbours).
        Those vertices within the same neighbourhood (given by neighbour indices) that have the same truth
        index are pulled together while others are being pushed away. The corresponding loss per vertex is scaled
        by the hierarchy index (which receives a penalty to be >0 at the same time) such that it can be interpreted
        as a confidence measure that the corresponding grouping is valid.
        
        As a consequence, this layer will aim to reshape the graph in a way that vertices from the same object form
        groups.
         
        Outputs:
         - reshaped features
         - new row splits
         - backgather indices
         - +[] other tensors with selection applied
        
        '''
        self.K = K
        self.radius = radius
        self.print_reduction = print_reduction
        self.loss_enabled = loss_enabled
        self.loss_scale = loss_scale
        self.loss_repulsion = loss_repulsion
        self.print_loss = print_loss
        
        if 'dynamic' in kwargs:
            super(LocalClusterReshapeFromNeighbours, self).__init__(**kwargs)
        else:
            super(LocalClusterReshapeFromNeighbours, self).__init__(dynamic=False,**kwargs)
        
    
    def get_config(self):
        config = {'K': self.K,
                  'radius': self.radius,
                  'print_reduction': self.print_reduction,
                  'loss_enabled': self.loss_enabled,
                  'loss_scale': self.loss_scale,
                  'loss_repulsion': self.loss_repulsion,
                  'print_loss': self.print_loss}
        
        base_config = super(LocalClusterReshapeFromNeighbours, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _sel_pass_shape(self, input_shape):
        shapes =  input_shape[5:-1] 
        return [(None, s[1:]) for s in shapes]

    def compute_output_shape(self, input_shapes): #features, nidx = inputs
        K = self.K
        if K < 0:
            K = input_shapes[3][-1]#no neighbour selection
        
        
        if len(input_shapes) > 6:
            return [(input_shapes[0][0], input_shapes[0][1]*K), (None, 1), (None, 1)] + self._sel_pass_shape(input_shapes)
        else:
            return (input_shapes[0][0], input_shapes[0][1]*K), (None, 1), (None, 1)

    
    def compute_output_signature(self, input_signature):
        
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
        super(LocalClusterReshapeFromNeighbours, self).build(input_shape)


    def call(self, inputs):
        features, distances, hierarchy, nidxs, row_splits, other, tidxs = 7*[None]
        
        if len(inputs) > 6:
            features, distances, hierarchy, nidxs, row_splits, *other, tidxs = inputs
        else:
            features, distances, hierarchy, nidxs, row_splits, tidxs = inputs
            other=[]
            
        sdist, snidx = distances,nidxs #  
        #generate loss
        if self.loss_enabled:
            #some headroom for radius
            lossval = self.loss_scale * LLLocalClusterCoordinates.raw_loss(
                sdist/(self.radius**2), hierarchy, snidx, tidxs,  #or distances/(1.5*self.radius)**2
                add_self_reference=False, repulsion_contrib=self.loss_repulsion,
                print_loss=self.print_loss,name=self.name)
            self.add_loss(lossval)
        # do the reshaping
        sdist, snidx = SortAndSelectNeighbours.raw_call(sdist, snidx,K=self.K, radius=self.radius)
        
        sel, rs, backgather = LocalClustering.raw_call(snidx,hierarchy,row_splits,
                                                       print_reduction=self.print_reduction,name=self.name)
        
        rs = tf.cast(rs, tf.int32)#just so keras knows
        #be explicit because of keras
        #backgather = tf.cast(backgather, tf.int32)
        seloutshapes =  [[-1,] + list(s.shape[1:]) for s in [snidx, sdist]+other] 
        snidx, sdist, *other = SelectFromIndices.raw_call(sel, [snidx, sdist]+other, seloutshapes)
        
        out = GraphClusterReshape()([features,snidx])
        
        return [out, rs, backgather] + other
        
        


### soft pixel section


class SoftPixelCNN(tf.keras.layers.Layer):
    def __init__(self, length_scale_momentum=0.01, mode: str='onlyaxes', subdivisions: int=3 ,**kwargs):
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
            f,_ = AccumulateKnn(self.length_scale*distancesq,  features, neighbour_indices, n_moments=0) # V x F
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
            self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions,use_bias=False,kernel_initializer=tf.keras.initializers.Orthogonal())

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(self.n_filters, activation='tanh')

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
        return (input_shapes[0][0], 2*self.n_filters),\
               (input_shapes[0][0], self.n_dimensions),\
               (input_shapes[0][0], self.n_neighbours-1),\
               (input_shapes[0][0], self.n_neighbours-1)
              
    

    def compute_neighbours_and_distancesq(self, coordinates, row_splits):
        idx,dist = SelectKnn(self.n_neighbours, coordinates,  row_splits,
                             max_radius= -1.0, tf_compatible=False)
        if self.return_self:
            return idx[:, 1:], dist[:, 1:], idx, dist
        return idx[:, 1:], dist[:, 1:], None, None


    def collect_neighbours(self, features, neighbour_indices, distancesq):

        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices, n_moments=0)
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
        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices, n_moments=0)
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
        f,_ = AccumulateKnn(tf.cast(idxs*0, tf.float32),  x, idxs, n_moments=0)
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
        f,_ = AccumulateKnn(tf.cast(neighbour_indices*0, tf.float32),  features, neighbour_indices, n_moments=0)
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

        # weights = gauss_of_lin(10. * distancesq)
        # weights = tf.expand_dims(weights, axis=-1)  # [SV, N, 1]
        # neighbour_features = tf.gather_nd(features, neighbour_indices)
        # neighbour_features *= weights
        # neighbours_max = tf.reduce_max(neighbour_features, axis=1)
        # neighbours_mean = tf.reduce_mean(neighbour_features, axis=1)
        #
        f,_ = AccumulateKnn(10.*distancesq,  features, neighbour_indices, n_moments=0)
        return f


        return tf.concat([neighbours_max, neighbours_mean], axis=-1)



    def call(self, inputs):
        x, neighbor_indices, distancesq = inputs
        return self.create_output_features(x, neighbor_indices, distancesq)


    def get_config(self):
        config = {'n_feature_transformation': self.n_feature_transformation,
                  }
        base_config = super(DistanceWeightedMessagePassing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class WeightedCovariances(tf.keras.layers.Layer):
    '''
    allows distances after each passing operation to be dynamically adjusted.
    this similar to FusedRaggedGravNetAggAtt, but incorporates the scaling in the message passing loop
    '''

    def __init__(self, **kwargs):
        super(WeightedCovariances, self).__init__(**kwargs)


    def build(self, input_shapes):
        super(WeightedCovariances, self).build(input_shapes)


    def collect_neighbours(self, features, neighbour_indices):
        neighbour_features = tf.gather_nd(features, neighbour_indices[..., tf.newaxis])
        return neighbour_features

    def call(self, inputs):
        x, coords, neighbor_indices = inputs
        coords_collected = self.collect_neighbours(coords, neighbor_indices) # [V, N, F]

        mean_est = coords[:, tf.newaxis, :]#just the central point #tf.reduce_mean(coords_collected, axis=1)[:, tf.newaxis, :]
        centered = coords_collected - mean_est #[V,N,F]
        centered_transposed = tf.transpose(centered, perm=[0, 2, 1]) # [V,F,N]
        # [V,F,N]x[V,N,F]
        cov_est = tf.linalg.matmul(centered_transposed, centered) / (tf.cast(tf.shape(coords_collected)[1], tf.float32) - 1.) # [V,F, F]

        weighted = cov_est[:, tf.newaxis,:,:] * x[:,:,tf.newaxis, tf.newaxis]
        weighted_flattened = tf.reshape(weighted, [tf.shape(coords_collected)[0], (coords.shape[1]*coords.shape[1]*x.shape[1])])

        return weighted_flattened
