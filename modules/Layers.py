
from caloGraphNN_keras import weighted_sum_layer,GlobalExchange,GravNet,GarNet

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

global_layers_list['GlobalExchange']=GlobalExchange
global_layers_list['GravNet']=GravNet
global_layers_list['GarNet']=GarNet
global_layers_list['weighted_sum_layer']=weighted_sum_layer

from LayersRagged import *

global_layers_list['GridMaxPoolReduction']=GridMaxPoolReduction
global_layers_list['CondensateAndSum']=CondensateAndSum
global_layers_list['RaggedGlobalExchange']=RaggedGlobalExchange
global_layers_list['RaggedConstructTensor']=RaggedConstructTensor
global_layers_list['GraphShapeFilters']=GraphShapeFilters
global_layers_list['GraphFunctionFilters']=GraphFunctionFilters
global_layers_list['VertexScatterer']=VertexScatterer
global_layers_list['RaggedNeighborBuilder']=RaggedNeighborBuilder
global_layers_list['RaggedVertexEater']=RaggedVertexEater
global_layers_list['RaggedNeighborIndices']=RaggedNeighborIndices


global_layers_list['RaggedSelectThreshold']=RaggedSelectThreshold


global_layers_list['FusedRaggedGravNet']=FusedRaggedGravNet
global_layers_list['FusedRaggedGravNet_simple']=FusedRaggedGravNet_simple





from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from Loss_tools import deltaPhi



class ExpMinusOne(Layer):
    def __init__(self, **kwargs):
        super(ExpMinusOne, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.math.expm1(inputs)
    
    
global_layers_list['ExpMinusOne']=ExpMinusOne


class CenterPhi(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, phi_feature_index, **kwargs):
        super(CenterPhi, self).__init__(**kwargs)
        self.phi_feature_index=phi_feature_index
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        phi = inputs[...,self.phi_feature_index:self.phi_feature_index+1]
        
        reference = inputs[...,0:1,self.phi_feature_index:self.phi_feature_index+1]
        
        n_phi = deltaPhi( reference, phi )
        
        rest_left  = inputs[...,:self.phi_feature_index]
        rest_right = inputs[...,self.phi_feature_index+1:]
        
        return tf.concat( [rest_left, n_phi,rest_right ] , axis=-1 )
    
    def get_config(self):
        config = {'phi_feature_index': self.phi_feature_index}
        base_config = super(CenterPhi, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
        

global_layers_list['CenterPhi']=CenterPhi    


class CreateZeroMask(Layer):
    def __init__(self, feature_index, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)
        self.feature_index=feature_index
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)
    
    def call(self, inputs):
        zeros = tf.zeros(shape=tf.shape(inputs)[:-1])
        mask = tf.where(inputs[:,:,self.feature_index]>0, zeros+1., zeros)
        mask = tf.expand_dims(mask,axis=2)
        return mask
    
    def get_config(self):
        config = {'feature_index': self.feature_index}
        base_config = super(CreateZeroMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
        
    

global_layers_list['CreateZeroMask']=CreateZeroMask





class AveragePoolVertices(Layer):
    def __init__(self, keepdims=False, **kwargs):
        super(AveragePoolVertices, self).__init__(**kwargs)
        self.keepdims=keepdims
    
    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return input_shape
        return (input_shape[0],input_shape[2])
    
    def call(self, inputs):
        #create mask
        n_nonmasked = tf.cast(tf.count_nonzero(tf.reduce_sum(inputs, axis=2), axis=-1), dtype='float32')
        n_nonmasked = tf.expand_dims(n_nonmasked, axis=1)
        rsum = tf.reduce_sum(inputs, axis=1)
        out = rsum/(n_nonmasked+K.epsilon())
        if self.keepdims:
            out = tf.expand_dims(out, axis=1)
            out = tf.tile(out, [1, tf.shape(inputs)[1], 1])
        return out
    
    def get_config(self):
        config = {'keepdims': self.keepdims}
        base_config = super(AveragePoolVertices, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
        
    

global_layers_list['AveragePoolVertices']=AveragePoolVertices

   
class TransformCoordinates(Layer):
    def __init__(self, feature_index_x=0, feature_index_y=1, feature_index_z=2, **kwargs):
        super(TransformCoordinates, self).__init__(**kwargs)
        self.feature_index_x=feature_index_x
        self.feature_index_y=feature_index_y
        self.feature_index_z=feature_index_z
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],3)
    
    def call(self, inputs):
        
        xcoord = inputs[:,:,self.feature_index_x:self.feature_index_x+1]
        ycoord = inputs[:,:,self.feature_index_y:self.feature_index_y+1]
        zcoord = inputs[:,:,self.feature_index_z:self.feature_index_z+1]
        	
        #transform to spherical coordinates
        r = tf.math.sqrt( xcoord**2 + ycoord**2 + zcoord**2 +K.epsilon())
        theta = tf.math.acos( zcoord / (r+K.epsilon()) )
        phi = tf.math.atan( ycoord / (xcoord+K.epsilon()) )
        
        ##replace nan values with 0 to deal with divergences
        #-> does not work for the gradient, needs to be fied before		
        thetazeros = tf.zeros(shape=tf.shape(theta))
        phizeros = tf.zeros(shape=tf.shape(phi))    
        theta = tf.where(tf.is_nan(theta), thetazeros, theta)
        phi = tf.where(tf.is_nan(phi), phizeros, phi)	  
        
        #replace cartesian coordinates with spherical ones calculated above    
        #by concatenating tensors along last axis (the features axis)
        transf = tf.concat( [r,theta,phi], -1)
        
        return transf

    
    def get_config(self):
        config = {'feature_index_x': self.feature_index_x, 'feature_index_y': self.feature_index_y, 'feature_index_z': self.feature_index_z}		
        base_config = super(TransformCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))


global_layers_list['TransformCoordinates']=TransformCoordinates   


class SortPredictionByEta(Layer):
    '''
    input: [predicted, input_features]
    arguments: input_energy_index (weight), input_eta_index (variable to be sorted by)
    output: sorted_predicted but 
    '''
    def __init__(self, input_energy_index, input_eta_index , **kwargs):
        super(SortPredictionByEta, self).__init__(**kwargs)
        self.input_energy_index=input_energy_index
        self.input_eta_index=input_eta_index
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def call(self, inputs):
        
        predicted_fracs = inputs[0]
        energies        = inputs[1][:,:,self.input_energy_index:self.input_energy_index+1]
        etas            = inputs[1][:,:,self.input_eta_index:self.input_eta_index+1]
        pred_energies   = energies*predicted_fracs
        pred_sumenergy  = tf.reduce_sum(energies*predicted_fracs, axis=1)
        
        
        #predicted_fracs  : B x V x Fracs
        #energies         : B x V x 1
        #etas             : B x V x 1
        #pred_energies    : B x V x Fracs
        #pred_sumenergy   : B x Fracs

        weighted_etas = tf.reduce_sum(pred_energies * etas, axis=1)/(pred_sumenergy+K.epsilon())
        weighted_etas = tf.where(tf.abs(weighted_etas)>0.1, weighted_etas, tf.zeros_like(weighted_etas)+500.)
        # B x Fracs
        
        ranked_etas, ranked_indices = tf.nn.top_k(-weighted_etas, tf.shape(pred_sumenergy)[1])
        
        ranked_indices = tf.expand_dims(ranked_indices, axis=2)
        
        batch_range = tf.range(0, tf.shape(predicted_fracs)[0])
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        
        batch_indices = tf.tile(batch_range, [1, tf.shape(predicted_fracs)[2], 1]) # B x Fracs x 1
        indices = tf.concat([batch_indices, ranked_indices], axis=-1) # B x Fracs x 2
        
        identity_matrix = tf.eye(tf.shape(pred_sumenergy)[1]) #Fracs x Fracs 1 matrix
        identity_matrix = tf.expand_dims(identity_matrix, axis=0) # 1 x F x F
        identity_matrix = tf.tile(identity_matrix, [tf.shape(pred_sumenergy)[0],1,1])  # B x F x F
        sorted_identity_matrix = tf.gather_nd(identity_matrix, indices) # B x F x F
        
        # sorted_identity_matrix : B x Fm x Fm
        # predicted_fracs        : B x V  x Ff
        
        # B x Fm x Fm --> B x V x Fm x Fm
        sorted_identity_matrix = tf.expand_dims(sorted_identity_matrix, axis=1)
        sorted_identity_matrix = tf.tile(sorted_identity_matrix, [1,tf.shape(predicted_fracs)[1],1,1])
        # B x V x Fm x Fm
        
        # predicted_fracs   : B x V  x Ff --> B x V x Ff x 1
        sorted_predicted_fractions = tf.expand_dims(predicted_fracs, axis=3)
        
        return tf.squeeze(tf.matmul(sorted_identity_matrix, sorted_predicted_fractions), axis=-1)
        
    
    def get_config(self):
        config = {'input_energy_index': self.input_energy_index,
                  'input_eta_index': self.input_eta_index}
        base_config = super(SortPredictionByEta, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
        
    


class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])
    
    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)
        
    
    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange 