
from caloGraphNN_keras import weighted_sum_layer,GlobalExchange,GravNet,GarNet

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

global_layers_list['GlobalExchange']=GlobalExchange
global_layers_list['GravNet']=GravNet
global_layers_list['GarNet']=GarNet
global_layers_list['weighted_sum_layer']=weighted_sum_layer



from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class CreateZeroMask(Layer):
    def __init__(self, feature_index, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)
        self.feature_index=feature_index
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)
    
    def call(self, inputs):
        zeros = tf.zeros(shape=tf.shape(inputs)[:-1])
        mask = tf.where(inputs[:,:,0]>0, zeros+1., zeros)
        mask = tf.expand_dims(mask,axis=2)
        return mask
    
    def get_config(self):
        config = {'feature_index': self.feature_index}
        base_config = super(CreateZeroMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
        
    

global_layers_list['CreateZeroMask']=CreateZeroMask
   
   
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
        weighted_etas = tf.where(tf.abs(weighted_etas)>0.1, etas, tf.zeros_like(weighted_etas)+500.)
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
        
    