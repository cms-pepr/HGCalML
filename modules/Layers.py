
from caloGraphNN_keras import weighted_sum_layer,GlobalExchange,GravNet,GarNet
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

global_layers_list['GlobalExchange']=GlobalExchange
global_layers_list['GravNet']=GravNet
global_layers_list['GarNet']=GarNet
global_layers_list['weighted_sum_layer']=weighted_sum_layer



from keras.layers import Layer
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
   