
from caloGraphNN_keras import weighted_sum_layer,GlobalExchange,GravNet,GarNet
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

global_layers_list['GlobalExchange']=GlobalExchange
global_layers_list['GravNet']=GravNet
global_layers_list['GarNet']=GarNet
global_layers_list['weighted_sum_layer']=weighted_sum_layer



from keras.layers import Layer
import tensorflow as tf

class MaskZeros(Layer):
    def __init__(self, **kwargs):
        super(MaskZeros, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return inputs
        
    
    def get_config(self):
        base_config = super(MaskZeros, self).get_config()
        return dict(list(base_config.items()))
    

global_layers_list['MaskZeros']=MaskZeros
   