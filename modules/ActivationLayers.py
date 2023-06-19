
import tensorflow as tf

class GroupSort(tf.keras.layers.Layer): 
    
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, input):
        
        return tf.sort(input, axis=-1)
        