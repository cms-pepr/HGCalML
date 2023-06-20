
import tensorflow as tf

class GroupSort(tf.keras.layers.Layer): 
    
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, input):
        
        return tf.sort(input, axis=-1)
        
        
class Sphere(tf.keras.layers.Layer): 
    
    def compute_output_shape(self, input_shapes):
        out = []
        for s in input_shapes:
            out.append(s)
        out[-1] += 1
        return out
    
    def call(self, x):
        norm = tf.reduce_sum(x**2, axis=-1,keepdims=True)
        norm = tf.sqrt(norm+1e-6)
        x = tf.concat([x / norm, norm], axis=-1)
        return input
        