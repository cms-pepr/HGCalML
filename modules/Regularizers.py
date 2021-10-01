import tensorflow as tf

'''
don't forget to register as custom objects (e.g. in Layers.py)
'''


class OffDiagonalRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength
        
    def get_config(self):
        return {'strength': self.strength}
    
    def __call__(self, x):
        diag = tf.eye(x.shape[-2], x.shape[-1])
        offdiag = x * (1.-diag)
        return self.strength * tf.reduce_mean(tf.square(offdiag))
