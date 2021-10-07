import tensorflow as tf

'''
don't forget to register as custom objects (e.g. in Layers.py)
'''


class OffDiagonalRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength):
        assert strength>=0
        self.strength = strength
        
    def get_config(self):
        return {'strength': self.strength}
    
    def __call__(self, x):
        diag = tf.eye(x.shape[-2], x.shape[-1])
        offdiag = x * (1.-diag)
        return self.strength * tf.reduce_mean(tf.square(offdiag))


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
    
    

class AverageDistanceRegularizer(tf.keras.layers.Layer):
    def __init__(self,  strength :float =1., printout: bool = False, **kwargs):
        '''
        Penalises if the average distance is not around 0.5
        To make sure a gradient for the GravNet distance weighting exists early on 
        and is most effective in shaping the space. This regulariser should be switched off
        later in the training (set strength to 0 and recompile)
        
        Inputs/outputs: distances (not modified)
        '''
        super(AverageDistanceRegularizer, self).__init__(**kwargs)
        self.strength = strength
        self.printout = printout
        assert strength >= 0
        
    def get_config(self):
        config = {'strength': self.strength,
                  'printout': self.printout
                  }
        base_config = super(AverageDistanceRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
        
    def compute_output_shape(self, input_shapes):
        return input_shapes
    
    def call(self, inputs):
        if self.strength == 0:
            return inputs
        dist = inputs
        dist = tf.sqrt(dist+1e-3)
        Nreal_neigh = tf.cast(tf.math.count_nonzero(inputs,axis=1),dtype='float32')
        avdist = tf.math.divide_no_nan(tf.reduce_sum(dist,axis=1),Nreal_neigh)
        avdist = tf.reduce_mean(avdist)
        loss = self.strength * (avdist-0.5)**2
        if self.printout:
            print(self.name,'average dist',float(avdist),'average neighbours',
                  float(tf.reduce_mean(Nreal_neigh)),
                  'average active neighbours',
                  float(tf.reduce_mean(tf.math.count_nonzero(
                      tf.logical_and(inputs<1.,inputs>0),axis=1))),
                  'penalty',float(loss))
                  
        self.add_loss(loss)
        return inputs
        
