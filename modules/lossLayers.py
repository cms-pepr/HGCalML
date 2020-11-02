import tensorflow as tf
from object_condensation import oc_loss


class LossLayerBase(tf.keras.layers.Layer):
    """Base class for HGCalML loss layers.
    
    Use the 'active' switch to switch off the loss calculation.
    This needs to be done by hand, and is not handled by the TF 'training' flag, since
    it might be desirable to 
     (a) switch it off during training, or 
     (b) calculate the loss also during inference (e.g. validation)
     
     
    The 'scale' argument determines a global sale factor for the loss. 
    """
    
    def __init__(self, active=True, scaler=1.**kwargs):
        super(LossLayerBase, self).__init__(**kwargs)
        
        self.active = active
        self.scale = scale
    
    def call(self, inputs):
        if self.active:
            self.add_loss(self.scale * self.loss(inputs))
        return inputs[0]
    
    def loss(self, inputs):
        '''
        Overwrite this function in derived classes.
        Input: always a list of inputs, the first entry in the list will be returned, and should be the features.
        The rest is free (but will probably contain the truth somewhere)
        '''
        return 0.


#naming scheme: LL<what the layer is supposed to do>
class LLClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    '''
    def __init__(self, **kwargs):
        super(LLClusterCoordinates, self).__init__(**kwargs)
    
    def loss(self, inputs):
        x, coords, truth, row_splits = inputs
        
        truth_indices = ... #FIXME, depends on the truth input
        
        zeros = tf.zeros_like(coords[:,0:1])
        
        V_att, V_rep,_,_,_,_=oc_loss(coords, zeros+1./2., #beta constant
                truth_indices, row_splits, 
                zeros, zeros,Q_MIN=1.0, S_B=0.,energyweights=None,
                use_average_cc_pos=True,payload_rel_threshold=0.9)
        
        return V_att+V_rep