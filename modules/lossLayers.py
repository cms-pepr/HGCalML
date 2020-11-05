import tensorflow as tf
from object_condensation import oc_loss
from betaLosses import obj_cond_loss, full_obj_cond_loss


class LossLayerBase(tf.keras.layers.Layer):
    """Base class for HGCalML loss layers.
    
    Use the 'active' switch to switch off the loss calculation.
    This needs to be done by hand, and is not handled by the TF 'training' flag, since
    it might be desirable to 
     (a) switch it off during training, or 
     (b) calculate the loss also during inference (e.g. validation)
     
     
    The 'scale' argument determines a global sale factor for the loss. 
    """
    
    def __init__(self, active=True, scale=1., **kwargs):
        super(LossLayerBase, self).__init__(dynamic=True, **kwargs)
        
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



    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


#naming scheme: LL<what the layer is supposed to do>
class LLClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    '''
    def __init__(self, **kwargs):
        super(LLClusterCoordinates, self).__init__(**kwargs)

    def loss(self, inputs):
        x, truth_dict, pred_dict, row_splits = inputs

        coords = pred_dict['predCCoords']
        truth_indices = pred_dict['truthHitAssignementIdx']

        print(x.shape, coords.shape, truth_indices.shape, row_splits.shape)
        zeros = tf.zeros_like(coords[:,0:1])
        
        V_att, V_rep,_,_,_,_=oc_loss(coords, zeros+1./2., #beta constant
                truth_indices, row_splits, 
                zeros, zeros,Q_MIN=1.0, S_B=0.,energyweights=None,
                use_average_cc_pos=True,payload_rel_threshold=0.9)
        
        return V_att+V_rep


class LLObjectCondensation(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    '''

    def __init__(self, energy_loss_weight=1., use_energy_weights=False, q_min=0.5, no_beta_norm=False,
                 potential_scaling=1., repulsion_scaling=1., s_b=1., position_loss_weight=1., timing_loss_weight=1.,
                 use_spectators=True, beta_loss_scale=1., use_average_cc_pos=False, payload_rel_threshold=0.1,
                 rel_energy_mse=False, smooth_rep_loss=False, pre_train=False, huber_energy_scale=2.,
                 downweight_low_energy=True, n_ccoords=2, energy_den_offset=1., noise_scaler=1.,
                 too_much_beta_scale=0.1, cont_beta_loss=False, log_energy=False, standard_configuration=None,
                 **kwargs):
        super(LLObjectCondensation, self).__init__(**kwargs)

        self.energy_loss_weight = energy_loss_weight
        self.use_energy_weights = use_energy_weights
        self.q_min = q_min
        self.no_beta_norm = no_beta_norm
        self.potential_scaling = potential_scaling
        self.repulsion_scaling = repulsion_scaling
        self.s_b = s_b
        self.position_loss_weight = position_loss_weight
        self.timing_loss_weight = timing_loss_weight
        self.use_spectators = use_spectators
        self.beta_loss_scale = beta_loss_scale
        self.use_average_cc_pos = use_average_cc_pos
        self.payload_rel_threshold = payload_rel_threshold
        self.rel_energy_mse = rel_energy_mse
        self.smooth_rep_loss = smooth_rep_loss
        self.pre_train = pre_train
        self.huber_energy_scale = huber_energy_scale
        self.downweight_low_energy = downweight_low_energy
        self.n_ccoords = n_ccoords
        self.energy_den_offset = energy_den_offset
        self.noise_scaler = noise_scaler
        self.too_much_beta_scale = too_much_beta_scale
        self.cont_beta_loss = cont_beta_loss
        self.log_energy = log_energy

        if standard_configuration is not None:
            raise NotImplemented('Not implemented yet')

    def loss(self, inputs):
        x, truth_dict, pred_dict, feat_dict, row_splits = inputs

        config = {
            'energy_loss_weight': self.energy_loss_weight,
            'use_energy_weights': self.use_energy_weights,
            'q_min': self.q_min,
            'no_beta_norm': self.no_beta_norm,
            'potential_scaling': self.potential_scaling,
            'repulsion_scaling': self.repulsion_scaling,
            's_b': self.s_b,
            'position_loss_weight': self.position_loss_weight,
            'timing_loss_weight': self.timing_loss_weight,
            'use_spectators': self.use_spectators,
            'beta_loss_scale': self.beta_loss_scale,
            'use_average_cc_pos': self.use_average_cc_pos,
            'payload_rel_threshold': self.payload_rel_threshold,
            'rel_energy_mse': self.rel_energy_mse,
            'smooth_rep_loss': self.smooth_rep_loss,
            'pre_train': self.pre_train,
            'huber_energy_scale': self.huber_energy_scale,
            'downweight_low_energy': self.downweight_low_energy,
            'n_ccoords': self.n_ccoords,
            'energy_den_offset': self.energy_den_offset,
            'noise_scaler': self.noise_scaler,
            'too_much_beta_scale': self.too_much_beta_scale,
            'cont_beta_loss': self.cont_beta_loss,
            'log_energy': self.log_energy,
        }

        loss = obj_cond_loss(truth_dict, pred_dict, feat_dict, row_splits, config)
        return loss

    def get_config(self):
        config = {
            'energy_loss_weight': self.energy_loss_weight,
            'use_energy_weights': self.use_energy_weights,
            'q_min': self.q_min,
            'no_beta_norm': self.no_beta_norm,
            'potential_scaling': self.potential_scaling,
            'repulsion_scaling': self.repulsion_scaling,
            's_b': self.s_b,
            'position_loss_weight': self.position_loss_weight,
            'timing_loss_weight': self.timing_loss_weight,
            'use_spectators': self.use_spectators,
            'beta_loss_scale': self.beta_loss_scale,
            'use_average_cc_pos': self.use_average_cc_pos,
            'payload_rel_threshold': self.payload_rel_threshold,
            'rel_energy_mse': self.rel_energy_mse,
            'smooth_rep_loss': self.smooth_rep_loss,
            'pre_train': self.pre_train,
            'huber_energy_scale': self.huber_energy_scale,
            'downweight_low_energy': self.downweight_low_energy,
            'n_ccoords': self.n_ccoords,
            'energy_den_offset': self.energy_den_offset,
            'noise_scaler': self.noise_scaler,
            'too_much_beta_scale': self.too_much_beta_scale,
            'cont_beta_loss': self.cont_beta_loss,
            'log_energy': self.log_energy,
        }
        base_config = super(LLObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

