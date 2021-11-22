import tensorflow as tf
from object_condensation import oc_loss
from betaLosses import obj_cond_loss
from oc_helper_ops import SelectWithDefault
import time


def huber(x, d):
    losssq  = x**2   
    absx = tf.abs(x)                
    losslin = d**2 + 2. * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)

class LossLayerBase(tf.keras.layers.Layer):
    """Base class for HGCalML loss layers.
    
    Use the 'active' switch to switch off the loss calculation.
    This needs to be done by hand, and is not handled by the TF 'training' flag, since
    it might be desirable to 
     (a) switch it off during training, or 
     (b) calculate the loss also during inference (e.g. validation)
     
     
    The 'scale' argument determines a global sale factor for the loss. 
    """
    
    def __init__(self, active=True, scale=1., 
                 print_loss=False,
                 return_lossval=False, **kwargs):
        super(LossLayerBase, self).__init__(**kwargs)
        
        self.active = active
        self.scale = scale
        self.print_loss = print_loss
        self.return_lossval=return_lossval
        
    def get_config(self):
        config = {'active': self.active ,
                  'scale': self.scale,
                  'print_loss': self.print_loss,
                  'return_lossval': self.return_lossval}
        base_config = super(LossLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        lossval = tf.constant([0.],dtype='float32')
        if self.active:
            lossval = self.scale * self.loss(inputs)
            if not self.return_lossval:
                self.add_loss(lossval)
        if self.return_lossval:
            return inputs[0], lossval
        else:
            return inputs[0]
    
    def loss(self, inputs):
        '''
        Overwrite this function in derived classes.
        Input: always a list of inputs, the first entry in the list will be returned, and should be the features.
        The rest is free (but will probably contain the truth somewhere)
        '''
        return tf.constant(0.,dtype='float32')

    def compute_output_shape(self, input_shapes):
        if self.return_lossval:
            return input_shapes[0], (None,)
        else:
            return input_shapes[0]



class CreateTruthSpectatorWeights(tf.keras.layers.Layer):
    def __init__(self, 
                 threshold, 
                 minimum, 
                 active,
                 **kwargs):
        '''
        active: does not enable a loss, but acts similar to other layers using truth information
                      as a switch to not require truth information at all anymore (for inference)
                      
        Inputs: spectator score, truth indices
        Outputs: spectator weights (1-minimum above threshold, 0 else)
        
        '''
        super(CreateTruthSpectatorWeights, self).__init__(**kwargs)
        self.threshold = threshold
        self.minimum = minimum
        self.active = active
        
    def get_config(self):
        config = {'threshold': self.threshold,
                  'minimum': self.minimum,
                  'active': self.active,
                  }
        base_config = super(CreateTruthSpectatorWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes   
    
    def call(self, inputs):
        if not self.active:
            return inputs[0]
        
        abovethresh = inputs[0] > self.threshold
        #notnoise = inputs[1] >= 0
        #noise can never be spectator
        return tf.where(abovethresh, tf.ones_like(inputs[0])-self.minimum, 0.)
    

#naming scheme: LL<what the layer is supposed to do>
class LLClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    '''
    def __init__(self, repulsion_contrib=0.5, **kwargs):
        self.repulsion_contrib=repulsion_contrib
        assert repulsion_contrib <= 1. and repulsion_contrib>= 0.
        
        if 'dynamic' in kwargs:
            super(LLClusterCoordinates, self).__init__(**kwargs)
        else:
            super(LLClusterCoordinates, self).__init__(dynamic=True,**kwargs)

    @staticmethod
    def raw_loss(inputs, repulsion_contrib, print_loss, name):
        
        use_avg_cc=1.
        coords, truth_indices, row_splits, beta_like = None, None, None, None
        if len(inputs) == 3:
            coords, truth_indices, row_splits = inputs
        elif len(inputs) == 4:
            coords, truth_indices, beta_like, row_splits = inputs
            use_avg_cc = 0.
        else:
            raise ValueError("LLClusterCoordinates requires 3 or 4 inputs")

        zeros = tf.zeros_like(coords[:,0:1])
        if beta_like is None:
            beta_like = zeros+1./2.
        else:
            beta_like = tf.nn.sigmoid(beta_like)
            beta_like = tf.stop_gradient(beta_like)#just informing, no grad # 0 - 1
            beta_like = 0.1* beta_like + 0.5 #just a slight scaling
        
        #this takes care of noise through truth_indices < 0
        V_att, V_rep,_,_,_,_=oc_loss(coords, beta_like, #beta constant
                truth_indices, row_splits, 
                zeros, zeros,Q_MIN=1.0, S_B=0.,energyweights=None,
                use_average_cc_pos=use_avg_cc)
        
        att = (1.-repulsion_contrib)*V_att
        rep = repulsion_contrib*V_rep
        lossval = att + rep
        if print_loss:
            print(name, lossval.numpy(), 'att loss:', att.numpy(), 'rep loss:',rep.numpy())
        return lossval

    def loss(self, inputs):
        return LLClusterCoordinates.raw_loss(inputs, self.repulsion_contrib, self.print_loss, self.name)
        
    def get_config(self):
        config = { 'repulsion_contrib': self.repulsion_contrib }
        base_config = super(LLClusterCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LLNoiseClassifier(LossLayerBase):
    def __init__(self, **kwargs):
        super(LLNoiseClassifier, self).__init__(**kwargs)
        
    def loss(self, inputs):
        score, tidxs = inputs
        truth = tf.cast(tidxs == -1,dtype='float32')
        classloss = tf.keras.losses.binary_crossentropy(truth, score)
        return tf.reduce_mean(classloss)

class LLLocalClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    Inputs: dist, nidxs, tidxs, specweight
    
    Attractive and repulsive potential:
    - Att: log(sqrt(2)*dist**2+1.)
    - Rep: 1/(dist**2+0.1)
    
    The crossing point is at about 1,1.
    The ratio of repulse and attractive potential at 
     - dist**2 = 1. is about 1
     - dist = 0.85, dist**2 = 0.75 is about 2.
     - dist = 0.7, dist**2 = 0.5 is about 3.5
     - dist = 0.5, dist**2 = 0.25 is about 10
    (might be useful for subsequent distance cut-offs)
    
    '''
    def __init__(self, **kwargs):
        
        super(LLLocalClusterCoordinates, self).__init__(**kwargs)
        #if 'dynamic' in kwargs:
        #    super(LLLocalClusterCoordinates, self).__init__(**kwargs)
        #else:
        #    super(LLLocalClusterCoordinates, self).__init__(dynamic=kwargs['print_loss'],**kwargs)
        self.time = time.time()

    @staticmethod
    def raw_loss(dist, nidxs, tidxs, specweight, print_loss, name):
        
        sel_tidxs = SelectWithDefault(nidxs, tidxs, -1)[:,:,0]
        sel_spec = SelectWithDefault(nidxs, specweight, 1.)[:,:,0]
        active = tf.where(nidxs>=0, tf.ones_like(dist), 0.)
        notspecmask = 1. #(1. - 0.5*sel_spec)#only reduce spec #tf.where(sel_spec>0, 0., tf.ones_like(dist))
        
        probe_is_notnoise = tf.cast(tidxs>=0,dtype='float32') [:,0] #V
        notnoisemask = tf.where(sel_tidxs<0, 0., tf.ones_like(dist))
        notnoiseweight = notnoisemask + (1.-notnoisemask)*0.01
        #notspecmask *= notnoisemask#noise can never be spec
        #mask spectators
        sameasprobe = tf.cast(sel_tidxs[:,0:1] == sel_tidxs,dtype='float32')
        #sameasprobe *= notnoisemask #always push away noise, also from each other
        
        #only not noise can be attractive
        attmask = sameasprobe*notspecmask*active
        repmask = (1.-sameasprobe)*notspecmask*active
        
        attr = tf.math.log(tf.math.exp(1.)*dist+1.) * attmask
        rep =  tf.exp(-dist)* repmask * notnoiseweight # 1./(dist+1.) * repmask #2.*tf.exp(-3.16*tf.sqrt(dist+1e-6)) * repmask  #1./(dist+0.1)
        nattneigh = tf.reduce_sum(attmask,axis=1)
        nrepneigh = tf.reduce_sum(repmask,axis=1)
        
        attloss =  probe_is_notnoise * tf.reduce_sum(attr,axis=1) #tf.math.divide_no_nan(tf.reduce_sum(attr,axis=1), nattneigh)#same is always 0
        attloss = tf.math.divide_no_nan(attloss, nattneigh)
        reploss =  probe_is_notnoise * tf.reduce_sum(rep,axis=1) #tf.math.divide_no_nan(tf.reduce_sum(rep,axis=1), nrepneigh)
        reploss = tf.math.divide_no_nan(reploss, nrepneigh)
        #noise does not actively contribute
        lossval = attloss+reploss
        lossval = tf.math.divide_no_nan(tf.reduce_sum(probe_is_notnoise * lossval),tf.reduce_sum(probe_is_notnoise))
        
        if print_loss:
            avattdist = probe_is_notnoise * tf.math.divide_no_nan(tf.reduce_sum(attmask*tf.sqrt(dist),axis=1), nattneigh)
            avattdist = tf.reduce_sum(avattdist)/tf.reduce_sum(probe_is_notnoise)
            
            avrepdist = probe_is_notnoise * tf.math.divide_no_nan(tf.reduce_sum(repmask*tf.sqrt(dist),axis=1), nrepneigh)
            avrepdist = tf.reduce_sum(avrepdist)/tf.reduce_sum(probe_is_notnoise)
            
            if hasattr(lossval, "numpy"):
                print(name, 'loss', lossval.numpy(),
                      'mean att neigh',tf.reduce_mean(nattneigh).numpy(),
                      'mean rep neigh',tf.reduce_mean(nrepneigh).numpy(),
                      'att', tf.reduce_mean(probe_is_notnoise *attloss).numpy(),
                      'rep',tf.reduce_mean(probe_is_notnoise *reploss).numpy(),
                      'dist (same)', avattdist.numpy(),
                      'dist (other)', avrepdist.numpy(),
                      )
            else:
                tf.print(name, 'loss', lossval,
                'mean att neigh',tf.reduce_mean(nattneigh),
                'mean rep neigh',tf.reduce_mean(nrepneigh))
            
                
        return lossval            
         
         
    def loss(self, inputs):
        dist, nidxs, tidxs, specweight = inputs
        if self.print_loss:
            if self.time>0:
                print(round((time.time()-self.time)*1000.),'ms')
                self.time = time.time()
        return LLLocalClusterCoordinates.raw_loss(dist, nidxs, tidxs, specweight,
                                                  self.print_loss, 
                                                  self.name)




### LLSelectedDistances
# slim
# takes neighbour indices and distances, and (truth) ass idx
# make indices TF compat by 
#   overwrite = tile( tf range)
#   tf.where(idx<0, overwrite, idx)
# gather asso idx
# select asso idx (no rs necessary already taken care of by kNN)
# repulsive/attractive using distances (will be 0 for self index)
#


class LLFullObjectCondensation(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    
    This is a copy of the above, reducing the nested function calls.
    
    keep the individual loss definitions as separate functions, even if they are trivial.
    inherit from this class to implement different variants of the loss ingredients without
    making the config explode (more)
    '''

    def __init__(self, *, energy_loss_weight=1., use_energy_weights=True, q_min=0.1, no_beta_norm=False,
                 potential_scaling=1., repulsion_scaling=1., s_b=1., position_loss_weight=1.,
                 classification_loss_weight=1., timing_loss_weight=1., use_spectators=True, beta_loss_scale=1.,
                 use_average_cc_pos=0.,
                  payload_rel_threshold=0.1, rel_energy_mse=False, smooth_rep_loss=False,
                 pre_train=False, huber_energy_scale=-1., downweight_low_energy=True, n_ccoords=2, energy_den_offset=1.,
                 noise_scaler=1., too_much_beta_scale=0., cont_beta_loss=False, log_energy=False, n_classes=0,
                 prob_repulsion=False,
                 phase_transition=0.,
                 phase_transition_double_weight=False,
                 alt_potential_norm=True,
                 print_time=True,
                 payload_beta_gradient_damping_strength=0.,
                 payload_beta_clip=0.,
                 kalpha_damping_strength=0.,
                 cc_damping_strength=0.,
                 standard_configuration=None,
                 beta_gradient_damping=0.,
                 alt_energy_loss=False,
                 repulsion_q_min=-1.,
                 super_repulsion=False,
                 use_local_distances=True,
                 energy_weighted_qmin=False,
                 super_attraction=False,
                 div_repulsion=False,
                 **kwargs):
        """
        Read carefully before changing parameters

        :param energy_loss_weight:
        :param use_energy_weights:
        :param q_min:
        :param no_beta_norm:
        :param potential_scaling:
        :param repulsion_scaling:
        :param s_b:
        :param position_loss_weight:
        :param classification_loss_weight:
        :param timing_loss_weight:
        :param use_spectators:
        :param beta_loss_scale:
        :param use_average_cc_pos: weight (between 0 and 1) of the average position vs. the kalpha position 
        :param payload_rel_threshold:
        :param rel_energy_mse:
        :param smooth_rep_loss:
        :param pre_train:
        :param huber_energy_scale:
        :param downweight_low_energy:
        :param n_ccoords:
        :param energy_den_offset:
        :param noise_scaler:
        :param too_much_beta_scale:
        :param cont_beta_loss:
        :param log_energy:
        :param n_classes: give the real number of classes, in the truth labelling, class 0 is always ignored so if you
                          have 6 classes, label them from 1 to 6 not 0 to 5. If n_classes is 0, no classification loss
                          is applied
        :param prob_repulsion
        :param phase_transition
        :param standard_configuration:
        :param alt_energy_loss: introduces energy loss with very mild gradient for large delta. (modified 1-exp form)
        :param kwargs:
        """
        if 'dynamic' in kwargs:
            super(LLFullObjectCondensation, self).__init__(**kwargs)
        else:
            super(LLFullObjectCondensation, self).__init__(dynamic=True,**kwargs)
            
        assert use_local_distances #fixed now, if they should not be used, pass 1s
        
        if too_much_beta_scale==0 and cont_beta_loss:
            raise ValueError("cont_beta_loss must be used with too_much_beta_scale>0")
        
        if huber_energy_scale>0 and alt_energy_loss:
            raise ValueError("huber_energy_scale>0 and alt_energy_loss exclude each other")

        self.energy_loss_weight = energy_loss_weight
        self.use_energy_weights = use_energy_weights
        self.q_min = q_min
        self.no_beta_norm = no_beta_norm
        self.potential_scaling = potential_scaling
        self.repulsion_scaling = repulsion_scaling
        self.s_b = s_b
        self.position_loss_weight = position_loss_weight
        self.classification_loss_weight = classification_loss_weight
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
        self.n_classes = n_classes
        self.prob_repulsion = prob_repulsion
        self.phase_transition = phase_transition
        self.phase_transition_double_weight = phase_transition_double_weight
        self.alt_potential_norm = alt_potential_norm
        self.print_time = print_time
        self.payload_beta_gradient_damping_strength = payload_beta_gradient_damping_strength
        self.payload_beta_clip = payload_beta_clip
        self.kalpha_damping_strength = kalpha_damping_strength
        self.cc_damping_strength = cc_damping_strength
        self.beta_gradient_damping=beta_gradient_damping
        self.alt_energy_loss=alt_energy_loss
        self.repulsion_q_min=repulsion_q_min
        self.super_repulsion=super_repulsion
        self.use_local_distances = use_local_distances
        self.energy_weighted_qmin=energy_weighted_qmin
        self.super_attraction = super_attraction
        self.div_repulsion=div_repulsion
        
        self.loc_time=time.time()
        self.call_count=0
        
        assert kalpha_damping_strength >= 0. and kalpha_damping_strength <= 1.

        if standard_configuration is not None:
            raise NotImplemented('Not implemented yet')
        
        
    def calc_energy_weights(self, t_energy):
        lower_cut = 0.5
        w = tf.where(t_energy > 10., 1., ((t_energy-lower_cut) / 10.)*10./(10.-lower_cut))
        return tf.nn.relu(w)
    
    def softclip(self, toclip, startclipat):
        toclip /= startclipat
        toclip = tf.where(toclip>1, tf.math.log(toclip+1.), toclip)
        toclip *= startclipat
        return toclip
        
            
    def calc_energy_loss(self, t_energy, pred_energy): 
        
        #FIXME: this is just for debugging
        #return (t_energy-pred_energy)**2
        eloss=0
        
        if self.huber_energy_scale > 0:
            l = tf.abs(t_energy-pred_energy)
            sqrt_t_e = tf.sqrt(t_energy+1e-3)
            l = tf.math.divide_no_nan(l, tf.sqrt(t_energy+1e-3) + self.energy_den_offset)
            eloss = huber(l, sqrt_t_e*self.huber_energy_scale)
        elif self.alt_energy_loss:
            ediff = (t_energy-pred_energy)
            l = tf.math.log(ediff**2/(t_energy+1e-3) + 1.)
            eloss = l
        else:
            eloss = tf.math.divide_no_nan((t_energy-pred_energy)**2,(t_energy + self.energy_den_offset))
        
        eloss = self.softclip(eloss, 10.) 
        return eloss

    def calc_qmin_weight(self, hitenergy):
        if not self.energy_weighted_qmin:
            return self.q_min
        
    
    def calc_position_loss(self, t_pos, pred_pos):
        if not self.position_loss_weight:
            t_pos = 0.
        if tf.shape(t_pos)[-1] == 3:#also has z component, but don't use it here
            t_pos = t_pos[:,0:2]
        #reduce risk of NaNs
        ploss = huber(tf.sqrt(tf.reduce_sum((t_pos-pred_pos) ** 2, axis=-1, keepdims=True)/(10**2) + 1e-2), 10.) #is in cm
        return self.softclip(ploss, 3.) 
    
    def calc_timing_loss(self, t_time, pred_time):
        if not self.timing_loss_weight:
            return pred_time**2
        
        tloss = huber((t_time - pred_time),2.) 
        return self.softclip(tloss, 6.) 
    
    def calc_classification_loss(self, t_pid, pred_id):
        '''
        to be implemented, t_pid is not one-hot encoded
        '''
        return 1e-8*tf.reduce_mean(pred_id**2,axis=-1,keepdims=True) #V x 1
    
    def loss(self, inputs):
        
        start_time = 0
        if self.print_time:
            start_time = time.time()
        
        pred_distscale=None
        rechit_energy=None
        t_spectator_weights=None
        if self.use_local_distances:
            if self.energy_weighted_qmin:
                raise ValueError("energy_weighted_qmin not implemented")

            else:
                #check for sepctator weights
                if len(inputs) == 14:
                    pred_beta, pred_ccoords, pred_distscale, pred_energy, pred_pos, pred_time, pred_id,\
                    t_idx, t_energy, t_pos, t_time, t_pid, t_spectator_weights,\
                    rowsplits = inputs
                elif len(inputs) == 13:
                    pred_beta, pred_ccoords, pred_distscale, pred_energy, pred_pos, pred_time, pred_id,\
                    t_idx, t_energy, t_pos, t_time, t_pid,\
                    rowsplits = inputs
                
        else:
            if len(inputs) == 13:
                pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id,\
                t_idx, t_energy, t_pos, t_time, t_pid,t_spectator_weights,\
                rowsplits = inputs
            elif len(inputs) == 12:
                pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id,\
                t_idx, t_energy, t_pos, t_time, t_pid,\
                rowsplits = inputs


        if rowsplits.shape[0] is None:
            return tf.constant(0,dtype='float32')
        
        energy_weights = self.calc_energy_weights(t_energy)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights)+1.
            
        
        
        q_min = self.q_min #self.calc_qmin_weight(rechit_energy)#FIXME
            
        #also kill any gradients for zero weight
        energy_loss = self.energy_loss_weight * self.calc_energy_loss(t_energy, pred_energy)
        position_loss = self.position_loss_weight * self.calc_position_loss(t_pos, pred_pos)
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(t_time, pred_time)
        classification_loss = self.classification_loss_weight * self.calc_classification_loss(t_pid, pred_id)
        
        full_payload = tf.concat([energy_loss,position_loss,timing_loss,classification_loss], axis=-1)
        
        if self.payload_beta_clip > 0:
            full_payload = tf.where(pred_beta<self.payload_beta_clip, 0., full_payload)
            #clip not weight, so there is no gradient to push below threshold!
        
        is_spectator = t_spectator_weights #not used right now, and likely never again (if the truth remains ok)
        if is_spectator is None:
            is_spectator = tf.zeros_like(pred_beta)
        
        att, rep, noise, min_b, payload, exceed_beta = oc_loss(
                                           x=pred_ccoords,
                                           beta=pred_beta,
                                           truth_indices=t_idx,
                                           row_splits=rowsplits,
                                           is_spectator=is_spectator,
                                           payload_loss=full_payload,
                                           Q_MIN=q_min,
                                           S_B=self.s_b,
                                           distance_scale=pred_distscale,
                                           energyweights=energy_weights,
                                           use_average_cc_pos=self.use_average_cc_pos,
                                           payload_rel_threshold=self.payload_rel_threshold,
                                           cont_beta_loss=self.cont_beta_loss,
                                           prob_repulsion=self.prob_repulsion,
                                           phase_transition=self.phase_transition>0. ,
                                           phase_transition_double_weight = self.phase_transition_double_weight,
                                           #removed
                                           #alt_potential_norm=self.alt_potential_norm,
                                           payload_beta_gradient_damping_strength=self.payload_beta_gradient_damping_strength,
                                           kalpha_damping_strength = self.kalpha_damping_strength,
                                           beta_gradient_damping=self.beta_gradient_damping,
                                           repulsion_q_min=self.repulsion_q_min,
                                           super_repulsion=self.super_repulsion,
                                           super_attraction = self.super_attraction,
                                           div_repulsion = self.div_repulsion
                                           )

        
        att *= self.potential_scaling
        rep *= self.potential_scaling * self.repulsion_scaling
        min_b *= self.beta_loss_scale
        noise *= self.noise_scaler
        exceed_beta *= self.too_much_beta_scale

        #unscaled should be well in range < 1.
        att = self.softclip(att, self.potential_scaling) 
        rep = self.softclip(rep, self.potential_scaling * self.repulsion_scaling) 
        #min_b = self.softclip(min_b, 5.)  # not needed, limited anyway
        #noise = self.softclip(noise, 5.)  # not needed limited to 1 anyway
        
        
        energy_loss = payload[0]
        pos_loss    = payload[1]
        time_loss   = payload[2]
        class_loss  = payload[3]
        
        
        #explicit cc damping
        ccdamp = self.cc_damping_strength * (0.02*tf.reduce_mean(pred_ccoords))**4# gently keep them around 0
        
        
        lossval = att + rep + min_b + noise + energy_loss + pos_loss + time_loss + class_loss + exceed_beta + ccdamp
            
        lossval = tf.reduce_mean(lossval)
        
        #loss should be <1 pretty quickly in most cases; avoid very hard hits from high LRs shooting to the moon
        
        
        if self.print_time:
            print('loss layer',self.name,'took',int((time.time()-start_time)*100000.)/100.,'ms',' call ',self.call_count)
            print('loss layer info:',self.name,'batch took',int((time.time()-self.loc_time)*100000.)/100.,'ms',
                  'for',len(rowsplits.numpy())-1,'batch element(s), and total ', pred_beta.shape[0], 'points')
            self.loc_time = time.time()
            self.call_count+=1
            
        if self.print_loss:
            minbtext = 'min_beta_loss'
            if self.phase_transition>0:
                minbtext = 'phase transition loss'
            print('avg beta', tf.reduce_mean(pred_beta))
            print('loss', lossval.numpy(),
                  'attractive_loss', att.numpy(),
                  'rep_loss', rep.numpy(),
                  minbtext, min_b.numpy(),
                  'noise_loss', noise.numpy(),
                  'energy_loss', energy_loss.numpy(),
                  'pos_loss', pos_loss.numpy(),
                  'time_loss', time_loss.numpy(),
                  'class_loss', class_loss.numpy(),
                  'exceed_beta',exceed_beta.numpy(),
                  'ccdamp', ccdamp.numpy(),'\n')

        return lossval

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
            'classification_loss_weight' : self.classification_loss_weight,
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
            'n_classes': self.n_classes,
            'prob_repulsion': self.prob_repulsion,
            'phase_transition': self.phase_transition,
            'phase_transition_double_weight': self.phase_transition_double_weight,
            'alt_potential_norm': self.alt_potential_norm,
            'print_time' : self.print_time,
            'payload_beta_gradient_damping_strength': self.payload_beta_gradient_damping_strength,
            'payload_beta_clip' : self.payload_beta_clip,
            'kalpha_damping_strength' : self.kalpha_damping_strength,
            'cc_damping_strength' : self.cc_damping_strength,
            'beta_gradient_damping': self.beta_gradient_damping,
            'repulsion_q_min': self.repulsion_q_min,
            'super_repulsion': self.super_repulsion,
            'use_local_distances': self.use_local_distances,
            'energy_weighted_qmin': self.energy_weighted_qmin,
            'super_attraction':self.super_attraction,
            'div_repulsion' : self.div_repulsion
        }
        base_config = super(LLFullObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



        