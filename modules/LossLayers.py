import tensorflow as tf
from object_condensation import oc_loss
from oc_helper_ops import SelectWithDefault
from oc_helper_ops import CreateMidx
import time
from baseModules import LayerWithMetrics

def one_hot_encode_id(t_pid, n_classes):
    valued_pids = tf.zeros_like(t_pid)+3 #defaults to 3 as that is the highest showerType value
    valued_pids = tf.where(tf.math.logical_or(t_pid==22, tf.abs(t_pid) == 11), 0, valued_pids) #isEM
    
    valued_pids = tf.where(tf.abs(t_pid)==211, 1, valued_pids) #isHad
    valued_pids = tf.where(tf.abs(t_pid)==2212, 1, valued_pids) #proton isChHad
    valued_pids = tf.where(tf.abs(t_pid)==321, 1, valued_pids) #K+
    
    valued_pids = tf.where(tf.abs(t_pid)==13, 2, valued_pids) #isMIP
    
    valued_pids = tf.where(tf.abs(t_pid)==111, 3, valued_pids) #pi0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==2112, 3, valued_pids) #neutron isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==130, 3, valued_pids) #K0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid)==310, 3, valued_pids) #K0 short
    valued_pids = tf.where(tf.abs(t_pid)==3122, 3, valued_pids) #lambda isNeutrHadOrOther
    
    valued_pids = tf.cast(valued_pids, tf.int32)[:,0]
    depth = n_classes #If n_classes=pred_id.shape[1], should we add an assert statement? 
    return tf.one_hot(valued_pids, depth)

def huber(x, d):
    losssq  = x**2   
    absx = tf.abs(x)                
    losslin = d**2 + 2. * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)

class LossLayerBase(LayerWithMetrics):
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
                 print_batch_time=False,
                 return_lossval=False, 
                 **kwargs):
        super(LossLayerBase, self).__init__(**kwargs)
        
        self.active = active
        self.scale = scale
        self.print_loss = print_loss
        self.print_batch_time = print_batch_time
        self.return_lossval=return_lossval
        with tf.init_scope():
            now = tf.timestamp()
        self.time = tf.Variable(-now, name=self.name+'_time', 
                                trainable=False)
        
    def get_config(self):
        config = {'active': self.active ,
                  'scale': self.scale,
                  'print_loss': self.print_loss,
                  'print_batch_time': self.print_batch_time,
                  'return_lossval': self.return_lossval}
        base_config = super(LossLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        super(LossLayerBase, self).build(input_shape)
    
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
    
    def maybe_print_loss(self,lossval):
        if self.print_loss:
            if hasattr(lossval, 'numpy'):
                print(self.name, 'loss', lossval.numpy())
            else:
                tf.print(self.name, 'loss', lossval)
        
        if self.print_batch_time or self.record_metrics:
            now = tf.timestamp() 
            prev = self.time
            prev = tf.where(prev<0.,now,prev)
            batchtime = now - prev            #round((time.time()-self.time)*1000.)/1000.
            tf.keras.backend.update(self.time,now)
            if self.print_batch_time:
                print(self.name,'batch time',batchtime*1000.,'ms')
            self.add_prompt_metric(batchtime, self.name+'_batch_time')
            

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
    

class LLFillSpace(LossLayerBase):
    def __init__(self, 
                 maxhits: int=1000,
                 runevery: int=-1,
                 **kwargs):
        '''
        calculated a PCA of all points in coordinate space and 
        penalises very asymmetric PCs. 
        Reduces the risk of falling back to a (hyper)surface
        
        Inputs:
         - coordinates, row splits, (truth index - optional. then only applied to non-noise)
        Outputs:
         - coordinates (unchanged)
        '''
        print('INFO: LLFillSpace: this is actually a regulariser: move to right file soon.')
        assert maxhits>0
        self.maxhits = maxhits
        self.runevery = runevery
        self.counter=-1
        if runevery < 0:
            self.counter = -2
        if 'dynamic' in kwargs:
            super(LLFillSpace, self).__init__(**kwargs)
        else:
            super(LLFillSpace, self).__init__(dynamic=True,**kwargs)
        
    
    def get_config(self):
        config={'maxhits': self.maxhits,
                'runevery': self.runevery }
        base_config = super(LLFillSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @staticmethod
    def _rs_loop(coords, tidx, maxhits=1000):
        #only select a few hits to keep memory managable
        nhits = coords.shape[0]
        sel = None
        if nhits > maxhits:
            sel = tf.random.uniform(shape=(maxhits,), minval=0, maxval=coords.shape[0]-1, dtype=tf.int32)
        else:
            sel = tf.range(coords.shape[0], dtype=tf.int32)
        sel = tf.expand_dims(sel,axis=1)
        coords = tf.gather_nd(coords, sel) # V' x C
        if tidx is not None:
            coords = coords[tidx>=0]
        #print('coords',coords.shape)
        means = tf.reduce_mean(coords,axis=0,keepdims=True)#1 x C
        coords -= means # V' x C
        #build covariance
        cov = tf.expand_dims(coords,axis=1)*tf.expand_dims(coords,axis=2)#V' x C x C
        cov = tf.reduce_mean(cov,axis=0,keepdims=False)# 1 x C x C
        #print('cov',cov)
        #get eigenvals
        eigenvals,_ = tf.linalg.eig(cov)#cheap because just once, no need for approx
        eigenvals = tf.cast(eigenvals, dtype='float32')
        #penalise one small EV (e.g. when building a surface)
        pen = tf.math.log((tf.math.divide_no_nan(tf.reduce_mean(eigenvals), 
                                    tf.reduce_min(eigenvals)+1e-6) - 1.)**2+1.)
        return pen
        
        
    @staticmethod
    def raw_loss(coords, rs, tidx, maxhits=1000):
        loss = tf.zeros_like(coords[0,0])
        for i in range(len(rs)-1):
            rscoords = coords[rs[i]:rs[i+1]]
            loss += LLFillSpace._rs_loop(rscoords, tidx, maxhits)
        return tf.math.divide_no_nan(loss ,tf.cast(rs.shape[0],dtype='float32'))
    
    def loss(self, inputs):
        assert len(inputs) == 2 or len(inputs) == 3 #coords, rs
        tidx = None
        if len(inputs) == 3:
            coords, rs, tidx = inputs
        else:
            coords, rs = inputs
        if self.counter >= 0: #completely optimise away increment
            if self.counter < self.runevery:
                self.counter+=1
                return tf.zeros_like(coords[0,0])
            self.counter = 0
        lossval = LLFillSpace.raw_loss(coords, rs, tidx, self.maxhits)
        self.maybe_print_loss(lossval)
        self.add_prompt_metric(lossval, self.name+'_loss')
        
        if self.counter == -1:
            self.counter+=1
        return lossval
    
    
#naming scheme: LL<what the layer is supposed to do>
class LLClusterCoordinates(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    Inputs:
    - coordinates
    - truth index
    - row splits
    '''
    def __init__(self, downsample:int = -1, **kwargs):
        if 'dynamic' in kwargs:
            super(LLClusterCoordinates, self).__init__(**kwargs)
        else:
            super(LLClusterCoordinates, self).__init__(dynamic=True,**kwargs)
            
        self.downsample = downsample
        
        #self.built = True #not necessary for loss layers

    def get_config(self):
        base_config = super(LLClusterCoordinates, self).get_config()
        return dict(list(base_config.items()) + list({'downsample':self.downsample}.items()))
             
    @staticmethod
    def _rs_loop(coords, tidx):
        Msel, M_not, N_per_obj = CreateMidx(tidx, calc_m_not=True) #N_per_obj: K x 1
        if N_per_obj is None:
            return 0.,0.,0. #no objects, discard
        N_per_obj = tf.cast(N_per_obj, dtype='float32')
        N_tot = tf.cast(tidx.shape[0], dtype='float32') 
        K = tf.cast(Msel.shape[0], dtype='float32') 
        
        padmask_m = SelectWithDefault(Msel, tf.ones_like(coords[:,0:1]), 0.)# K x V' x 1
        coords_m = SelectWithDefault(Msel, coords, 0.)# K x V' x C
        #create average
        av_coords_m = tf.reduce_sum(coords_m * padmask_m,axis=1) # K x C
        av_coords_m = tf.math.divide_no_nan(av_coords_m, N_per_obj) #K x C
        av_coords_m = tf.expand_dims(av_coords_m,axis=1) ##K x 1 x C
        
        distloss = tf.reduce_sum((av_coords_m-coords_m)**2,axis=2)
        distloss = tf.math.log(tf.math.exp(1.)*distloss+1.) * padmask_m[:,:,0]
        distloss = tf.math.divide_no_nan(tf.reduce_sum(distloss,axis=1),
                                         N_per_obj[:,0])#K
        distloss = tf.math.divide_no_nan(tf.reduce_sum(distloss),K)
        
        repdist = tf.expand_dims(coords, axis=0) - av_coords_m #K x V x C
        repdist = tf.reduce_sum(repdist**2,axis=-1,keepdims=True) #K x V x 1
        reploss = M_not * tf.exp(-repdist)#K x V x 1
        #downweight noise
        reploss *= tf.expand_dims((1. - 0.9* tf.cast( tidx < 0, dtype='float32' )),axis=0)
        reploss = tf.reduce_sum(reploss,axis=1)/( N_tot-N_per_obj )#K x 1
        reploss = tf.reduce_sum(reploss)/(K+1e-3)
        
        return distloss+reploss, distloss, reploss
    
    @staticmethod
    def raw_loss(acoords, atidx, rs, downsample):
        
        lossval = tf.zeros_like(acoords[0,0])
        reploss = tf.zeros_like(acoords[0,0])
        distloss = tf.zeros_like(acoords[0,0])
        
        nbatches = rs.shape[0]-1
        for i in range(nbatches):
            coords = acoords[rs[i]:rs[i+1]]
            tidx = atidx[rs[i]:rs[i+1]]
            
            if downsample>0 and downsample < coords.shape[0]:
                sel = tf.random.uniform(shape=(downsample,), minval=0, maxval=coords.shape[0]-1, dtype=tf.int32)
                sel = tf.expand_dims(sel,axis=1)
                coords = tf.gather_nd(coords, sel)
                tidx = tf.gather_nd(tidx, sel)
            
            if tidx.shape[0]<20:
                continue #does not make sense
            tlv, tdl, trl = LLClusterCoordinates._rs_loop(coords,tidx)
            lossval += tlv
            distloss += tdl
            reploss += trl
        nbatches = tf.cast(nbatches,dtype='float32')
        return lossval/nbatches, distloss/nbatches, reploss/nbatches

    def maybe_print_loss(self, lossval,distloss, reploss, tidx):
        LossLayerBase.maybe_print_loss(self, lossval)
        #must be eager
        if self.print_loss:
            print(self.name,'attractive',distloss.numpy(),
                  'repulsive',reploss.numpy(),'n_noise',
                  tf.reduce_sum(tf.cast(tidx==-1,dtype='int32')).numpy()
                  )
        
    def loss(self, inputs):
        assert len(inputs) == 3
        coords, tidx, rs = inputs
        lossval,distloss, reploss = LLClusterCoordinates.raw_loss(coords, tidx, rs, self.downsample)
        
        self.add_prompt_metric(lossval, self.name+'_loss')
        self.add_prompt_metric(distloss, self.name+'_att_loss')
        self.add_prompt_metric(reploss, self.name+'_rep_loss')
        
        self.maybe_print_loss(lossval,distloss, reploss, tidx)
        
        return lossval
    


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
         
    def maybe_print_loss(self, lossval):
        pass #overwritten here
         
    def loss(self, inputs):
        dist, nidxs, tidxs, specweight = inputs
        return LLLocalClusterCoordinates.raw_loss(dist, nidxs, tidxs, specweight,
                                                  self.print_loss, 
                                                  self.name)





class LLNotNoiseClassifier(LossLayerBase):
    
    def __init__(self, **kwargs):
        '''
        Inputs:
        - score
        - truth index (ignored if switched off)
        - spectator weights (optional)
        
        Returns:
        - score (unchanged)
        
        '''
        super(LLNotNoiseClassifier, self).__init__(**kwargs)
        
    @staticmethod
    def raw_loss(score, tidx, specweight):
        truth = tf.cast(tidx >= 0,dtype='float32')
        classloss = (1.-specweight[:,0]) * tf.keras.losses.binary_crossentropy(truth, score)
        return tf.reduce_mean(classloss)
        
    def loss(self, inputs):
        assert len(inputs) > 1 and len(inputs) < 4
        score, tidx, specweight = None, None, None
        if len(inputs) == 2:
            score, tidx = inputs
            specweight = tf.zeros_like(score)
        else:
            score, tidx, specweight = inputs
        lossval = LLNotNoiseClassifier.raw_loss(score, tidx, specweight)
        
        isnotnoise = tf.cast(tidx>=0,dtype='float32')
        accuracy = tf.reduce_sum(isnotnoise * tf.cast(score>0.5,dtype='float32'))/tf.reduce_sum(isnotnoise)
        self.add_prompt_metric(accuracy,self.name+'_accuracy')
        self.add_prompt_metric(lossval,self.name+'_loss')
        
        self.maybe_print_loss(lossval)
        return lossval
        

class LLNeighbourhoodClassifier(LossLayerBase):
    def __init__(self, **kwargs):
        '''
        Inputs:
        - score: high means neighbourhood is of same object as first point
        - neighbour indices (ignored if switched off)
        - truth index (ignored if switched off)
        - spectator weights (optional)
        
        Returns:
        - score (unchanged)
        
        '''
        super(LLNeighbourhoodClassifier, self).__init__(**kwargs)
        
    @staticmethod
    def raw_loss(score, nidx, tidxs, specweights):
        # score: V x 1
        # nidx: V x K
        # tidxs: V x 1
        # specweight: V x 1
        
        n_tidxs = SelectWithDefault(nidx, tidxs, -1)[:,:,0] # V x K
        tf.assert_equal(tidxs,n_tidxs[:,0:1])#sanity check to make sure the self reference is in the nidxs
        n_tidxs = tf.where(n_tidxs<0,-10,n_tidxs) #set noise to -10
        
        #the actual check
        n_good = tf.cast(n_tidxs==tidxs, dtype='float32')#noise is always bad
        
        #downweight spectators but don't set them to zero
        n_active = tf.where(nidx>=0, tf.ones_like(nidx,dtype='float32'), 0.) # V x K
        truthscore = tf.math.divide_no_nan(
            tf.reduce_sum(n_good,axis=1,keepdims=True),
            tf.reduce_sum(n_active,axis=1,keepdims=True)) #V x 1
        #cut at 90% same
        truthscore = tf.where(truthscore>0.9,1.,truthscore*0.) #V x 1
        
        lossval = tf.keras.losses.binary_crossentropy(truthscore, score)#V
        
        specweights = specweights[:,0]#V
        isnotnoise = tf.cast(tidxs>=0, dtype='float32')[:,0] #V
        obj_lossval = tf.math.divide_no_nan(tf.reduce_sum(specweights*isnotnoise*lossval) , tf.reduce_sum(specweights*isnotnoise))
        noise_lossval = tf.math.divide_no_nan(tf.reduce_sum((1.-isnotnoise)*lossval) , tf.reduce_sum(1.-isnotnoise))
        
        lossval = obj_lossval + 0.1*noise_lossval #noise doesn't really matter so much
    
        return lossval
        
    def loss(self, inputs):
        assert len(inputs) > 2 and len(inputs) < 5
        score, nidx, tidxs, specweights = None, None, None, None
        if len(inputs) == 3:
            score, nidx, tidxs = inputs
            specweights = tf.ones_like(score)
        else:
            score, nidx, tidxs, specweights = inputs
            
        lossval = LLNeighbourhoodClassifier.raw_loss(score, nidx, tidxs, specweights)
        self.maybe_print_loss(lossval)
        return lossval    


                
class LLEdgeClassifier(LossLayerBase):
    
    def __init__(self, **kwargs):
        '''
        Noise hits (truth index < 0) are never classified as belonging together
        
        Inputs:
        - score
        - neighbour index
        - truth index (ignored if switched off)
        - spectator weights (optional)
        
        Returns:
        - score (unchanged)
        '''
        super(LLEdgeClassifier, self).__init__(**kwargs)
        
    @staticmethod
    def raw_loss(score, nidx, tidx, specweight):
        # nidx = V x K
        # tidx = V x 1
        # specweight: V x 1
        # score: V x K-1 x 1
        n_tidxs = SelectWithDefault(nidx, tidx, -1)# V x K x 1
        tf.assert_equal(tidx,n_tidxs[:,0]) #check that the nidxs have self-reference
        
        #int32 goes up to -2.something e-9
        n_tidxs = tf.where(n_tidxs<0,-1000000000,n_tidxs)#set to -V for noise
        
        n_active = tf.where(nidx>=0, tf.ones_like(nidx,dtype='float32'), 0.)[:,1:] # V x K-1
        specweight = tf.clip_by_value(specweight,0.,1.)
        n_specw = SelectWithDefault(nidx, specweight, -1.)[:,1:,0]# V x K-1
        
        #now this will be false for all noise
        n_sameasprobe = tf.cast(tf.expand_dims(tidx, axis=2) == n_tidxs[:,1:,:], dtype='float32') # V x K-1 x 1
        
        lossval =  tf.keras.losses.binary_crossentropy(n_sameasprobe, score)# V x K-1
        lossval *= n_active
        lossval *= (1.- 0.9*n_specw)#reduce spectators, but don't remove them
        
        lossval = tf.math.divide_no_nan( tf.reduce_sum(lossval,axis=1), tf.reduce_sum(n_active,axis=1) ) # V 
        lossval *= (1.- 0.9*specweight[:,0])#V
        return tf.reduce_mean(lossval)
        
    def loss(self, inputs):
        assert len(inputs) > 2 and len(inputs) < 5
        score, nidx, tidx, specweight = None, None, None, None
        if len(inputs) == 3:
            score, nidx, tidx = inputs
            specweight = tf.zeros_like(score[:,0])
        else:
            score, nidx, tidx, specweight = inputs
        lossval = LLEdgeClassifier.raw_loss(score, nidx, tidx, specweight)
        self.add_prompt_metric(lossval, self.name+'_loss')
        self.maybe_print_loss(lossval)
        return lossval


class LLFullObjectCondensation(LossLayerBase):
    '''
    Cluster using truth index and coordinates
    
    This is a copy of the above, reducing the nested function calls.
    
    keep the individual loss definitions as separate functions, even if they are trivial.
    inherit from this class to implement different variants of the loss ingredients without
    making the config explode (more)
    '''

    def __init__(self, *, energy_loss_weight=1., 
                 use_energy_weights=True, 
                 alt_energy_weight=False,
                 train_energy_correction=True,
                 q_min=0.1, no_beta_norm=False,
                 noise_q_min=None,
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
                 dynamic_payload_scaling_onset=-0.005,
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
        :param dynamic_payload_scaling_onset: only apply payload loss to well reconstructed showers. typical values 0.1 (negative=off)
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
        self.train_energy_correction = train_energy_correction
        self.q_min = q_min
        self.noise_q_min = noise_q_min
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
        self.dynamic_payload_scaling_onset = dynamic_payload_scaling_onset
        self.alt_energy_weight = alt_energy_weight
        self.loc_time=time.time()
        self.call_count=0
        
        assert kalpha_damping_strength >= 0. and kalpha_damping_strength <= 1.

        if standard_configuration is not None:
            raise NotImplemented('Not implemented yet')
        
        
    def calc_energy_weights(self, t_energy, t_pid):
        if self.alt_energy_weight:
            p0=1.49224
            p1=0.000581188
            p2=2.31003
            w = 1./tf.exp( - (p0* tf.math.log(p1*t_energy+1e-6) + p2))/0.161885;#average weight is one
            w = tf.where(tf.abs(t_pid)==13, w+0.05, w)#otherwise muons (with assigned dep weight) are too low
            return w
        lower_cut = 0.5
        w = tf.where(t_energy > 10., 1., ((t_energy-lower_cut) / 10.)*10./(10.-lower_cut))
        return tf.nn.relu(w)
        #flatten energy weights
        
    
    def softclip(self, toclip, startclipat,softness=0.1):
        toclip /= startclipat
        soft = softness*tf.math.log((toclip-(1.-softness))/softness)+1.
        toclip = tf.where(toclip>1, soft , toclip)
        toclip *= startclipat
        return toclip
        
    def calc_energy_correction_factor_loss(self, t_energy, t_idx, hit_energy, pred_energy, row_splits): 
        #
        # all V x 1
        #
        depe=[]
        for b in tf.range(row_splits.shape[0]-1):
            dep = self._calc_and_scatter_dep_energy_pre_rs(
                t_energy[row_splits[b]:row_splits[b + 1]], 
                t_idx[row_splits[b]:row_splits[b + 1]], 
                hit_energy[row_splits[b]:row_splits[b + 1]], 
                pred_energy[row_splits[b]:row_splits[b + 1]]
                )
            depe.append(dep)
        dep_energies = tf.concat(depe,axis=0)
        
        #corrtruth = tf.math.divide_no_nan(t_energy, dep_energies)
        #corrtruth = tf.where(t_idx<0,1.,corrtruth)#make it 1 for noise
        #
        #corrtruth = tf.where(corrtruth>5.,5.,corrtruth)#remove outliers
        #corrtruth = tf.where(corrtruth<.2,.2,corrtruth)
        
        #calo-like
        ediff = (t_energy - pred_energy*dep_energies)/tf.sqrt(t_energy+1e-3)
        
        eloss = None
        if self.huber_energy_scale>0:
            eloss = huber(ediff, self.huber_energy_scale)
        else:
            eloss = tf.math.log(ediff**2 + 1. + 1e-5)
            
        eloss = self.softclip(eloss, 0.4) 
        return eloss
        
    def _calc_and_scatter_dep_energy_pre_rs(self, t_energy, t_idx, hit_energy, pred_energy):   
        
        nt_idx = t_idx + 1 #make noise object
        
        objsel, _, _ = CreateMidx(nt_idx, calc_m_not=False)
        obj_hit_e = SelectWithDefault(objsel, hit_energy, 0.) # K x V-obj x 1
        obj_dep_e = tf.reduce_sum(obj_hit_e,axis=1)#K x 1
        
        _, idxs, _ = tf.unique_with_counts(nt_idx[:,0])#for backgather, same indices as in objsel
        idxs = tf.expand_dims(idxs, axis=1)
        scat_dep_e = tf.gather_nd(obj_dep_e,idxs)
        return scat_dep_e
        
        
            
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
        
        eloss = self.softclip(eloss, 0.2) 
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
    
    def calc_classification_loss(self, t_idx, t_pid, pred_id, row_splits):
        depth = pred_id.shape[1]#add n_classes here?
        truthclass  = one_hot_encode_id(t_pid, depth)
        
        #cpclass=[]
        #for b in tf.range(row_splits.shape[0]-1):
        #    rstidx = t_idx[row_splits[b]:row_splits[b + 1]]
        #    objsel, _, _ = CreateMidx(rstidx, calc_m_not=False)
        #    objsel = objsel[:,0:1] 
        #    tclasses = SelectWithDefault(objsel, truthclass, 0.)#K x 1 x 1
        #    tclasses = tf.
        #    rstidx,uvidx = tf.unique(rstidx)
        #    _, bgidxs, _ = tf.unique_with_counts(rstidx[:,0])
        #    gtclass = truthclass
        #
        #FIXME: add class weights here
        
        return tf.expand_dims(tf.keras.metrics.categorical_crossentropy(truthclass, pred_id), axis=1)

    def loss(self, inputs):
        
        assert len(inputs) == 15
        
        
        pred_beta, pred_ccoords, pred_distscale, pred_energy, pred_pos, pred_time, pred_id,\
        rechit_energy,\
        t_idx, t_energy, t_pos, t_time, t_pid, t_spectator_weights,\
        rowsplits = inputs
                    
        tf.assert_equal(rowsplits[-1], pred_beta.shape[0])#guard
        
        if rowsplits.shape[0] is None:
            return tf.constant(0,dtype='float32')
        
        energy_weights = self.calc_energy_weights(t_energy,t_pid)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights)+1.
            
        
        
        q_min = self.q_min #self.calc_qmin_weight(rechit_energy)#FIXME
            
        #also kill any gradients for zero weight
        energy_loss = None
        if self.train_energy_correction:
            energy_loss = self.energy_loss_weight * self.calc_energy_correction_factor_loss(t_energy, 
                                                                                            t_idx, 
                                                                                            rechit_energy, 
                                                                                            pred_energy, 
                                                                                            rowsplits)
        else:
            energy_loss = self.energy_loss_weight * self.calc_energy_loss(t_energy, pred_energy)
            
        position_loss = self.position_loss_weight * self.calc_position_loss(t_pos, pred_pos)
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(t_time, pred_time)
        classification_loss = self.classification_loss_weight * self.calc_classification_loss(t_idx, t_pid, pred_id, rowsplits)
        
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
                                           noise_q_min = self.noise_q_min,
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
                                           div_repulsion = self.div_repulsion,
                                           dynamic_payload_scaling_onset=self.dynamic_payload_scaling_onset
                                           )

        self.add_prompt_metric(att+rep,self.name+'_dynamic_payload_scaling')
        
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
        
        self.add_prompt_metric(lossval,self.name+'_loss')
        self.add_prompt_metric(att,self.name+'_attractive_loss')
        self.add_prompt_metric(rep,self.name+'_repulsive_loss')
        self.add_prompt_metric(min_b,self.name+'_min_beta_loss')
        self.add_prompt_metric(noise,self.name+'_noise_loss')
        self.add_prompt_metric(energy_loss,self.name+'_energy_loss')
        self.add_prompt_metric(pos_loss,self.name+'_position_loss')
        self.add_prompt_metric(time_loss,self.name+'_time_loss')
        self.add_prompt_metric(class_loss,self.name+'_class_loss')
        self.add_prompt_metric(exceed_beta,self.name+'_exceed_beta_loss')
        
        self.maybe_print_loss(lossval)

        return lossval

    def get_config(self):
        config = {
            'energy_loss_weight': self.energy_loss_weight,
            'alt_energy_weight': self.alt_energy_weight,
            'use_energy_weights': self.use_energy_weights,
            'train_energy_correction': self.train_energy_correction,
            'q_min': self.q_min,
            'no_beta_norm': self.no_beta_norm,
            'potential_scaling': self.potential_scaling,
            'repulsion_scaling': self.repulsion_scaling,
            's_b': self.s_b,
            'noise_q_min': self.noise_q_min,
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
            'div_repulsion' : self.div_repulsion,
            'dynamic_payload_scaling_onset': self.dynamic_payload_scaling_onset
        }
        base_config = super(LLFullObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



        
