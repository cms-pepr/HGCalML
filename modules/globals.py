
'''
file reserved for global constants used by multiple modules

(from a c++ perspective:)
Use classes as 'namespace' and implement through
getters and setters so that const-ness is ensured

As these are namespace equivalentes, Class capitalisation
does not apply here.

See the example of the cluster space class, which can be accessed by

from globals import cluster_space

> print(cluster_space.noise_coord) 
100.
> cluster_space.noise_coord=10.
TypeError

'''
####### pure helper classes, no need to touch

class _metaconst(type):
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise TypeError(key+' is constant.')
    
class _const(object, metaclass=_metaconst):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        raise TypeError(name+' is constant.')


####### actual constants here, need to inherit from _const

class cluster_space(_const):
    '''
    coordinates used for noise that was removed when it is
    assigned a place in clustering space when the hits are 
    scattered back.
    '''
    noise_coord = 100.
    

class hit_keys(_const):
    '''
    keys for features/pred/truth per hit that cannot be passed transparently
    as they need to be used in functions.
    These should be used at any occurence to avoid issues when they're changed.
    
    Another good (maybe better) place for them would be TrainData_NanoML.
    Also, there use the inheritance from _const.
    
    So far, this is not used (yet)
    '''
    rec_energy = 'recHitEnergy'
    # ...
    
class pu(_const):
    '''
    special constants associated to pile up to make
    'standard-style' particle-in-pu plots. Don't overuse
    these, we don't really want to really distinguish between
    PU and 'main' event at this stage of reconstruction.
    '''
    
    '''
    The hgcal has 2x3M sensors, so there will be max 6M truth showers
    in one event, so 10M is a good offset, and well within int32 range.
    Still, if used, please always add a safety check before such as e.g.:
      if np.max(t_idx) >= pu.t_idx_offset:
          raise ValueError(...)
    '''
    t_idx_offset = 1e7


'''
The only non-const global.
In case TF gradients should be used instead of custom OP gradients.
This will increase resource usage, and is mostly a panic switch 
to make sure weird behaviour is not caused by the (well tested) custom gradients

This needs to be imported and changed before any other import of ops
'''
    
knn_ops_use_tf_gradients=False   
acc_ops_use_tf_gradients=False   

