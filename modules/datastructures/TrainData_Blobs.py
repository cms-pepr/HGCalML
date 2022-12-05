import matplotlib.pyplot as plt
from DeepJetCore import TrainData, SimpleArray
import numpy as np


from datastructures.TrainData_NanoML import n_id_classes, TrainData_NanoML

class TrainData_Blobs(TrainData_NanoML):
    def __init__(self):
        TrainData.__init__(self)
        
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        
        from sklearn.datasets import make_blobs
        import os
        #this assumes that the 'input files' are <number>_xyz
        
        filename = os.path.basename(filename)
        
        seed = int(filename.split('_')[0])+1
        np.random.seed(seed)
        
        
        
        '''
        self.featurenames = [
            'recHitEnergy', 0
            'recHitEta',
            'isTrack',
            'recHitTheta',
            'recHitR',
            'recHitX', 5
            'recHitY', 6
            'recHitZ', 7
            'recHitTime',
            'recHitHitR' 9
            ]
        '''
        
        def make_features(xy):
            e = np.zeros((xy.shape[0],1),dtype='float32') + 1.
            eta = e*0. + 1
            eta_to_r = np.zeros((xy.shape[0],3),dtype='float32')
            z = np.zeros((xy.shape[0],1),dtype='float32') + 2.
            last = np.zeros((xy.shape[0],2),dtype='float32')
            return np.concatenate([e,eta,eta_to_r, xy, z, last],axis=-1)
        
        def make_truth(f,tidx):
            maxtidx = np.max(tidx)
            
            t_energy = f[:,0:1]*0.
            
            for t in range(maxtidx+1):
                sel = tidx == t
                nhits = np.sum( tidx[sel]*0. + 1. )
                t_energy[sel] = nhits/10.
            
            
            
            t_pid = np.zeros((f.shape[0],n_id_classes),dtype='float32')
            t_pid[:,-1] = 1 #just put it to one class
            
            return t_energy, t_pid
            
            
        
        def make_event(seed):
            
            nblobs = np.random.randint(2,7, size=1)[0]
            #print('nblobs',nblobs)
            nhits = np.random.randint(1800,1900, size=(nblobs))
            #print('nhits',nhits)
            
            #add some noise, like 10 hits
            fnoise = np.random.rand(10,2)*10.
            tnoise = np.zeros((fnoise.shape[0]),dtype='int32')-1
            
            f, tidx = make_blobs(n_samples=nhits, n_features=2,
                   random_state=seed)
            
            f = np.concatenate([fnoise,f],axis=0)
            tidx = np.concatenate([tnoise,tidx],axis=0)
            
            f = make_features(f)
            
            t_energy, t_pid = make_truth(f,tidx)
            
            return f, tidx, t_energy, t_pid 
        
        rs = [0]
        f_all = []
        t_idx_all = []
        t_energy_all = []
        t_pid_all = []
        
        
        for _ in range(1000):#1k events per file
            f, tidx, t_energy, t_pid = make_event(seed)
            
            f_all.append(f)
            t_idx_all.append(tidx)
            t_energy_all.append(t_energy)
            t_pid_all.append(t_pid)
            
            rs.append(f.shape[0]+rs[-1])
            
            seed += 1000
            
        rs = np.array(rs, dtype='int64')    
            
        f_all = np.array(np.concatenate(f_all,axis=0).astype('float32'))
        t_idx_all =  np.array(np.concatenate(t_idx_all,axis=0).astype('int32')[..., np.newaxis])
        t_energy_all = np.array(np.concatenate(t_energy_all,axis=0).astype('float32'))
        t_pid_all = np.array(np.concatenate(t_pid_all,axis=0).astype('float32'))
            
        zeros = np.zeros_like(t_energy_all)   
        
        print(f_all.shape, t_idx_all.shape, t_energy_all.shape, t_pid_all.shape,rs) 
        
        out = [f_all, 
               t_idx_all, 
               t_energy_all, 
                np.concatenate([zeros,zeros,zeros],axis=-1),
                zeros,
                t_pid_all,
                zeros,
                zeros+1,
                t_energy_all,
                zeros
                ]
        
        out = [SimpleArray(o, rs) for o in out]
        
        return out,[],[]
    #[farr, 
    #            t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
    #            t['t_pid'], t['t_spectator'], t['t_fully_contained'],
    #            t['t_rec_energy'], t['t_is_unique'] ],[], []
                
                
                
                