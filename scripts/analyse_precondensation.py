#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from DeepJetCore.modeltools import load_model
from DeepJetCore import DataCollection
from LossLayers import LossLayerBase
import pandas as pd
import plotly.express as px
from datastructures import TrainData_NanoML
from tqdm import tqdm
import threading

class shower(object):
    
    def __init__(self, hitdict :dict, t_energy, t_rec_energy, t_idx, i_event, t_contained): #etc
        self.hitdict = hitdict
        self.t_energy = t_energy
        self.t_idx = t_idx
        self.i_event = i_event
        self.t_contained = t_contained
        
        self.m_rec_energy = None
        
    def rec_energy(self):
        if self.m_rec_energy is None:
            self.m_rec_energy = np.sum(self.hitdict['energy'])
        return self.m_rec_energy
    
    def df(self):
        dfd = self.hitdict.copy()
        dfd['t_energy'] = np.zeros_like(self.hitdict['x']) + self.t_energy
        dfd['t_rec_energy'] = np.zeros_like(self.hitdict['x']) + self.t_rec_energy
        dfd['t_contained'] = np.zeros_like(self.hitdict['x']) + self.t_contained
        dfd['t_idx'] = np.zeros_like(self.hitdict['x'], dtype='int32') + self.t_idx
        return pd.DataFrame.from_dict(dfd)
    
    def quickplot(self, outfile = None, show = False):
        
        assert outfile is not None or show
        
        tmpdf = self.df()
        hover_data = ['t_energy','t_idx','t_rec_energy','t_contained']
        fig = px.scatter_3d(tmpdf, x="x", y="z", z="y", 
                                    color="t_idx", size="energy",
                                    hover_data=hover_data,
                                    template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        if outfile is not None:
            fig.write_html(outfile)
        if show:
            fig.show(renderer='chrome')
        

lost_showers=[]

def accumulate_stats(pred, raw, i_e, tdclass):
    
    td = tdclass()
    feat = td.createFeatureDict(raw)
    truth = td.createTruthDict(raw)
    t_idx = truth['truthHitAssignementIdx']
    u_t_idx = np.unique(t_idx)
    sel_idx_up = pred['sel_idx_up']
    
    #e_selected = tf.gather_nd(raw['t_energy'],sel_idx_up).numpy()
    t_idx_selected =  tf.gather_nd(t_idx, sel_idx_up).numpy()
    u_t_idx_selected = np.unique(t_idx_selected)
    
    for ut in u_t_idx:
        if not ut in u_t_idx_selected: #these are lost
            
            t_energy = truth['truthHitAssignedEnergies'][t_idx == ut][0]
            t_contained = truth['truthHitFullyContainedFlag'][t_idx == ut][0]
            #print('lost energy', t_energy)
            
            if t_energy < 5.:
                continue
            s = shower({'x': feat['recHitX'][t_idx == ut],
                        'y': feat['recHitY'][t_idx == ut],
                        'z': feat['recHitZ'][t_idx == ut],
                        'energy': feat['recHitEnergy'][t_idx == ut]},
                       t_energy, ut, i_e, t_contained
                       )
            lost_showers.append(
                s
                )
            print('lost shower: contained ', t_contained,' with energy', t_energy, 'rec', s.rec_energy(), 'nhits', len(feat['recHitX'][t_idx == ut]))
        

# find unique truth indices and just loop

model_path = '../models/precond3a.h5'
data_path = '../Train/data/Dec/train_data/dataCollection.djcdc'
m = load_model(model_path)
for l in m.layers:
    if isinstance(l, LossLayerBase):
        print('deactivating', l.name)
        l.active=False

#prepare input
dc = DataCollection(data_path)
#dc.samples = dc.samples[0:1]
dc.setBatchSize(1)
gen = dc.invokeGenerator()
gen.setSkipTooLargeBatches(False)
nevts = gen.getNBatches()
#gen.debuglevel = 100

times = []
lenght=[]
import time

stat_thread = None

with tqdm(total=nevts) as bar:

    for i_e, e in enumerate(gen.feedNumpyData()):
        #print(i_e,'/',nevts)
        
        #if i_e > 20:
        #    break
        if stat_thread is not None:
            stat_thread.join()
            
        lenght.append(len(e[0]))
        st = time.time()
        out = m(e)
        times.append(time.time() - st)
        
        stat_thread = threading.Thread(target=accumulate_stats, args=(out,e[0],i_e,dc.dataclass))
        stat_thread.start()
        
        bar.update(1)

import pickle

with open('pre_cond_data.pkl','wb') as f:
    pickle.dump(lost_showers,f)
    pickle.dump(times,f)
    pickle.dump(lenght,f)

times = np.array(times)
lenght = np.array(lenght)

print('mean time', np.mean(times), 'for mean hits', np.mean(lenght))
print('total time', np.sum(times))

for i,s in enumerate(lost_showers):
    if s.t_contained > 0:
        s.quickplot('plts_contained/'+str(i)+'.html')




    #now that's all the things