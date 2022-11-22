'''
Layers with the sole purpose of debugging.
These should not be included in 'real' models
'''


import tensorflow as tf
import plotly.express as px
import pandas as pd
import numpy as np
from plotting_tools import shuffle_truth_colors
from oc_helper_ops import SelectWithDefault
from sklearn.metrics import roc_curve
from DeepJetCore.training.DeepJet_callbacks import publish
import queue
import os




def quick_roc(fpr,tpr,thresholds):
    df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    }, index=thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"
    
    fig_thresh = px.line(
        df, title='TPR and FPR at every threshold',
        width=700, height=500
    )
    
    idx_1per = find_nearest_idx(fpr, 0.01)
    idx_35per = find_nearest_idx(fpr, 0.035)
    tpr_1per = round(tpr[idx_1per],3)
    tpr_35per = round(tpr[idx_35per],3)
    
    fig_thresh.add_annotation(x=thresholds[idx_1per], y=tpr_1per,
            text=str(tpr_1per)+" @ 1%",
            showarrow=True,
            arrowhead=1)
    
    fig_thresh.add_annotation(x=thresholds[idx_35per], y=tpr_35per,
            text=str(tpr_35per)+" @ 3.5%",
            showarrow=True,
            arrowhead=1)
    
    fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    return fig_thresh

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def down_sample(tods: list, max_inputs=10000):
    assert len(tods)>0
    if tods[0].shape[0] > max_inputs:
        selidxs = np.arange(tods[0].shape[0])
        selidxs = np.random.choice(selidxs, size=max_inputs, replace=False)
        out=[]
        for it in tods:
            it = it.numpy()
            out.append(it[selidxs])
        tods = out
    return tods
    
class _DebugPlotBase(tf.keras.layers.Layer):    
    def __init__(self,
                 plot_every: int,
                 outdir :str='' , 
                 plot_only_training=True,
                 publish = None,
                 **kwargs):
        
        if 'dynamic' in kwargs:
            super(_DebugPlotBase, self).__init__(**kwargs)
        else:
            super(_DebugPlotBase, self).__init__(dynamic=False,**kwargs)
            
        self.plot_every = plot_every
        self.plot_only_training = plot_only_training
        if len(outdir) < 1:
            self.plot_every=0
        self.outdir = outdir
        self.counter=-1
        if not os.path.isdir(os.path.dirname(self.outdir)): #could not be created
            self.outdir=''
            
        self.publish = publish
            
        
    def get_config(self):
        config = {'plot_every': self.plot_every,
                  'outdir': self.outdir,
                  'publish': self.publish}
        base_config = super(_DebugPlotBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def build(self,input_shape):
        super(_DebugPlotBase, self).build(input_shape)
        
    def plot(self, inputs, training=None):
        raise ValueError("plot(self, inputs, training=None) needs to be implemented in inheriting class")
    
    def create_base_output_path(self):
        return self.outdir+'/'+self.name
    
    def call(self, inputs, training=None):
        out=inputs
        if isinstance(inputs,list):
            out=inputs[0]
        if (training is None or training == False) and self.plot_only_training:#only run in training mode
            return out
        
        if inputs[0].shape[0] is None or inputs[0].shape[0] == 0:
            return out
        
        if self.plot_every <=0:
            return out
        if not hasattr(out, 'numpy'): #only in eager
            return out
        
        #plot initial state
        if self.counter>=0 and self.counter < self.plot_every:
            self.counter+=1
            return out
        
        if len(self.outdir)<1:
            return out
        
        #only now plot
        self.counter=0
        
        os.system('mkdir -p '+self.outdir)
            
        self.plot(inputs,training)
        return out
    

def switch_off_debug_plots(keras_model):
    for l in keras_model.layers:
        if isinstance(l, _DebugPlotBase):
            l.plot_every = -1
    return keras_model
    
class PlotEdgeDiscriminator(_DebugPlotBase):    
    def __init__(self, average=16, **kwargs):
        '''
        Takes as input
         - edge score 
         - neighbour indices
         - truth indices 
         
        Returns edge score  (unchanged)
        '''
        super(PlotEdgeDiscriminator, self).__init__(**kwargs) 
        
        self.prev_batches = queue.Queue(average)
        #self.outdir = '/eos/home-j/jkiesele/www/files/temp/Sept2022/tmp/pf_pre_test3b/'
        #self.plot_every = 1200
    
    def get_config(self):
        config = {'average': self.prev_batches.maxsize}#outdir/publish is explicitly not saved and needs to be set again every time
        base_config = super(PlotEdgeDiscriminator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def plot(self, inputs, training=None):   

        assert len(inputs) == 4 or len(inputs) == 3
        if len(inputs) == 4:
            e_s, nidx, t_idx, t_energy = inputs
        else:
            e_s, nidx, t_idx = inputs
            t_energy = tf.ones_like(t_idx, dtype='float32')
            
        n_t_idx = SelectWithDefault(nidx,t_idx,-4)#mask the ones that are masked.
        n_t_idx = tf.where(n_t_idx<0,-2,n_t_idx)#make no connections for noise

        from LossLayers import _calc_energy_weights   
        # energy weighted
        eweight = _calc_energy_weights(t_energy)
        eweight = SelectWithDefault(nidx,eweight, 0)
        eweight = eweight[:,1:]
        
        same = tf.cast(t_idx == n_t_idx[:,1:,0], dtype='float32')
        #just reshape to flat
        n_t_idx = tf.reshape(n_t_idx[:,1:,0],[-1])
        same = tf.reshape(same, [-1,1])
        e_s = tf.reshape(e_s, [-1,1])
        eweight = tf.reshape(eweight, [-1,1])
        #remove non-existing neighbours
        same = same[n_t_idx>-3]
        e_s = e_s[n_t_idx>-3]
        eweight = eweight[n_t_idx>-3]
        
        
        data={'same':  same.numpy(),'edge_score':  e_s.numpy()}
        df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
        fig = px.histogram(df, x="edge_score", color="same",log_y=False)
        fig.write_html(self.create_base_output_path()+".html")
        
        max_inputs = 10000
        e_s, same, eweight = down_sample([e_s,same,eweight], max_inputs)
        
        self.prev_batches.put([e_s, same, eweight])
        
        if self.prev_batches.full():
            qsize = self.prev_batches.maxsize
            e_s = np.concatenate([self.prev_batches.queue[i][0] for i in range(qsize)],axis=0)
            same = np.concatenate([self.prev_batches.queue[i][1] for i in range(qsize)],axis=0)
            eweight = np.concatenate([self.prev_batches.queue[i][2] for i in range(qsize)],axis=0)
            
            self.prev_batches.get()#remove first
            
        fpr, tpr, thresholds = roc_curve(same, e_s)
        fig = quick_roc(fpr, tpr, thresholds)
        fig.write_html(self.create_base_output_path()+"_roc.html")
        
        if self.publish is not None:
            publish(self.create_base_output_path()+"_roc.html", self.publish)
        
        fpr, tpr, thresholds = roc_curve(same, e_s, sample_weight = eweight)
        fig = quick_roc(fpr, tpr, thresholds)
        fig.write_html(self.create_base_output_path()+"_weighted_roc.html")
        
        if self.publish is not None:
            publish(self.create_base_output_path()+"_weighted_roc.html", self.publish)
            
        
        
class PlotNoiseDiscriminator(_DebugPlotBase):    
    def __init__(self,**kwargs):
        '''
        Takes as input
         - not_noise_score
         - truth indices 
         
        Returns edge score  (unchanged)
        '''
        super(PlotNoiseDiscriminator, self).__init__(**kwargs) 
    
    def plot(self, inputs, training=None):   

        assert len(inputs) == 2
        
        score,t_idx = inputs 
        t_idx = tf.cast(t_idx>=0, dtype='float32')
        data={'not_noise':  t_idx.numpy(),'not_noise_score':  score.numpy()}
        df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
        fig = px.histogram(df, x="not_noise_score", color="not_noise",log_y=False)
        fig.write_html(self.create_base_output_path()+".html")
        
        
        max_inputs = 10000
        t_idx, score = down_sample([t_idx,score], max_inputs)
        
        fpr, tpr, thresholds = roc_curve(t_idx, score)
        fig = quick_roc(fpr, tpr, thresholds)
        fig.write_html(self.create_base_output_path()+"_roc.html")
        
        if self.publish is not None:
            publish(self.create_base_output_path()+"_roc.html", self.publish)
             
    
class PlotCoordinates(_DebugPlotBase):
    
    def __init__(self, **kwargs):
        '''
        Takes as input
         - coordinate 
         - features (first will be used for size)
         - truth indices 
         - row splits
         
        Returns coordinates (unchanged)
        '''
        super(PlotCoordinates, self).__init__(**kwargs)
        
        
    def plot(self, inputs, training=None):
        
        coords, features, hoverfeat, nidx, tidx, rs = 6*[None]
        if len(inputs) == 4:
            coords, features, tidx, rs = inputs
        elif len(inputs) == 5:
            coords, features, hoverfeat, tidx, rs = inputs
        elif len(inputs) == 6:
            coords, features, hoverfeat, nidx, tidx, rs = inputs
        
        #just select first
        coords = coords[0:rs[1]]
        tidx = tidx[0:rs[1]]
        if len(tidx.shape) <2:
            tidx = tidx[...,tf.newaxis]
        features = features[0:rs[1]]
        if hoverfeat is not None:
            hoverfeat = hoverfeat[0:rs[1]]
            hoverfeat = hoverfeat.numpy()
            
        if nidx is not None:
            nidx = nidx[0:rs[1]]
            n_tidxs = SelectWithDefault(nidx, tidx, -2)
            n_sameasprobe = tf.cast(tf.expand_dims(tidx, axis=2) == n_tidxs[:,1:,:], dtype='float32')
            av_same = tf.reduce_mean(n_sameasprobe, axis=1)# V x 1
            
        #just project
        for i in range(coords.shape[1]-2):
            data={
                'X':  coords[:,0+i:1+i].numpy(),
                'Y':  coords[:,1+i:2+i].numpy(),
                'Z':  coords[:,2+i:3+i].numpy(),
                'tIdx': tidx[:,0:1].numpy(),
                'features': features[:,0:1].numpy()
                }
            hoverdict={}
            if hoverfeat is not None:
                for j in range(hoverfeat.shape[1]):
                    hoverdict['f_'+str(j)] = hoverfeat[:,j:j+1]
                data.update(hoverdict)
                
            if nidx is not None:
                data.update({'av_same': av_same})
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            df['orig_tIdx']=df['tIdx']
            rdst = np.random.RandomState(1234567890)#all the same
            shuffle_truth_colors(df,'tIdx',rdst)
            
            hover_data=['orig_tIdx']+[k for k in hoverdict.keys()]
            if nidx is not None:
                hover_data.append('av_same')
            fig = px.scatter_3d(df, x="X", y="Y", z="Z", 
                                color="tIdx",
                                size='features',
                                hover_data=hover_data,
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.outdir+'/'+self.name+'_'+str(i)+".html")
            
            
            if self.publish is not None:
                publish(self.outdir+'/'+self.name+'_'+str(i)+".html", self.publish)
            
            df = df[df['orig_tIdx']>=0]
            
            fig = px.scatter_3d(df, x="X", y="Y", z="Z", 
                                color="tIdx",
                                size='features',
                                hover_data=hover_data,
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.create_base_output_path()+'_'+str(i)+"_no_noise.html")
            
            
            if self.publish is not None:
                publish(self.create_base_output_path()+'_'+str(i)+"_no_noise.html", self.publish)
        
