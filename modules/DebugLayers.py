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
    
    fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    return fig_thresh
    
class _DebugPlotBase(tf.keras.layers.Layer):    
    def __init__(self,
                 plot_every: int,
                 outdir :str='' , 
                 plot_only_training=True,
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
        import os
        if plot_every > 0 and len(self.outdir):
            os.system('mkdir -p '+self.outdir)
        if not os.path.isdir(os.path.dirname(self.outdir)): #could not be created
            self.outdir=''
            
        
    def get_config(self):
        config = {'plot_every': self.plot_every}#outdir is explicitly not saved and needs to be set again every time
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
        self.plot(inputs,training)
        return out
    
    
    
class PlotEdgeDiscriminator(_DebugPlotBase):    
    def __init__(self,**kwargs):
        '''
        Takes as input
         - edge score 
         - neighbour indices
         - truth indices 
         
        Returns edge score  (unchanged)
        '''
        super(PlotEdgeDiscriminator, self).__init__(**kwargs) 
    
    def plot(self, inputs, training=None):   

        assert len(inputs) == 3
        
        e_s, nidx, t_idx = inputs
        n_t_idx = SelectWithDefault(nidx,t_idx,-4)#mask the ones that are masked.
        n_t_idx = tf.where(n_t_idx<0,-2,n_t_idx)#make no connections for noise

        same = tf.cast(t_idx == n_t_idx[:,1:,0], dtype='float32')
        #just reshape to flat
        n_t_idx = tf.reshape(n_t_idx[:,1:,0],[-1])
        same = tf.reshape(same, [-1,1])
        e_s = tf.reshape(e_s, [-1,1])
        #remove non-existing neighbours
        same = same[n_t_idx>-3]
        e_s = e_s[n_t_idx>-3]
        
        data={'same':  same.numpy(),'edge_score':  e_s.numpy()}
        df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
        fig = px.histogram(df, x="edge_score", color="same",log_y=False)
        fig.write_html(self.create_base_output_path()+".html")
        
        fpr, tpr, thresholds = roc_curve(same, e_s)
        fig = quick_roc(fpr, tpr, thresholds)
        fig.write_html(self.create_base_output_path()+"_roc.html")
        
        
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
        
        fpr, tpr, thresholds = roc_curve(t_idx, score)
        fig = quick_roc(fpr, tpr, thresholds)
        fig.write_html(self.create_base_output_path()+"_roc.html")
        
        
             
    
class PlotCoordinates(_DebugPlotBase):
    
    def __init__(self,**kwargs):
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
        
        assert len(inputs) == 4
        
        coords, features, tidx, rs = inputs
        
        #just select first
        coords = coords[0:rs[1]]
        tidx = tidx[0:rs[1]]
        features = features[0:rs[1]]
        
        #just project
        for i in range(coords.shape[1]-2):
            data={
                'X':  coords[:,0+i:1+i].numpy(),
                'Y':  coords[:,1+i:2+i].numpy(),
                'Z':  coords[:,2+i:3+i].numpy(),
                'tIdx': tidx[:,0:1].numpy(),
                'features': features[:,0:1].numpy()
                }
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            df['orig_tIdx']=df['tIdx']
            rdst = np.random.RandomState(1234567890)#all the same
            shuffle_truth_colors(df,'tIdx',rdst)
            
            hover_data=['orig_tIdx']
            fig = px.scatter_3d(df, x="X", y="Y", z="Z", 
                                color="tIdx",
                                size='features',
                                hover_data=hover_data,
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.outdir+'/'+self.name+'_'+str(i)+".html")
            
            df = df[df['orig_tIdx']>=0]
            
            fig = px.scatter_3d(df, x="X", y="Y", z="Z", 
                                color="tIdx",
                                size='features',
                                hover_data=hover_data,
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.create_base_output_path()+'_'+str(i)+"_no_noise.html")
            
        
