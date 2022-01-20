'''
Layers with the sole purpose of debugging.
These should not be included in 'real' models
'''


import tensorflow as tf
import plotly.express as px
import pandas as pd
import numpy as np
from plotting_tools import shuffle_truth_colors
    
class PlotCoordinates(tf.keras.layers.Layer):
    
    def __init__(self,
                 plot_every: int,
                 outdir :str='' , 
                 **kwargs):
        '''
        Takes as input
         - coordinate 
         - features (first will be used for size)
         - truth indices 
         - row splits
         
        Returns coordinates (unchanged)
        '''
        
        if 'dynamic' in kwargs:
            super(PlotCoordinates, self).__init__(**kwargs)
        else:
            super(PlotCoordinates, self).__init__(dynamic=False,**kwargs)
            
        
        self.plot_every = plot_every
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
        base_config = super(PlotCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def build(self,input_shape):
        super(PlotCoordinates, self).build(input_shape)
        
    def call(self, inputs, training=None):
        
        coords, features, tidx, rs = inputs
        if self.plot_every <=0:
            return coords
        if not hasattr(coords, 'numpy'): #only in eager
            return coords
        if training is None or training == False:#only run in training mode
            return coords
        
        #plot initial state
        if self.counter>=0 and self.counter < self.plot_every:
            self.counter+=1
            return inputs[0]
        
        if len(self.outdir)<1:
            return inputs[0]
        
        #print('making debug plot')
        self.counter=0
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
            fig.write_html(self.outdir+'/'+self.name+'_'+str(i)+"_no_noise.html")
            
        
        return inputs[0]
        
