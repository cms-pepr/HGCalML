'''
Layers with the sole purpose of debugging.
These should not be included in 'real' models
'''


import tensorflow as tf
import plotly.express as px
import pandas as pd
import numpy as np
from plotting_callbacks import shuffle_truth_colors
    
class PlotCoordinates(tf.keras.layers.Layer):
    
    def __init__(self, plot_every: int=100,
                 outdir :str='.', **kwargs):
        '''
        Takes as input
         - coordinate 
         - features (first will be used for size)
         - truth indices 
         - row splits
         
        Returns coordinates (unchanged)
        '''
        super(PlotCoordinates, self).__init__(**kwargs)
        
        self.plot_every = plot_every
        self.outdir = outdir
        self.counter=-1
        import os
        os.system('mkdir -p '+self.outdir)
    
    def get_config(self):
        config = {'plot_every': self.plot_every,
                  'outdir': self.outdir
                  }
        base_config = super(PlotCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def call(self, inputs):
        
        coords, features, tidx, rs = inputs
        if not hasattr(coords, 'numpy'): #inly in eager
            return inputs[0]
        
        #plot initial state
        if self.counter>=0 and self.counter < self.plot_every:
            self.counter+=1
            return inputs[0]
        print('making debug plot')
        self.counter=0
        #just select first
        coords = coords[0:rs[1]]
        tidx = tidx[0:rs[1]]
        features = features[0:rs[1]]
        
        data={
            'X':  coords[:,0:1].numpy(),
            'Y':  coords[:,1:2].numpy(),
            'Z':  coords[:,2:3].numpy(),
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
        fig.write_html(self.outdir+'/'+self.name+".html")
        
        df = df[df['orig_tIdx']>=0]
        
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", 
                            color="tIdx",
                            size='features',
                            hover_data=hover_data,
                            template='plotly_dark',
                color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(self.outdir+'/'+self.name+"_no_noise.html")
        
        
        return inputs[0]
        

