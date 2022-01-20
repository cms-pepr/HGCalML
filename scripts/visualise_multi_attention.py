#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import tqdm

# Generate nicely looking random 3D-field
np.random.seed(0)
l = 40
mgrid = (np.mgrid[:l, :l, :l]-l/2)/l
X, Y, Z = mgrid
#vol = np.zeros((l, l, l))



mgrid = np.transpose(mgrid,[1,2,3,0])
mgrid = np.expand_dims(mgrid,axis=-1)
print(mgrid.shape)
#read data

file3="~/Downloads/multiattention.html.3.df.pkl"
file5="~/Downloads/multiattention.html.5.df.pkl"
file8="~/Downloads/multiattention.html.8.df.pkl"

file=file8

outdir='plts8'

df=pd.read_pickle(file)

#check number of coordinates
cols = df.columns
coords = np.unique([int(c[-1]) for c in cols])
points = np.unique([int(c[-3]) for c in cols])
#find name

import os
os.system('mkdir -p '+outdir)
def printevent(event,counter,outdir=outdir):
    data=[]
    vardata = []
    for pi in points:
        pdata=[]
        pvar=[]
        for ci in coords:
            d = df['pre_selection_add_stage_0_att_gn1_coord_add_mean_'+str(pi)+'_'+str(ci)]
            pdata.append(d[event])
            pvar.append(df['pre_selection_add_stage_0_att_gn1_coord_add_var_'+str(pi)+'_'+str(ci)][event])
        data.append(pdata)
        vardata.append(pvar)
    data = np.array(data)
    vardata= np.array(vardata)
    
    
    def trfdata(x):
        x = np.transpose(x,[1,0])
        return np.expand_dims(x,axis=(0,1,2))
        
    #process to plot
    data = trfdata(data)
    vardata = trfdata(vardata)
    
    vol = np.exp(-3.*(data-mgrid)**2/vardata )
    #print(vol.shape)
    vol = np.prod(vol,axis=3)#the x**2 axis
    vol = np.sum(vol,axis=-1)#the points axis
    #insert data here. make data span a function for mesh grid
    #pts = (l * np.random.rand(3, 15)).astype(np.int)
    #vol[tuple(indices for indices in pts)] = 1
    
    
    #from scipy import ndimage
    #vol = ndimage.gaussian_filter(vol, 1)
    vol /= vol.max()
    
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=vol.flatten(),
        isomin=0.2,
        isomax=0.7,
        opacity=0.1,
        surface_count=25,
        ))
    #fig.update_layout(scene_xaxis_showticklabels=False,
    #                  scene_yaxis_showticklabels=False,
    #                  scene_zaxis_showticklabels=False)
    #
    #fig.show(renderer='chrome')
    if counter>=0:
        fig.write_image(outdir+'/'+str(counter).zfill(10)+'.png')
    else:
        fig.write_html(outdir+'/last.html')
    

printevent(len(df)-1,-1)
exit()
nframes=60
events = np.arange(0, len(df)-1, (len(df)-1) // nframes)
#events = [0]
print(events)
counter=0
lastev=0
for e in tqdm.tqdm(events):
    printevent(e,counter)
    counter+=1
    lastev=e
    

