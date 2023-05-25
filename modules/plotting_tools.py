

'''
This file! is supposed to contain only some helper functions for plotting
and coordinate transformations. If your function is longer than a few lines
here is probably not the right place for it

'''

import numpy as np
import os

def calc_r(x,y):
    return np.sqrt(x ** 2 + y ** 2)

def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.)

def calc_phi(x, y):
    return np.arctan2(x, y)

def rotation(counter):
    angle_in = 10. * counter + 60.
    while angle_in >= 360: angle_in -= 360
    while angle_in <= -360: angle_in -= 360
    return angle_in          

def publish(file_to_publish, publish_to_path):
    cpstring = 'cp -f '
    if "@" in publish_to_path:
        cpstring = 'scp '
    basefilename = os.path.basename(file_to_publish)
    os.system(cpstring + file_to_publish + ' ' + publish_to_path +'_'+basefilename+ ' 2>&1 > /dev/null') 

def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx",rdst=None):
    ta = df[qualifier]
    unta = np.unique(ta)
    unta = unta[unta>-0.1]
    if rdst is None:
        np.random.shuffle(unta)
    else:
        rdst.shuffle(unta)
    out = ta.copy()
    for i in range(len(unta)):
        out[ta ==unta[i]]=i
    df[qualifier] = out
       
    
def reassign_indices_random(ta,rdst=None):   
    unta = np.unique(ta)
    unta = unta[unta>-0.1]
    if rdst is None:
        np.random.shuffle(unta)
    else:
        rdst.shuffle(unta)
    out = ta.copy()
    for i in range(len(unta)):
        out[ta ==unta[i]]=i
    return out   


