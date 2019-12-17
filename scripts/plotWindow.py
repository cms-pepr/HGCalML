#!/usr/bin/env python

import DeepJetCore
import os
from argparse import ArgumentParser
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
from DeepJetCore.TrainData import TrainData
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import copy
from multiprocessing import Process

def toXYZ(rho,eta,phi):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    z = 0
    if rho>0:
        z = rho * math.sinh(eta)
    else:
        z = 0
    return x,y,z




@jit(nopython=True)   
def getpositions_and_select(vx,vy,vz, takeall):
    
    selected= [0 for i in range(len(vx))]
    def distance2(x,y,z, x1,y1,z1):
        return (x-x1)**2 + (y-y1)**2 + (z-z1)**2
    selected_pos=[]
    for e in range(len(vx)):
        x,y,z = vx[e],vy[e],vz[e]
        use=True
        for s in selected_pos:
            if distance2(s[0],s[1],s[2],x,y,z) < 0.0001:
                use=False
                break
        if use:
            selected[e] = 1
        if use or takeall:
            selected_pos.append([x,y,z])
        
    return selected_pos,selected
  
   
def normalise(directions):
    norm = np.sqrt(np.sum(directions**2, axis=-1))
    return directions/norm
    

def selectEvent(rs, feat, truth, event):
    
    #get event
    feat = feat[rs[event]:rs[event+1],...]
    
    print(feat.shape)
    return feat, truth[rs[event]:rs[event+1],...]



def makePlot(outfile, 
             rechit_e, rechit_x, rechit_y, rechit_z,
             truthasso,truthenergy,
             truthX,truthY,truthZ,
             truthdirX,truthdirY,truthdirZ,
             movie=False,
             scale=True,
             isetaphi=False,
             drawarrows=True
             ):
    
    #if e < 19: return
    #copy properly
    
    
    fig = plt.figure(figsize=(10,4))
    ax = [fig.add_subplot(1,2,1, projection='3d'), fig.add_subplot(1,2,2, projection='3d')]

    
    not_assigned = truthasso < 0
    print('event', e)
    print('not assigned ',np.count_nonzero(not_assigned, axis=-1))
    
    
    selpos,select       = getpositions_and_select(truthX,truthY,truthZ,takeall=True)
    selpos = np.array(selpos, dtype='float32')
    select = np.array(select, dtype='float')
    
    #select[truthenergy<1]=0.
    
    print('selpos',selpos.shape)
    print('select',select.shape)
    
    selpos=selpos[select>0.1]
    
    seldirs,_  = getpositions_and_select(truthdirX,truthdirY,truthdirZ,takeall=True)
    seldirs = np.array(seldirs , dtype='float32')
    seldirs = seldirs[select>0.1]
    selposz = np.abs(selpos[:,2])>300
    selpos = selpos[selposz]
    seldirs = seldirs[selposz]
    
    #seldirs = normalise(seldirs)
    if isetaphi: # behave differently relative to impact point
        seldirs[:,0] -= selpos[:,0]
        seldirs[:,1] -= selpos[:,1]
    
    print('seldirs',seldirs.shape)
    print('selpos',selpos.shape)
    
    
    
    rechit_nonoise = np.array(rechit_e)
    rechit_nonoise[not_assigned] = 0
    totalenergy = np.sum(rechit_e)
    nonoise_energy = np.sum(rechit_nonoise)
    noisefraction = 1. - nonoise_energy/totalenergy
    print('totalenergy ',totalenergy)
    print('noisefraction ',noisefraction)
    print('noise energy ',totalenergy - nonoise_energy)
    
    
    
    
    pl = plotter_3d(output_file=outdir+"/plot")
    
    pl.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_e , c =truthasso)
    pl.marker_scale=1.
    pl.plot3d(ax=ax[0])
    
    
    pl.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_nonoise , c =truthasso)
    pl.plot3d(ax=ax[1])
    
    zoffset = 10
    if np.mean(selpos[:,2]) < 0:
        zoffset*=-1.
    
    dirlength = np.expand_dims(np.sqrt(np.sum(seldirs**2, axis=-1)), axis=1)
    seldirs/=dirlength #normalise
    scaling = 1. + np.log(dirlength+1)
    scaling = 1. + np.log(scaling)
    if scale:
        seldirs *= scaling*10.
    #seldirs *= (2. + np.log(scaling)) 
    #x and z switched for CMS style coordinates
    def addarrows(ax):
        if not drawarrows:
            return
        ax.quiver(selpos[:,2],selpos[:,1],selpos[:,0], 
                 seldirs[:,2],seldirs[:,1],seldirs[:,0], #length=10, normalize=True,
                 color='r',
                 pivot='tip',
                 alpha=0.5)
    
    addarrows(ax[1])
    fig.savefig(outfile+".pdf")
    plt.close()
    
    
    
    
    if movie:
        pl2 = plotter_3d(output_file=outdir+"/plot_noise")
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_e , c =truthasso)
    
        mm = movie_maker(pl2, output_file=outfile+"_mm_noise", silent=False, axfunc=addarrows)
        mm.make_movie()
        
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_nonoise , c =truthasso)
        mm2 = movie_maker(pl2, output_file=outfile+"_mm_nonoise", silent=False, axfunc=addarrows, dpi=200)
        mm2.make_movie()
    

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--movie', help='Also create a movie', action='store_true' , default=False )
parser.add_argument('-n', help='Event', default='0' )
parser.add_argument('--default', help='Use default simclusters (not hgctruth merged)', action='store_true' , default=False )
args = parser.parse_args()

infile = args.inputFile
outdir=args.outputDir+"/"
nevents=int(args.n)
make_movie = args.movie
os.system('mkdir -p '+outdir)
treename = "WindowNTupler/tree"
if args.default:
    treename = "WindowNTuplerDefaultTruth/tree"


td = TrainData()
if infile[-5:] == "djctd":
    td.readFromFile(infile)
else:
    from datastructures import TrainData_window
    td=TrainData_window()
    td.readFromSourceFile(infile)
    print("nelements",td.nElements())
#td.skim(event)
feat_rs = td.transferFeatureListToNumpy()
truth_rs = td.transferTruthListToNumpy()

feat = feat_rs[0]
rs = feat_rs[1][:,0]
truth = truth_rs[0]

def worker(eventno):
    pfeat, ptruth = selectEvent(rs,feat,truth,eventno)
    
    rechit_e = pfeat[:,0]
    
    rechit_x = pfeat[:,5]
    rechit_y = pfeat[:,6]
    rechit_z = pfeat[:,7]
    
    rechit_eta = pfeat[:,1]
    rechit_phi = pfeat[:,2]
    rechit_r = pfeat[:,4]
    
    truthasso = ptruth[:,0]
    truthenergy = ptruth[:,1]
    
    truthX = ptruth[:,2]
    truthY = ptruth[:,3]
    truthZ = ptruth[:,4]
    truthdirX = ptruth[:,5]
    truthdirY = ptruth[:,6]
    truthdirZ = ptruth[:,7]
    
    truthEta = ptruth[:,8]
    truthPhi = ptruth[:,9]
    truthR = ptruth[:,10]
    truthdirEta = ptruth[:,11]
    truthdirPhi = ptruth[:,12]
    truthdirR = ptruth[:,13]
    
    p = Process(target=makePlot, args=(outdir+str(eventno), 
             rechit_e, rechit_x, rechit_y, rechit_z,
             truthasso,truthenergy,
             truthX,truthY,truthZ,
             truthdirX,truthdirY,truthdirZ,
             make_movie))
    p.start()
    #makePlot()
    
    p2 = Process(target=makePlot,args=(outdir+"etaphir_"+str(eventno), 
             rechit_e, rechit_eta, rechit_phi, rechit_r,
             truthasso,truthenergy,
             truthEta,truthPhi,truthR,
             truthdirEta,truthdirPhi,truthdirR,
             make_movie,
             False,
             True,
             False))
    p2.start()
    return p,p2
    

#from multiprocessing import Pool
#p = Pool()
#print(p.map(worker, range(nevents)))

#exit()
for e in range(nevents):
    worker(e)
    
    
    