#!/usr/bin/env python3

import DeepJetCore
import os
from argparse import ArgumentParser
from plotting_tools import plotter_3d, movie_maker, plotter_fraction_colors
from DeepJetCore.TrainData import TrainData
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit, njit
import copy
from multiprocessing import Process
import random
import uproot

def toXYZ(rho,eta,phi):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    z = 0
    if rho>0:
        z = rho * math.sinh(eta)
    else:
        z = 0
    return x,y,z



@njit
def firstmatch(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

#@jit(nopython=True)   
def getpositions_and_select(vx,vy,vz, takeall,truthasso):
    
    selected= [0 for i in range(len(vx))]
    selected_pos= [[0.,0.,0.] for i in range(len(vx))]
    
    unique_truth = np.unique(truthasso)
    
    for i in unique_truth:
        if i < 0:
            continue
        e = firstmatch(truthasso, i)
        #print(e)
        e = e[0]
        selected[e] = 1.
        
        x,y,z = vx[e],vy[e],vz[e]
        selected_pos[e]=[x,y,z]
    
        
    return selected_pos,selected
  
   
def normalise(directions):
    norm = np.sqrt(np.sum(directions**2, axis=-1))
    return directions/norm
    

def selectEvent(rs, feat, truth, event):
    
    #get event
    featout = feat[rs[event]:rs[event+1],...]
    
    #print(feat.shape)
    return featout, truth[rs[event]:rs[event+1],...]



def makePlot(outfile, 
             rechit_e, rechit_x, rechit_y, rechit_z,
             is_track,
             truthasso_in,truthenergy,
             truthX,truthY,truthZ,
             truthdirX,truthdirY,truthdirZ,
             movie=False,
             scale=True,
             isetaphi=False,
             drawarrows=True,
             show=False,
             real_truth_energy=None
             ):
    
    #if e < 19: return
    #copy properly
    truthasso = np.array(truthasso_in)
    
    fig = plt.figure(figsize=(24,8))
    ax = [fig.add_subplot(1,4,1, projection='3d'), fig.add_subplot(1,4,2, projection='3d'),
          fig.add_subplot(1,4,3), fig.add_subplot(1,4,4)]

    
    not_assigned = truthasso < 0
    #print('event', e)
    print('not assigned ',np.count_nonzero(not_assigned, axis=-1),' of total ',len(truthasso))
    
    #print('truthZ',truthZ)
    selpos,select       = getpositions_and_select(truthX,truthY,truthZ,takeall=True,truthasso=truthasso)
    
    selpos = np.array(selpos, dtype='float32')
    select = np.array(select, dtype='float')
    
    #select[truthenergy<1]=0.
    
    #print('selpos',selpos.shape)
    #print('select',select.shape)
    
    selpos=selpos[select>0.1]
    
    seldirs,_  = getpositions_and_select(truthdirX,truthdirY,truthdirZ,takeall=True,truthasso=truthasso)
    seldirs = np.array(seldirs , dtype='float32')
    seldirs = seldirs[select>0.1]
    truth_sel = np.abs(selpos[:,2])>0
    selpos = selpos[truth_sel]
    seldirs = seldirs[truth_sel]
    
    #seldirs = normalise(seldirs)
    if isetaphi: # behave differently relative to impact point
        seldirs[:,0] -= selpos[:,0]
        seldirs[:,1] -= selpos[:,1]
    
    #print('seldirs',seldirs.shape)
    #print('selpos',selpos.shape)
    
    
    
    rechit_nonoise = np.array(rechit_e)
    rechit_nonoise[not_assigned] = 0
    totalenergy = np.sum(rechit_e)
    nonoise_energy = np.sum(rechit_nonoise)
    noisefraction = 1. - nonoise_energy/totalenergy
    print('totalenergy ',totalenergy)
    print('noisefraction ',noisefraction)
    print('noise energy ',totalenergy - nonoise_energy)
    print('tracks',np.sum(is_track))
    #randomise the colours a bit
    
    
    rgbcolor = plt.get_cmap('prism')((truthasso+1.)/(np.max(truthasso)+1.))[:,:-1]

    rgbcolor[truthasso<0]=[0.92,0.92,0.92]
    
    scatter_size = np.log(rechit_e + 1)+0.1
    ax[2].scatter(rechit_x, rechit_y,
              s=np.log(rechit_e + 1.)+0.1,
              c=rgbcolor)
    
    
    ax[2].scatter(rechit_x[is_track], rechit_y[is_track],
              s=scatter_size[is_track]*10.,
              c=rgbcolor[is_track]/2.,
              marker='x')
    
    scatter_size[not_assigned] = 0
    ax[3].scatter(rechit_x, rechit_y,
              s=scatter_size,
              c=rgbcolor)
    
    
    #print('selpos',selpos)
    ax[3].scatter(selpos[:,0],selpos[:,1],
                  s = 1.0,
                  marker='+',
                  c='k')
    
    ax[3].scatter(rechit_x[is_track], rechit_y[is_track],
              s=scatter_size[is_track]*10.,
              c=rgbcolor[is_track]/2.,
              marker='x')
    
    sel_truth_e = truthenergy[select>0.1][truth_sel]
    sel_real_truth_e=None
    if real_truth_energy is not None:
        sel_real_truth_e = real_truth_energy[select>0.1][truth_sel]
    sel_truth_asso = truthasso[select>0.1][truth_sel]
    #print('showers',sel_truth_asso)
    total_impact_E=0
    total_reco_E=0
    total_dep_energies=[]
    
    fontsize=2#'x-small'
    
    for i in range(len(sel_truth_e)):
            #predicted
            ax[3].text(selpos[i,0],selpos[i,1],
                    s = str(round(sel_truth_e[i])),
                    verticalalignment='bottom', horizontalalignment='right',
                    rotation=10,
                    fontsize=fontsize,
                    fontstyle='italic')
            
            if sel_real_truth_e is not None:
                ax[3].text(selpos[i,0],selpos[i,1],
                        s = str(round(sel_real_truth_e[i])),
                        verticalalignment='top', horizontalalignment='left',
                        rotation=10,
                        fontsize=fontsize,
                        fontstyle='normal')
                
            total_impact_E += sel_real_truth_e[i]
            #recalculate
            thisidx = sel_truth_asso[i]
            allreco = np.sum(rechit_e[truthasso == thisidx])
            total_dep_energies.append(allreco)
            total_reco_E += allreco
            
            ax[3].text(selpos[i,0],selpos[i,1],
                    s = str(round(allreco))[:4],
                    verticalalignment='top', horizontalalignment='right',
                    rotation=10,
                    fontsize=fontsize,
                    fontstyle='italic')
            
            #print('shower', thisidx, 'has deposited energy of',allreco,' precalc depo of',sel_truth_e[i], 'impact of',sel_real_truth_e[i])
            
    print('total reco non-noise E', total_reco_E, ' vs total impact E', total_impact_E)        
    
    
    pl = plotter_3d(output_file=outdir+"/plot", colorscheme=None)
    
    pl.set_data(x = rechit_x.copy() , y=rechit_y.copy()   , z=rechit_z.copy(), e=rechit_e.copy() , c =rgbcolor)
    pl.marker_scale=1.
    pl.plot3d(ax=ax[0])
    
    
    pl.set_data(x = rechit_x.copy() , y=rechit_y.copy()   , z=rechit_z.copy(), e=rechit_nonoise.copy() , c =rgbcolor)
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
    
    
    if show:
        plt.show()
    
    
    fig.savefig(outfile+".pdf")
    plt.close()
    
    
    
    
    if movie:
        pl2 = plotter_3d(output_file=outdir+"/plot_noise")
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_e , c =truthasso)
    
        #mm = movie_maker(pl2, output_file=outfile+"_mm_noise", silent=False, axfunc=addarrows)
        #mm.make_movie()
        
        pl2.set_data(x = rechit_x , y=rechit_y   , z=rechit_z, e=rechit_nonoise , c =truthasso)
        mm2 = movie_maker(pl2, output_file=outfile+"_mm_nonoise", silent=False, axfunc=addarrows, dpi=600)
        mm2.make_movie()
        
    print('plotting energy')
    
    def plot_loghist(x, bins):
        hist, bins = np.histogram(x, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(x, bins=logbins)
        plt.xscale('log')
  
    plot_loghist(total_dep_energies,20)
    plt.savefig(outfile+"_depE.pdf")
    

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--movie', help='Also create a movie', action='store_true' , default=False )
parser.add_argument('--show', help='Also create a movie', action='store_true' , default=False )
parser.add_argument('-n', help='Event', default='0' )
parser.add_argument('--default', help='Use default simclusters (not hgctruth merged)', action='store_true' , default=False )
parser.add_argument('--no_parallel', help="Don't use multiple threads", action='store_true' , default=False )
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
    from datastructures import TrainData_window, TrainData_window_tracks
    td=TrainData_window_tracks()
    td.readFromSourceFile(infile, treename=treename)
    print("nelements",td.nElements())
    
#td.skim(event)
feat_rs = td.transferFeatureListToNumpy()
truth_rs = td.transferTruthListToNumpy()


feat = feat_rs[0]
hitid = feat[:,2:3].copy()
feat[:,2]=np.arctan2(feat[:,6], feat[:,5])#swapped somehow

feat = np.concatenate([feat,hitid],axis=-1)
rs = feat_rs[1][:,0]
truth = truth_rs[0]

nevents = min(td.nElements(),nevents)
print('plotting',nevents,'events')

def worker(eventno, show=False):
    pfeat, ptruth = selectEvent(rs,feat,truth,eventno)
    
    rechit_e = pfeat[:,0]
    
    rechit_x = pfeat[:,5]
    rechit_y = pfeat[:,6]
    rechit_z = pfeat[:,7]
    
    rechit_eta = pfeat[:,1]
    rechit_phi = pfeat[:,2]
    
    #just to get the plot right
    phi_mean = np.arctan2(np.mean(rechit_x),np.mean(rechit_y))
    is_track = pfeat[:,-1] < 0
    
    #rechit_phi -= phi_mean
    #rechit_phi = np.unwrap(rechit_phi)
    
    rechit_r = pfeat[:,4]
    
    truthasso = ptruth[:,0]
    truthenergy = ptruth[:,1]
    real_truth_energy = ptruth[:,15]
    
    truthX = ptruth[:,2]
    truthY = ptruth[:,3]
    truthZ = ptruth[:,4]
    truthdirX = ptruth[:,5]
    truthdirY = ptruth[:,6]
    truthdirZ = ptruth[:,7]
    
    truthEta = ptruth[:,8]
    truthPhi = ptruth[:,9]
    #truthPhi -= phi_mean
    #truthPhi = np.unwrap(truthPhi)
    
    truthR =  truthZ / np.cos(2* np.arctan(np.exp(-truthEta)))
    truthdirEta = ptruth[:,11]
    truthdirPhi = ptruth[:,12]
    truthdirR = ptruth[:,13]
    
    ticlAsso = ptruth[:,17]
    ticlE    = ptruth[:,18]
    
    #with plt.xkcd():
    print('>>>>>>>>>>>>>> plotting x/y')
    makePlot(outdir+str(eventno), 
             rechit_e.copy(), rechit_x.copy(), rechit_y.copy(), rechit_z.copy(),is_track.copy(),
             truthasso.copy(),truthenergy.copy(),
             truthX.copy(),truthY.copy(),truthZ.copy(),
             truthdirX.copy(),truthdirY.copy(),truthdirZ.copy(),
             make_movie,
             show=show,
             real_truth_energy=real_truth_energy.copy())
    
    
    print('>>>>>>>>>>>>>> plotting eta/phi')
    makePlot(outdir+"etaphir_"+str(eventno), 
             rechit_e.copy(), rechit_eta.copy(), rechit_phi.copy(), rechit_r.copy(),is_track.copy(),
             truthasso.copy(),truthenergy.copy(),
             truthEta.copy(),truthPhi.copy(),truthR.copy(),
             truthdirEta.copy(),truthdirPhi.copy(),truthdirR.copy(),
             make_movie,
             False,
             True,
             False,
             show=show,
             real_truth_energy=real_truth_energy)
    
    print('>>>>>>>>>>>>>> plotting ticl')
    makePlot(outdir+"ticl_"+str(eventno), 
             rechit_e.copy(), rechit_x.copy(), rechit_y.copy(), rechit_z.copy(),is_track.copy(),
             ticlAsso.copy(),truthenergy.copy(),
             truthX.copy(),truthY.copy(),truthZ.copy(),
             truthdirX.copy(),truthdirY.copy(),truthdirZ.copy(),
             make_movie,
             show=show,
             real_truth_energy=real_truth_energy.copy())
        

    return True
    
if args.show or args.no_parallel:
    for e in range(nevents):
        worker(e, show=args.show)
    exit()

from multiprocessing import Pool
p = Pool()
print(p.map(worker, range(nevents)))

#exit()
#for e in range(nevents):
#    worker(e)
    
    
    