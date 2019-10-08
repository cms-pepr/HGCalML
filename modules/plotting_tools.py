
'''
This file! is supposed to contain only plotting functionality that
does not rely on more python packages than matplotlib, numpy and pickle.
The idea is that the plotters can also be used locally in an easy way

'''

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from matplotlib import cm
import os
from multiprocessing import Pool
import random
import glob

def create_max_color_diff(col_list):
    ncol=len(col_list)
    idxa=0
    idxb=int(float(ncol)/2.+0.51)
    out=[]
    uuidx=0
    while len(out)<len(col_list):
        if uuidx==0:
            out.append(col_list[idxa])
            idxa+=1
            uuidx=1
        else:
            out.append(col_list[idxb])
            idxb+=1
            uuidx=0
    return out
        
        
    

class base_plotter(object):
    def __init__(self):
        self.output_file=""
        self.parallel=False
        self.interactive=False
        self.data=None
        self.fig=None
    
    def save_binary(self, outfilename):
        with open(outfilename,'w') as outfile:
            pickle.dump(self.output_file,outfile)
            pickle.dump(self.parallel,outfile)
            pickle.dump(self.interactive,outfile)
            pickle.dump(self.data,outfile)
        
    def load_binary(self, infilename):
        with open(infilename,'r') as infile:
            self.output_file = pickle.load(infile)
            self.parallel    = pickle.load(infile)
            self.interactive = pickle.load(infile)
            self.data        = pickle.load(infile)
    
    
    def set_data(self, x, y, z=None, e=None, c=None):
        self.data={'x' : x,
                   'y' : y,
                   'z' : z,
                   'e' : e,
                   'c' : c}
        
    def _check_dimension(self,ndims):
        if self.data is None:
            return False
        x, y, z, e, c = self.data['x'], self.data['y'], self.data['z'], self.data['e'], self.data['c']
        if ndims>=1:
            if y is None:
                return False
        if ndims>=2:
            if z is None:
                return False    
        if ndims>=3:
            if e is None:
                return False
        return True
        
    def save_image(self):
        if self.fig is None:
            return
        self.fig.savefig(self.output_file+'.pdf')
        
    def reset(self):
        plt.close()
        
class plotter_2d(base_plotter):
    def __init__(self, output_file="", parallel=False, interactive=False):
        base_plotter.__init__(self)
        self.output_file=output_file
        self.parallel=parallel
        self.interactive=interactive
        self.data=None
        self.marker_scale=1.
        
    def plot2d(self, ax = None):
        if not self._check_dimension(2):
            print(self.data)
            raise Exception("plot2d: no 2D data")
        
        x, y, z, e, c = self.data['x'], self.data['y'], self.data['z'], self.data['e'], self.data['c']

        
        ax.scatter(x=x, y=y, c=z)
        
        self.fig = None
        return ax

class plotter_3d(base_plotter):
    def __init__(self, output_file="", parallel=False, interactive=False):
        base_plotter.__init__(self)
        self.output_file=output_file
        self.parallel=parallel
        self.interactive=interactive
        self.data=None
        self.marker_scale=1.
        
    
    
    def plot3d(self, e_scaling='sqrt', cut=None, ax=None):
        
        if not self._check_dimension(3):
            print(self.data)
            raise Exception("plot3d: no 3D data")

        x, y, z, e, c = self.data['x'], self.data['y'], self.data['z'], self.data['e'], self.data['c']

        fig=None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        #switch for standard CMS coordinates
        
        zs = np.reshape(x,[1,-1])
        ys = np.reshape(y,[1,-1])
        xs = np.reshape(z,[1,-1])
        es = np.reshape(e,[1,-1])
        #flattened_sigfrac = np.reshape(truth_list[0][:,:,:,0],[1,-1])
        #ax.set_axis_off()
        if e_scaling is not None and e_scaling == 'sqrt':
            es = np.sqrt(np.abs(e))
        else:
            es=e
        if c is None:
            c=e
            c/=np.min(c)
            c=np.log(np.log(np.log(np.log(e+1)+1)+1)+1) #np.log(np.log(np.log(es+1)+1)+1)
            
            
        size_scaling = e
        #size_scaling /= np.max(size_scaling)
        #size_scaling -= np.min(size_scaling)-0.01
        #size_scaling = np.exp(size_scaling*5.)
        size_scaling /=  np.max(size_scaling)
        size_scaling *= 40.
        
        #c = size_scaling #/=np.min(c)
        ax.view_init(30, 130)
        ax.scatter(xs=xs, ys=ys, zs=zs, c=c, s=self.marker_scale*size_scaling)
        
        if self.interactive:
            plt.show()
            
        self.fig = fig
        return ax
        
class plotter_fraction_colors(plotter_3d):
    def __init__(self,**kwargs ):
        plotter_3d.__init__(self,**kwargs)
        self.colorscheme='rainbow'
        self.randomise_color=False
        self.rseed=4
        self.gray_noise=True
        self.highlight_noise=False
        self.noise_color=(0.,0.,0.,.1)
        self.not_gray_interval=0.1
        
        
    def set_data(self, x, y, z, e, fractions):
        marker_colors = self.make_simcluster_marker_colours(fractions)
        if  self.highlight_noise:
            sel =  np.all(marker_colors == self.noise_color, axis=-1)
            e [sel]*=10.
        plotter_3d.set_data(self,x,y,z,e,marker_colors)
        
    def make_simcluster_marker_colours(self,truth_per_event):
    
        n_simclusters = truth_per_event.shape[1]
        cmap = cm.get_cmap(self.colorscheme)  # type: matplotlib.colors.ListedColormap
        
        scaler = cmap.N / n_simclusters -1
        
        color_map = [cmap(scaler*i) for i in range(n_simclusters)]#cmap.colors  # type: list
        
        max_sim = np.array(np.argmax(truth_per_event, axis=1),dtype='int')
        
        marker_colors = np.array([self.noise_color for i in range(len(truth_per_event))])
        
        if len(color_map)<n_simclusters:
            print('colour map too small ',len(color_map), 'for', n_simclusters, 'simclusters')
        
        select = [i for i in range(n_simclusters)]
        if self.randomise_color:
            if self.rseed is not None:
                random.seed(self.rseed)
            random.shuffle(select) 
        else:
            select = create_max_color_diff(select)
        
        #truth_per_event: [[ f,f,f,f], [f,f,f,f,]] 
        sum_truthfrac = np.abs(np.sum(truth_per_event, axis=-1))
        with_simcluster = np.abs(sum_truthfrac-1.) < self.not_gray_interval
        
        #can be vectorised FIXME
        for i in range(truth_per_event.shape[0]):
            if with_simcluster[i] or not self.gray_noise:
                for j in range(truth_per_event.shape[1]):
                    if max_sim[i] == int(j):
                        marker_colors[i] = color_map[select[j]]
   
        return marker_colors
                  
class movie_maker(object):
    def __init__(self, plotter, output_file, fullround=True, prefix="mm_", silent=True):
        self.plotter=plotter
        self.prefix=prefix
        self.glob_counter=0.
        self.fullround=fullround
        self.output_file=output_file
        self.silent=silent
       
        
    def make_movie(self):
        #return
        was_interactive = self.plotter.interactive
        self.plotter.interactive = False
        ax = self.plotter.plot3d()
        os.system('mkdir -p '+self.output_file)
        all_prefix=self.output_file+'/'+self.prefix+'_'
        self.glob_counter=0
        def worker(angle_in):
            if not self.silent:
                print(self.glob_counter)
            outputname = all_prefix+str(self.glob_counter).rjust(10, '0')+'.png'
            self.glob_counter+=1
            while angle_in>=360: angle_in-=360
            while angle_in<=-360: angle_in-=360
            ax.view_init(30, angle_in)
            self.plotter.fig.savefig(outputname, dpi=300)
            #plt.close()
            
        from multiprocessing import Pool
        angles = [float(i) for i in range(360)] 
        
        
        for a in angles:
            worker(a)
        
        os.system('ffmpeg -r 20 -f image2  -i '+ all_prefix +'%10d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 '+ all_prefix +'movie.mp4')
        os.system('rm -f '+ all_prefix +'*.png')
        self.plotter.interactive = was_interactive  
        
class snapshot_movie_maker_Nplots(object):
    '''
    creates 4 plots:
    truth, prediction
    coordinatesA, coordinatesB
    '''
    def __init__(self, output_file, nplots=4, glob_counter=0, ignore_existing_files=False, 
                 plottertype=plotter_fraction_colors):
        self.output_file=output_file
        os.system('mkdir -p '+output_file)
        self.tmp_out_prefix=output_file+'/tmp/'
        os.system('mkdir -p '+self.tmp_out_prefix)
        self.glob_counter=glob_counter
        self.offset_counter=0
        if nplots>4:
            raise Exception("snapshot_movie_maker_Nplots: maximum 4 plots")
        self.plotters=[plottertype( interactive=False ) for i in range(nplots)]
        for i in range(len(self.plotters)):
            self.plotters[i].gray_noise=True
        self.plot_data={}
        self.rjust=10
        
        #check if there are already files, and set offset FIXME
        if False and not ignore_existing_files:
            previous = glob.glob(self.tmp_out_prefix+"/*.png")
            numbers = [int(p[0:-4]) for p in previous]
            self.offset_counter= max(numbers)
            print(self.offset_counter)
            
        
    #fast, don't calcualte anything here
    def set_plot_data(self, plt_idx, x, y, z, e, fractions):
        self.plot_data[str(plt_idx)] = [x, y, z, e, fractions]
        
    def reset(self):
        self.plot_data={}
        for plt_idx in range(len(self.plotters)):
            self.plotters[plt_idx].reset()
        plt.close()
            
    def make_snapshot(self):
        x_idx=[0,0,1,1]
        y_idx=[0,1,0,1] 
        
        plt.rcParams.update({'axes.labelsize': 'small'})
        fig, axs = None, None
        if len(self.plotters)>2:
            fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
        if len(self.plotters)<=2:
            fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection='3d'), figsize=matplotlib.figure.figaspect(3./8.))
            
            
        for plt_idx in range(len(self.plotters)):
            self.plotters[plt_idx].reset()
            self.plotters[plt_idx].set_data(self.plot_data[str(plt_idx)][0], 
                                            self.plot_data[str(plt_idx)][1],
                                            self.plot_data[str(plt_idx)][2],
                                            self.plot_data[str(plt_idx)][3],
                                            self.plot_data[str(plt_idx)][4])
            if len(self.plotters)>2:
                ax = axs[x_idx[plt_idx]][y_idx[plt_idx]]
            else:
                ax = axs[plt_idx]
            self.plotters[plt_idx].plot3d(ax=ax)
            angle_in=self.glob_counter+80.
            while angle_in>=360: angle_in-=360
            while angle_in<=-360: angle_in-=360
            ax.view_init(30, angle_in)
        outputname = self.tmp_out_prefix+str(self.glob_counter+self.offset_counter).rjust(self.rjust, '0')+'.png'
        fig.savefig(outputname, dpi=300)
        plt.close()    
        
    def increment_counter(self):
        self.glob_counter+=1 
        
    def end_job(self):
        os.system('ffmpeg -r 20 -f image2  -i '+ self.tmp_out_prefix +'$w%10d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 '+ self.output_file +'_movie.mp4')
        os.system('rm -f '+ self.tmp_out_prefix +'*.png')
        
        
        
        
        
        
    