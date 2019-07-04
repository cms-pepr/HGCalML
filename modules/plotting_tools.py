
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

class base_plotter(object):
    def __init__(self):
        self.output_file=""
        self.parallel=False
        self.interactive=False
        self.data=None
    
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
        

class plotter_3d(base_plotter):
    def __init__(self, output_file="", parallel=False, interactive=False):
        base_plotter.__init__(self)
        self.output_file=output_file
        self.parallel=parallel
        self.interactive=interactive
        self.data=None
        
        
    
    
    def plot3d(self, e_scaling='sqrt', cut=None):
        
        if not self._check_dimension(3):
            print(self.data)
            raise Exception("plot3d: no 3D data")

        x, y, z, e, c = self.data['x'], self.data['y'], self.data['z'], self.data['e'], self.data['c']

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
        size_scaling *= 20.
        
        #c = size_scaling #/=np.min(c)
        
        ax.scatter(xs=xs, ys=ys, zs=zs, c=c, s=size_scaling, cmap='YlOrBr')
        fig.savefig(self.output_file)
        if self.interactive:
            plt.show()
        plt.close()
        
        
            
            
        