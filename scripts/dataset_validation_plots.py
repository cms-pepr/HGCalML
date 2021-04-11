#!/usr/bin/env python3


import tensorflow as tf
import tqdm
from datastructures import TrainData_NanoML
from DeepJetCore.DataCollection import DataCollection
from importlib import reload
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser = ArgumentParser(
    'Dataset validation plots script')
parser.add_argument('-d', help="Data collection file")
parser.add_argument('-p', help="PDF file path")


args = parser.parse_args()
dc = DataCollection(args.d)
td = dc.dataclass() #this is actually saved
#JK: this combination enforces one event per batch, then the extra row split loop is not needed
batchsize = 1
dc.setBatchSize(batchsize)
print("Invoking generator")
gen = dc.invokeGenerator()
gen.setSkipTooLargeBatches(False)

# gen.setBuffer(td)
print("n batches")
n_batches = gen.getNBatches()
print(n_batches)
print("probably ready")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
counter = 0
epoch = 0

def get(reset_after=False):
    global counter, n_batches, epoch  #why do you use global variables to often? there is no need here
    try:
        feat, truth = next(gen.feedNumpyData())  # this is  [ [features],[truth],[None] ]
    except:
        gen.prepareNextEpoch()
        epoch += 1
        feat, truth = next(gen.feedNumpyData())  # this is  [ [features],[truth],[None] ]
        counter = 0
    if reset_after:
        gen.prepareNextEpoch()
    return feat, truth, epoch


def get_event_and_make_dict(reset_after=False):
    feat, truth, epoch = get(reset_after)
    #print(len(feat), len(truth))
    truth = truth[0]
    # truth = truth[:, 0, :]
    feat,  truth_sid, truth_energy, truth_pos, truth_time, truth_particle_id, row_splits = td.interpretAllModelInputs(feat)
    all_dict = td.createFeatureDict(feat)
    all_dict.update( td.createTruthDict(truth, truth_sid) )
    return all_dict


class plotter_class(object):
    def __init__(self,pdf):
        self.datadictx={}
        self.datadicty={}
        self.modedict={}
        self.kwargsdict={} 
        self.pdf=pdf 
    
    def add(self, plotname, x, y=None, log=False, xlabel='', ylabel='',xlim=[],ylim=[],  **kwargs):
        newplot=False
        try:
            self.datadictx[plotname]
        except: #new plot
            newplot=True
            self.datadictx[plotname]=[]
            if y is not None:
                self.datadicty[plotname]=[]
            else:
                self.datadicty[plotname]=None
            self.modedict[plotname]=(log,xlabel,ylabel,xlim,ylim)
            self.kwargsdict[plotname]=kwargs
            
            
        self.datadictx[plotname] += x.tolist() if isinstance(x, np.ndarray) else [x]
        if y is not None:
            self.datadicty[plotname] += y.tolist() if isinstance(y, np.ndarray) else [y]
        
    def _make_hists(self):
        for dx,dy,m,k in zip(self.datadictx.values(), self.datadicty.values(), 
                         self.modedict.values(), self.kwargsdict.values()):
            self._do_a_hist(dx,dy,log=m[0],xlabel=m[1], ylabel=m[2],xlim=m[3],ylim=m[4],**k)
        
    

    def _do_a_hist(self, x, y=None, log=False, xlabel='', ylabel='',
                   xlim=[],ylim=[],
                   **kwargs):
        pdf=self.pdf

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if log:
            ax.set_yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        xnp=np.squeeze(np.array(x))
        if len(xlim)==2:
            xnp = np.where(xnp < xlim[0], xlim[0], xnp)
            xnp = np.where(xnp > xlim[1], xlim[1], xnp)
        if y is None:
            ax.hist(xnp,**kwargs)
        else:
            ynp=np.squeeze(np.array(y))
            if len(ylim) ==2:
                ynp = np.where(ynp < ylim[0], ylim[0], ynp)
                ynp = np.where(ynp > ylim[1], ylim[1], ynp)
            ax.hist2d(xnp,ynp,**kwargs)
        textstr = 'Min: %f\nMax: %f' % (np.min(x), np.max(x))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        pdf.savefig()
            
    def write_to_pdf(self):  
        self._make_hists()
          
            
pdf = PdfPages(args.p)
plotter=plotter_class(pdf) 

print('reading events...')

########### loop events here and make plots
#### plots need to be defined only once here. 

#### JK -> SR: I have seen this a few times in your scripts, and it makes them hard 
####           to use for others (and for yourself at some point).
####           Really try to avoid needing to know some definition from before to 
####           get back the data from somewhere else.

#### some examples given below
#### if limits in x and y are defined, over/underflow are merged into last bin
#### kwargs other than x, y=None, log=False, xlabel='', ylabel='',xlim=[],ylim=[]
#### are passed through to matplotlib

d = get_event_and_make_dict(True)#just to print the options
print('accessible variables:','\n=================')
for k in d.keys():
    print(k)
print('==============')

for i in tqdm.tqdm(range(min([500,gen.getNBatches()]))):#500 events or less
    
    d = get_event_and_make_dict()
            
    
    # JK: now we can use the numpy arrays explicitly, and we know exactly what we put
    # in each plot, also passing matplotlib kwargs can make our life easier
    
    scidxs,simcluster_selection = np.unique(d['truthHitAssignementIdx'],return_index=True)
    simcluster_selection = simcluster_selection[scidxs >= 0]
    
    plotter.add("# of truth showers", len(simcluster_selection), 
                log=True, xlabel='# truth showers', ylabel='Events', color='tab:green')

    
    depvstrue = d['truthHitAssignedDepEnergies'][simcluster_selection]/ \
                (d['truthHitAssignedEnergies'][simcluster_selection]+1e-6)
    depvstrue = np.where(d['truthHitAssignedEnergies'][simcluster_selection]==0, 1, depvstrue)
    
    plotter.add("Deposted over true energy vs. eta", x = depvstrue,
                y = np.abs(d['truthHitAssignedEta'][simcluster_selection]),
                xlabel='$E_{dep}/E_{true}$', ylabel='Truth |$\eta$|',
                xlim=[-0.2,2]
                )
    
    plotter.add("Deposted over true energy", x = depvstrue,
                xlabel='$E_{dep}/E_{true}$$', ylabel='Events',
                log=True,
                xlim=[-0.2,2],
                bins=40
                )


    

print('writing plots...')
plotter.write_to_pdf()    
pdf.close()
print('done')
    







exit()

#OLD JK: You can see below that in order to make one plot, you need to 
#        define the right key in 3 different places simultaneously.


def make_a_pdf(dict): #be careful when overwriting builtins (e.g. dict)
    
    do_a_hist(np.array(dict['num_vertices']), xlabel='num_vertices', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_energy']), xlabel='rechit_energy', ylabel='Frequency', log=True, pdf=pdf)
    do_a_hist(np.array(dict['rechit_eta']), xlabel='rechit_eta', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_theta']), xlabel='rechit_theta', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_R']), xlabel='rechit_R', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_x']), xlabel='rechit_x', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_y']), xlabel='rechit_y', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_z']), xlabel='rechit_z', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['rechit_z']), xlabel='rechit_z_abs', ylabel='Frequency', log=False, pdf=pdf, abs=True)
    do_a_hist(np.array(dict['rechit_time']), xlabel='rechit_time', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_idx']), xlabel='truth_shower_idx', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_energy']), xlabel='truth_shower_energy', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_x']), xlabel='truth_shower_x', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_y']), xlabel='truth_shower_y', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_z']), xlabel='truth_shower_z', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_z']), xlabel='truth_shower_z_abs', ylabel='Frequency', log=False, pdf=pdf, abs=True)
    do_a_hist(np.array(dict['truth_shower_dir_x']), xlabel='truth_shower_dir_x', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_dir_y']), xlabel='truth_shower_dir_y', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_dir_z']), xlabel='truth_shower_dir_z', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_eta']), xlabel='truth_shower_eta', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_phi']), xlabel='truth_shower_phi', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_t']), xlabel='truth_shower_t', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_dir_eta']), xlabel='truth_shower_dir_eta', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_dir_r']), xlabel='truth_shower_dir_r', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_energy_dep']), xlabel='truth_shower_energy_dep', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['ticl_shower_id']), xlabel='ticl_shower_id', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['ticl_shower_energy']), xlabel='ticl_shower_energy', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['truth_shower_pid']), xlabel='truth_shower_pid', ylabel='Frequency', log=False, pdf=pdf)
    do_a_hist(np.array(dict['num_truth_showers']), xlabel='num_truth_showers', ylabel='Frequency', log=False, pdf=pdf)
    pdf.close()
       
def run(get): #why do you need to pass the function here?
    plotter = dict()
    
    # JK: it is not particularly maintainable if you need to define every plot name/identifier twice.
    # this is a source of bugs and makes it not particulaly user-friendly to add new plots
    
    plotter['num_vertices'] = []
    plotter['rechit_energy'] = []
    plotter['rechit_eta'] = []
    plotter['rechit_theta'] = []
    plotter['rechit_R'] = []
    plotter['rechit_x'] = []
    plotter['rechit_y'] = []
    plotter['rechit_z'] = []
    plotter['rechit_time'] = []
    plotter['truth_shower_idx'] = []
    plotter['truth_shower_energy'] = []
    plotter['truth_shower_x'] = []
    plotter['truth_shower_y'] = []
    plotter['truth_shower_z'] = []
    plotter['truth_shower_dir_x'] = []
    plotter['truth_shower_dir_y'] = []
    plotter['truth_shower_dir_z'] = []
    plotter['truth_shower_eta'] = []
    plotter['truth_shower_phi'] = []
    plotter['truth_shower_t'] = []
    plotter['truth_shower_dir_eta'] = []
    plotter['truth_shower_dir_r'] = []
    plotter['truth_shower_energy_dep'] = []
    plotter['ticl_shower_id'] = []
    plotter['ticl_shower_energy'] = []
    plotter['truth_shower_pid'] = []
    plotter['num_truth_showers'] = []
    for i in range(50):
        print("processing ", i)
        td = TrainData_NanoML()
        feat, truth, epoch = get()
        print(len(feat), len(truth))
        truth = truth[0]
        # truth = truth[:, 0, :]
        feat,  truth_sid, truth_energy, truth_pos, truth_time, truth_particle_id, row_splits = td.interpretAllModelInputs(feat)
        
        # print(row_splits)
        # print(truth_pos.shape, truth_energy.shape, truth_sid.shape, truth_particle_id.shape, feat.shape)
        # feat_dict = td.createFeatureDict(feat)
        # network_input = tf.concat((feat_dict['recHitEnergy'], feat_dict['recHitX'], feat_dict['recHitY'], np.abs(feat_dict['recHitZ']), feat_dict['recHitTime'], feat_dict['recHitTheta'] ), axis=-1)
        # feat = feat[:,0,:]
        
        # JK: using the dicts here would be much more user friendly and easier to keep consistent.
        
        plotter['rechit_energy'] += feat[:, 0].tolist()
        plotter['rechit_eta'] += feat[:, 1].tolist()
        plotter['rechit_theta'] += feat[:, 3].tolist()
        plotter['rechit_R'] += feat[:, 4].tolist()
        plotter['rechit_x'] += feat[:, 5].tolist()
        plotter['rechit_y'] += feat[:, 6].tolist()
        plotter['rechit_z'] += feat[:, 7].tolist()
        plotter['rechit_time'] += feat[:, 8].tolist()
        num_vertices = (row_splits[1:] - row_splits[:-1]).tolist()
        # print(num_vertices)
        plotter['num_vertices'] += num_vertices
        # plotter['rechit_energy'] += feat[:,0].tolist()
        # plotter['rechit_x'] += feat[:,5].tolist()
        # plotter['rechit_y'] += feat[:,6].tolist()
        # plotter['rechit_z'] += feat[:,7].tolist()
        # plotter['rechit_eta'] += feat[:,1].tolist()
        
        #JK: explicit row split loop here can make things even more complicated, see solution on top
        for j in range(len(row_splits)-1):
            start = int(row_splits[j])
            end = int(row_splits[j+1])
            _ , unique_idx = np.unique(truth_sid[start:end], return_index=True)
            plotter['num_truth_showers'] += [len(unique_idx)]
            plotter['truth_shower_idx'] += (truth[start:end, 0][unique_idx]).tolist()
            plotter['truth_shower_energy'] += (truth[start:end, 1][unique_idx]).tolist()
            plotter['truth_shower_x'] += (truth[start:end, 2][unique_idx]).tolist()
            plotter['truth_shower_y'] += (truth[start:end, 3][unique_idx]).tolist()
            plotter['truth_shower_z'] += (truth[start:end, 4][unique_idx]).tolist()
            plotter['truth_shower_dir_x'] += (truth[start:end, 5][unique_idx]).tolist()
            plotter['truth_shower_dir_y'] += (truth[start:end, 6][unique_idx]).tolist()
            plotter['truth_shower_dir_z'] += (truth[start:end, 7][unique_idx]).tolist()
            plotter['truth_shower_eta'] += (truth[start:end, 8][unique_idx]).tolist()
            plotter['truth_shower_phi'] += (truth[start:end, 9][unique_idx]).tolist()
            plotter['truth_shower_t'] += (truth[start:end, 10][unique_idx]).tolist()
            plotter['truth_shower_dir_eta'] += (truth[start:end, 11][unique_idx]).tolist()
            plotter['truth_shower_dir_r'] += (truth[start:end, 12][unique_idx]).tolist()
            plotter['truth_shower_energy_dep'] += (truth[start:end, 13][unique_idx]).tolist()
            plotter['ticl_shower_id'] += (truth[start:end, 14][unique_idx]).tolist()
            plotter['ticl_shower_energy'] += (truth[start:end, 15][unique_idx]).tolist()
            plotter['truth_shower_pid'] += (truth[start:end, 16][unique_idx]).tolist()
    make_a_pdf(plotter)
run(get)