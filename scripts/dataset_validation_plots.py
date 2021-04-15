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
parser.add_argument('-p', help="PDF file path (will be ignored in validate mode)")
parser.add_argument('-n', help="Number of events to produce dataset stats pdf on", default="50")
parser.add_argument('--validate', dest='validate', action='store_true')
parser.set_defaults(validate=False)


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

        if len(x)==0:
            x = np.zeros((1000))
            textstr = 'Error: size zero array.'
        else:
            textstr = 'Min: %f\nMax: %f' % (np.min(x), np.max(x))

        if len(x) > 1000000 and y is None:
            index = np.random.choice(len(x), 1000000, replace=False)
            x = np.array(x)[index]

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
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        pdf.savefig()
            
    def write_to_pdf(self):  
        self._make_hists()
          
            
pdf = PdfPages(args.p)
plotter=plotter_class(pdf) 

print('reading events...')

d = get_event_and_make_dict(True)#just to print the options
print('accessible variables:','\n=================')
plot_variables = d.keys()
for k in d.keys():
    print(k)
print('==============')


override_settings = dict()
override_settings['recHitEnergy'] = (True, [0,5], [])
override_settings['recHitTime'] = (True, [0, 10], [])

additional = dict()
additional['recHitZ-abs'] = ('recHitZ', True, [], [], True)


def validate_event(event_dict):
    scidxs, simcluster_selection = np.unique(event_dict['truthHitAssignementIdx'], return_index=True)
    simcluster_selection = simcluster_selection[scidxs >= 0]

    if np.min(event_dict['truthHitAssignementIdx']) < -1:
        print("VALIDATE ERROR: truthHitAssignementIdx has value <-1")

    if np.min(event_dict['truthHitAssignedEnergies'][event_dict['truthHitAssignementIdx'] >= 0]) <= 0:
        print("VALIDATE ERROR: truthHitAssignedEnergies has value <= 0")

    # TODO: Add more checks later here


N = int(args.n)
if args.validate:
    for i in tqdm.tqdm(range(gen.getNBatches())):#500 events or less
        d = get_event_and_make_dict()
        validate_event(d)
else:
    for i in tqdm.tqdm(range(min(N, gen.getNBatches()))):#500 events or less

        d = get_event_and_make_dict()
        # JK: now we can use the numpy arrays explicitly, and we know exactly what we put
        # in each plot, also passing matplotlib kwargs can make our life easier

        scidxs,simcluster_selection = np.unique(d['truthHitAssignementIdx'],return_index=True)
        simcluster_selection = simcluster_selection[scidxs >= 0]

        plotter.add("# of truth showers", len(simcluster_selection),
                    log=True, xlabel='# truth showers', ylabel='Events', color='tab:green')

        plotter.add("Num vertices", len(d['truthHitAssignementIdx']),
                    log=True, xlabel='# vertices', ylabel='Events', color='tab:green')


        for k in plot_variables:
            if k not in override_settings:
                plotter.add(k, np.array(d[k]).flatten(),
                            log=False, xlabel=k, ylabel='Frequency')
            else:
                plotter.add(k, np.array(d[k]).flatten(),
                            log=override_settings[k][0], xlabel=k, ylabel='Frequency', xlim=override_settings[k][1], ylim=override_settings[k][2])

        for k in additional:
            x = d[additional[k][0]]
            if additional[k][4]:
                x = np.abs(x)
            plotter.add(k, np.array(x).flatten(),
                        log=additional[k][1], xlabel=k, ylabel='Frequency', xlim=additional[k][2], ylim=additional[k][3])


        depvstrue = d['truthHitAssignedDepEnergies'][simcluster_selection]/ \
                    (d['truthHitAssignedEnergies'][simcluster_selection]+1e-6)
        depvstrue = np.where(d['truthHitAssignedEnergies'][simcluster_selection]==0, 1, depvstrue)

        plotter.add("Deposted over true energy vs. eta", x = depvstrue,
                    y = np.abs(d['truthHitAssignedEta'][simcluster_selection]),
                    xlabel='$E_{dep}/E_{true}$', ylabel='Truth |$\eta$|',
                    xlim=[-0.2,2]
                    )

        plotter.add("Deposited energy over z", x = d['truthHitAssignedZ'][simcluster_selection],
                    y = d['truthHitAssignedEnergies'][simcluster_selection],
                    xlabel='truthHitAssignedZ', ylabel='truthHitAssignedEnergies'
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





    

