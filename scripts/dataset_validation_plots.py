#!/usr/bin/env python3


import tensorflow as tf
import tqdm
from datastructures import TrainData_NanoML
from DeepJetCore.DataCollection import DataCollection
from importlib import reload
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


from argparse import ArgumentParser
parser = ArgumentParser(
    'Dataset validation hplots script')
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
#gpus = tf.config.list_physical_devices('GPU')
gpus=0
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
    allfeat, _, epoch = get(reset_after)
    #print(len(feat), len(truth))
    # truth = truth[:, 0, :]
    feat,  truth_sid, truth_energy, truth_pos, truth_time, truth_particle_id,truth_spectator, truth_fully_contained, row_splits = td.interpretAllModelInputs(allfeat)
    all_dict = td.createFeatureDict(feat)
    all_dict.update( td.createTruthDict(allfeat) )
    return all_dict


def calc_energy_weights(t_energy):
    lower_cut = 0.5
    w = tf.where(t_energy > 10., 1., ((t_energy-lower_cut) / 10.)*10./(10.-lower_cut))
    return tf.nn.relu(w)


class plotter_class(object):
    def __init__(self,pdf):
        self.datadictx={}
        self.datadictweights={}
        self.datadicty={}
        self.modedict={}
        self.kwargsdict={} 
        self.pdf=pdf 
    
    def add(self, plotname, x, y=None, weight=None, log=False, xlabel='', ylabel='',xlim=[],ylim=[],  **kwargs):
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
            if weight is not None:
                self.datadictweights[plotname]=[]
            else:
                self.datadictweights[plotname]=None
            self.modedict[plotname]=(log,xlabel,ylabel,xlim,ylim)
            self.kwargsdict[plotname]=kwargs
            
            
        self.datadictx[plotname] += x.tolist() if isinstance(x, np.ndarray) else [x]
        if y is not None:
            self.datadicty[plotname] += y.tolist() if isinstance(y, np.ndarray) else [y]
        if weight is not None:
            self.datadictweights[plotname] += weight.tolist() if isinstance(weight, np.ndarray) else [weight]
        
    def _make_hists(self):
        for dx,dy,dweights,m,k in zip(self.datadictx.values(), self.datadicty.values(), self.datadictweights.values(), 
                         self.modedict.values(), self.kwargsdict.values()):
            self._do_a_hist(dx,dy,dweights,log=m[0],xlabel=m[1], ylabel=m[2],xlim=m[3],ylim=m[4],**k)
        
    

    def _do_a_hist(self, x, y=None,weights=None, log=False, xlabel='', ylabel='',
                   xlim=[],ylim=[],
                   **kwargs):
        pdf=self.pdf
        

        if len(x)==0:
            x = np.zeros((1000))
            textstr = 'Error: size zero array.'
        else:
            textstr = 'Min: %.2f\nMax: %.2f' % (np.min(x), np.max(x))

       # if len(x) > 1000000 and y is None:
       #     index = np.random.choice(len(x), 1000000, replace=False)
       #     x = np.array(x)[index]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if log:
            ax.set_yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        xnp=np.squeeze(np.array(x))
        weightsnp=np.squeeze(np.array(weights))
        if len(xlim)==2 and weights is None:
            xnp = np.where(xnp < xlim[0], xlim[0], xnp)
            xnp = np.where(xnp > xlim[1], xlim[1], xnp)
        if y is None:
            if weights is None:
                ax.hist(xnp,**kwargs)
            else:
                kwargs['color'] = 'dodgerblue'
                ax.hist(xnp,**kwargs,histtype='step',label='original')
                kwargs['color'] = 'red'
  
                #FIXME
                weightsnp = np.ones_like(xnp)

                ax.hist(xnp,weights=weightsnp,**kwargs,histtype='step',label='weighted')
                ax.legend(loc="upper right")
        else:
            ynp=np.squeeze(np.array(y))
            if len(ylim) ==2:
                ynp = np.where(ynp < ylim[0], ylim[0], ynp)
                ynp = np.where(ynp > ylim[1], ylim[1], ynp)
            ax.hist2d(xnp,ynp,**kwargs)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        pdf.savefig()
            
    def write_to_pdf(self):  
        self._make_hists()
          
            
pdf = PdfPages(args.p)
plotter=plotter_class(pdf) 

print('reading events...')

d = get_event_and_make_dict(True)#just to print the options
print('accessible variables:','\n=================')

#update plotting variables to only the relevant ones :


plot_variables = d.keys()
for k in d.keys():
    print(k)




print('==============')


override_settings = dict()
override_settings['recHitEnergy'] = (True, [0,5], [], 50)
override_settings['recHitTime'] = (True, [0, 10], [], 50)
override_settings['recHitEta'] = (True, [], [], 50)
override_settings['truthHitAssignedEta'] = (True, [], [], 200)

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


pid_counter = Counter()

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

        front_face_z = 323

        noise_filter = (d['truthHitAssignementIdx'] > -1)
        hgcal_front_face_filter = d['truthHitFullyContainedFlag'] # < - on front, > not on front
        #pid_filter = (abs(d['truthHitAssignedPIDs'])!=2112)
        #eta_filter = (abs(d['truthHitAssignedEta'])<1.5)
        #energy_true = (d['truthHitAssignedEnergies'] > 0)
        energy_true = (d['truthHitAssignedEnergies'] > 0.)
        energy_depvstrue = 0. #d['truthHitAssignedDepEnergies']/ \
                    #(d['truthHitAssignedEnergies']+1e-6)
        energy_depvstrue = np.where(d['truthHitAssignedEnergies']<=1e-6, -1, energy_depvstrue)
        energy_depvstrue_energy_filter = (energy_depvstrue > 0.) 
        
        selection_filter = np.logical_and(np.logical_and(np.logical_and(noise_filter , hgcal_front_face_filter ), energy_true  ), energy_depvstrue_energy_filter )
        selection_filter = selection_filter.flatten()

        for key in d.keys():
            d[key] = d[key][selection_filter] #apply all selections 
        #d['truthHitAssignedEnergies'] = np.where(d['truthHitAssignedEnergies']>3, d['truthHitAssignedEnergies'], d['truthHitAssignedDepEnergies'])
        
        _,simcluster_sel,recHits_counts = np.unique(d['truthHitAssignementIdx'],return_index=True,return_counts=True)#shower based only (not hits)

 
        plotter.add("Num rec hit counts", x=recHits_counts,
                    log=True, xlabel='# rec hit counts', ylabel='Events', color='dodgerblue') 

        energy_depvstrue = 1.#d['truthHitAssignedDepEnergies']/ \
                    #(d['truthHitAssignedEnergies']+1e-6)
        energy_depvstrue = np.where(d['truthHitAssignedEnergies']<=1e-6, -1, energy_depvstrue)
        energy_depvstrue = energy_depvstrue[simcluster_sel] #showers only 

        plotter.add("# of truth showers", len(simcluster_sel),
                    log=True, xlabel='# truth showers', ylabel='Events', color='dodgerblue')


        for k in plot_variables:
            if k not in override_settings:
                plotter.add(k, np.array(d[k][simcluster_sel]).flatten(),
                            log=False, xlabel=k, ylabel='Frequency', bins=50, color='dodgerblue')
            else:
                plotter.add(k, np.array(d[k][simcluster_sel]).flatten(),
                            log=override_settings[k][0], xlabel=k, ylabel='Frequency', xlim=override_settings[k][1], ylim=override_settings[k][2], bins=override_settings[k][3],color='dodgerblue')

        for k in additional:
            x = d[additional[k][0]][simcluster_sel]
            if additional[k][4]:
                x = np.abs(x)
            plotter.add(k, np.array(x).flatten(),
                        log=additional[k][1], xlabel=k, ylabel='Frequency', xlim=additional[k][2], ylim=additional[k][3],color='dodgerblue',bins=100)



        

        

        plotter.add("Deposted over true energy", x = energy_depvstrue,
                    xlabel='$E_{dep}/E_{true}$', ylabel='Events',
                    log=True,
                    xlim=[-2, 7],
                    bins=200, color='dodgerblue'
                    )
        
        plotter.add("Deposted over true energy, weighted", x = energy_depvstrue.flatten(), weight=calc_energy_weights(d['truthHitAssignedEnergies'][simcluster_sel]).numpy().flatten(),
                    xlabel='$E_{dep}/E_{true}$ weighted', ylabel='Events',
                    log=True,
                    bins=np.linspace(-2,7,200), color='dodgerblue'
                    )



        x2_y2 = np.sqrt(d['truthHitAssignedX'][simcluster_sel]**2+d['truthHitAssignedY'][simcluster_sel]**2)
        
        plotter.add(r'Truth $\sqrt{x^{2}+y^{2}}$', x = x2_y2,
                    xlabel=r'Truth $\sqrt{x^{2}+y^{2}}$', ylabel='Frequency',
                    bins=50,color='dodgerblue')

        pid_counter+=Counter(d['truthHitAssignedPIDs'][simcluster_sel].flatten())


    print('Truth Hit Assigned PID Counter : ')
    count_once = []
    for value, count in pid_counter.most_common():
        if value in count_once :continue
        print('{}'.format(int(value)), count, end='')
        count_once.append(-1*value)
        if pid_counter.get(int(-1*value))!=None:
            print('/{}'.format(int(pid_counter[int(-1*value)])))
        else: print()

    print('writing hplots...')
    plotter.write_to_pdf()
    pdf.close()
    print('done')



    

