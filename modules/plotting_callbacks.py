
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL","move implementations still in use to callbacks")

import matplotlib
import matplotlib.pyplot as plt
from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot, createRandomizedColors, make_eta_phi_projection_truth_plot
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from multiprocessing import Process
import numpy as np
from datastructures import TrainData_OC
import matplotlib.gridspec as gridspec
import os
import tempfile
import random

'''
standard output is:
pred_beta, 
pred_ccoords,
pred_energy, 
pred_pos, 
pred_time, 
pred_id,
rs]+backgatheredids
'''

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

def publish(path):
    temp_name = next(tempfile._get_candidate_names())
    temp_name = self.outputfile + temp_name + '.png'
    fig.savefig(temp_name)
    cpstring = 'cp -f '
    if "@" in path:
        cpstring = 'scp '
    os.system(cpstring + temp_name + ' ' + path + '.png > /dev/null')
    os.system('rm -f ' + temp_name)  


class plotClusteringDuringTraining(PredictCallback):
    def __init__(self,
                 outputfile,
                 cycle_colors=False,
                 publish=None,
                 n_keep=3,
                 use_backgather_idx=7,#and more,
                 batchsize=300000,
                 type='png',
                 **kwargs):
        
        super(plotClusteringDuringTraining, self).__init__(function_to_apply=self.make_plot, batchsize=batchsize,**kwargs)
        self.outputfile = outputfile
        self.cycle_colors = cycle_colors
        self.n_keep = n_keep
        self.use_backgather_idx=use_backgather_idx
        self.publish = publish
        self.type=type
        ## preparation
        os.system('mkdir -p '+os.path.dirname(outputfile))
        
        #internals
        self.keep_counter = 0
        if self.td.nElements() > 1:
            raise ValueError("plotEventDuringTraining: only one event allowed")

        self.gs = gridspec.GridSpec(2, 2, height_ratios=[0.3,0.7])
        self.plot_process = None

    def make_plot(self, counter, feat, predicted, truth):
        if self.plot_process is not None:
            self.plot_process.join()

        self.keep_counter += 1
        if self.keep_counter > self.n_keep:
            self.keep_counter = 0

        self.plot_process = Process(target=self._make_plot, args=(counter, feat, predicted, truth))
        self.plot_process.start()
        # self._make_plot(counter,feat,predicted,truth)

    def _make_plot(self, counter, feat, predicted, truth):
        try:
            td = TrainData_OC()#contains all dicts
            #row splits not needed
            feats = td.createFeatureDict(feat[0])
            backgather = predicted[self.use_backgather_idx]
            truths = td.createTruthDict(truth[0])
            
            fig = plt.figure(figsize=(10, 12))
            ax = [fig.add_subplot(self.gs[0, 0], projection='3d'),
                  fig.add_subplot(self.gs[0, 1], projection='3d'),
                  fig.add_subplot(self.gs[1, :], projection='polar')    ]
            
            
            cmap = createRandomizedColors('jet', seed=truth[0].shape[0])
            
            make_original_truth_shower_plot(plt, ax[0],
                                            backgather,
                                            feats['recHitEnergy'],
                                            feats['recHitX'],
                                            feats['recHitY'],
                                            feats['recHitZ'],
                                            cmap=cmap,
                                            predBeta=np.ones_like(backgather))
            
            
            ax[0].view_init(30, rotation(counter))
            
            make_original_truth_shower_plot(plt, ax[1],
                                            truths['truthHitAssignementIdx'],
                                            feats['recHitEnergy'],
                                            feats['recHitX'],
                                            feats['recHitY'],
                                            feats['recHitZ'],
                                            cmap=cmap,
                                            predBeta=np.ones_like(backgather))
            
            ax[1].view_init(30, rotation(counter))
            
            eta = calc_eta(feats['recHitX'][:,0], feats['recHitY'][:,0], feats['recHitZ'][:,0])
            phi = calc_phi(feats['recHitX'][:,0], feats['recHitY'][:,0])
            r = calc_r(feats['recHitX'][:,0], feats['recHitY'][:,0])
            
            rgbcolor = cmap((backgather[:,0] + 1.) / (np.max(backgather[:,0]) + 1.))
            
            #polar
            ax[2].scatter(phi, feats['recHitTheta'][:,0], c=rgbcolor, s=np.log(1.+ feats['recHitEnergy'][:,0]))
            
            plt.tight_layout()
            if self.type=="png":
                fig.savefig(self.outputfile + str(self.keep_counter) + "."+self.type, dpi=400)
            else:
                fig.savefig(self.outputfile + str(self.keep_counter) + "."+self.type)
            
            fig.clear()
            plt.close(fig)
            plt.clf()
            plt.cla()
            plt.close()
            

        except Exception as e:
            print(e)
            raise e


class plotEventDuringTraining(PredictCallback):
    def __init__(self,
                 outputfile,
                 log_energy=False,
                 cycle_colors=False,
                 publish=None,
                 n_keep=3,
                 n_ccoords=2,
                 **kwargs):
        super(plotEventDuringTraining, self).__init__(function_to_apply=self.make_plot, **kwargs)
        self.outputfile = outputfile
        os.system('mkdir -p '+os.path.dirname(outputfile))
        self.cycle_colors = cycle_colors
        self.log_energy = log_energy
        self.n_keep = n_keep
        self.n_ccoords = n_ccoords
        self.keep_counter = 0
        if self.td.nElements() > 1:
            raise ValueError("plotEventDuringTraining: only one event allowed")

        self.gs = gridspec.GridSpec(2, 2)
        self.plot_process = None
        self.publish = publish

    def make_plot(self, counter, feat, predicted, truth):
        if self.plot_process is not None:
            self.plot_process.join()

        self.keep_counter += 1
        if self.keep_counter > self.n_keep:
            self.keep_counter = 0

        self.plot_process = Process(target=self._make_plot, args=(counter, feat, predicted, truth))
        self.plot_process.start()
        # self._make_plot(counter,feat,predicted,truth)

    def _make_plot(self, counter, feat, predicted, truth):

        # make sure it gets reloaded in the fork
        # doesn't really seem to help though
        # from importlib import reload
        # global matplotlib
        # matplotlib=reload(matplotlib)
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # import matplotlib.gridspec as gridspec

        # exception handling is weird for keras fit right now... explicitely print exceptions at the end
        try:
            '''
            [pred_beta, 
             pred_ccoords,
             pred_energy, 
             pred_pos, 
             pred_time, 
             pred_id
            '''
            
            td = TrainData_OC()#contains all dicts
            #row splits not needed
            feats = td.createFeatureDict(feat[0])
            truths = td.createTruthDict(truth[0])
            
            predBeta = predicted[0]
            predCCoords = predicted[1]
            predEnergy = predicted[2]
            predX = predicted[3][:,0:1]
            predY = predicted[3][:,1:2]

            
            print('predBeta',predBeta.shape)
            print('predCCoords',predCCoords.shape)
            print('predEnergy',predEnergy.shape)
            print('predX',predX.shape)
            
            

            fig = plt.figure(figsize=(10, 8))
            ax = None
            if predCCoords.shape[1] == 2:
                ax = [fig.add_subplot(self.gs[0, 0], projection='3d'),
                      fig.add_subplot(self.gs[0, 1]),
                      fig.add_subplot(self.gs[1, :])]
            elif predCCoords.shape[1] == 3:
                ax = [fig.add_subplot(self.gs[0, 0], projection='3d'),
                      fig.add_subplot(self.gs[0, 1], projection='3d'),
                      fig.add_subplot(self.gs[1, :])]

            predBeta = np.clip(predBeta, 1e-6, 1. - 1e-6)

            seed = predBeta.shape[0]
            if self.cycle_colors:
                seed += counter

            cmap = createRandomizedColors('jet', seed=seed)

            identified = make_cluster_coordinates_plot(plt, ax[1],
                                                       truths['truthHitAssignementIdx'],  # [ V  x 1] or [ V ]
                                                       predBeta,  # [ V  x 1] or [ V ]
                                                       predCCoords,
                                                       beta_threshold=0.5, distance_threshold=0.75,
                                                       beta_plot_threshold=0.01,
                                                       cmap=cmap)

            make_original_truth_shower_plot(plt, ax[0],
                                            truths['truthHitAssignementIdx'],
                                            feats['recHitEnergy'],
                                            feats['recHitX'],
                                            feats['recHitY'],
                                            feats['recHitZ'],
                                            cmap=cmap,
                                            predBeta=predBeta)

            angle_in = 10. * counter + 60.
            while angle_in >= 360: angle_in -= 360
            while angle_in <= -360: angle_in -= 360
            ax[0].view_init(30, angle_in)
            if predCCoords.shape[1] == 3:
                ax[1].view_init(30, angle_in)

            
            if self.log_energy:
                predEnergy = np.exp(predEnergy) - 1

            def calc_eta(x, y, z):
                rsq = np.sqrt(x ** 2 + y ** 2)
                return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.)

            def calc_phi(x, y):
                return np.arctan2(x, y)

            predEta = calc_eta(predX , predY,
                               np.sign(feats['recHitZ']) * 322.1)
            predPhi = calc_phi(predX , predY)

            trueEta = calc_eta(truths['truthHitAssignedX'], truths['truthHitAssignedY'] + 1e-3,
                               truths['truthHitAssignedZ'] + 1e-3)
            truePhi = calc_phi(truths['truthHitAssignedX'], truths['truthHitAssignedY'] + 1e-3)

            make_eta_phi_projection_truth_plot(plt, ax[2],
                                               truths['truthHitAssignementIdx'],
                                               feats['recHitEnergy'],
                                               calc_eta(feats['recHitX'], feats['recHitY'], feats['recHitZ']),
                                               calc_phi(feats['recHitX'], feats['recHitY']),
                                               predEta,  # data['predEta']+feats['recHitEta'],
                                               predPhi,  # data['predPhi']+feats['recHitRelPhi'],
                                               trueEta,
                                               truePhi,
                                               truths['truthHitAssignedEnergies'],
                                               predBeta,
                                               predCCoords,
                                               cmap=cmap,
                                               identified=identified,
                                               predEnergy=predEnergy
                                               )

            plt.tight_layout()
            fig.savefig(self.outputfile + str(self.keep_counter) + ".pdf")

            if self.publish is not None:
                temp_name = next(tempfile._get_candidate_names())
                temp_name = self.outputfile + temp_name + '.png'
                fig.savefig(temp_name, dpi=400)
                cpstring = 'cp -f '
                if "@" in self.publish:
                    cpstring = 'scp '
                os.system(cpstring + temp_name + ' ' + self.publish + '.png > /dev/null')
                os.system('rm -f ' + temp_name)

            fig.clear()
            plt.close(fig)
            plt.clf()
            plt.cla()
            plt.close()


        except Exception as e:
            print(e)
            raise e

