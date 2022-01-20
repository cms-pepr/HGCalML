
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL","move implementations to callbacks")
raise ImportError("MODULE",__name__,"will be removed")

import matplotlib
matplotlib.use('Agg')

from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot, createRandomizedColors, make_eta_phi_projection_truth_plot
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from index_dicts import create_index_dict, create_feature_dict, split_feat_pred
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from multiprocessing import Process
import numpy as np
import tempfile
import os
from DeepJetCore.TrainData import TrainData
from DeepJetCore.dataPipeline import TrainDataGenerator
import tensorflow as tf
import copy



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
            pred = predicted[0]
            feat = feat[0]  # remove row splits
            truth = truth[0]  # remove row splits, anyway one event

            _, pred = split_feat_pred(pred)
            data = create_index_dict(truth, pred, usetf=False, n_ccoords=self.n_ccoords)
            feats = create_feature_dict(feat)

            fig = plt.figure(figsize=(10, 8))
            ax = None
            if self.n_ccoords == 2:
                ax = [fig.add_subplot(self.gs[0, 0], projection='3d'),
                      fig.add_subplot(self.gs[0, 1]),
                      fig.add_subplot(self.gs[1, :])]
            elif self.n_ccoords == 3:
                ax = [fig.add_subplot(self.gs[0, 0], projection='3d'),
                      fig.add_subplot(self.gs[0, 1], projection='3d'),
                      fig.add_subplot(self.gs[1, :])]

            data['predBeta'] = np.clip(data['predBeta'], 1e-6, 1. - 1e-6)

            seed = truth.shape[0]
            if self.cycle_colors:
                seed += counter

            cmap = createRandomizedColors('jet', seed=seed)

            identified = make_cluster_coordinates_plot(plt, ax[1],
                                                       data['truthHitAssignementIdx'],  # [ V  x 1] or [ V ]
                                                       data['predBeta'],  # [ V  x 1] or [ V ]
                                                       data['predCCoords'],
                                                       beta_threshold=0.5, distance_threshold=0.75,
                                                       beta_plot_threshold=0.01,
                                                       cmap=cmap)

            make_original_truth_shower_plot(plt, ax[0],
                                            data['truthHitAssignementIdx'],
                                            feats['recHitEnergy'],
                                            feats['recHitX'],
                                            feats['recHitY'],
                                            feats['recHitZ'],
                                            cmap=cmap,
                                            predBeta=data['predBeta'])

            angle_in = 10. * counter + 60.
            while angle_in >= 360: angle_in -= 360
            while angle_in <= -360: angle_in -= 360
            ax[0].view_init(30, angle_in)
            if self.n_ccoords == 3:
                ax[1].view_init(30, angle_in)

            predEnergy = data['predEnergy']
            if self.log_energy:
                predEnergy = np.exp(predEnergy) - 1

            def calc_eta(x, y, z):
                rsq = np.sqrt(x ** 2 + y ** 2)
                return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.)

            def calc_phi(x, y):
                return np.arctan2(x, y)

            predEta = calc_eta(data['predX'] + feats['recHitX'], data['predY'] + feats['recHitY'],
                               np.sign(feats['recHitZ']) * 322.1)
            predPhi = calc_phi(data['predX'] + feats['recHitX'], data['predY'] + feats['recHitY'])

            trueEta = calc_eta(data['truthHitAssignedX'], data['truthHitAssignedY'] + 1e-3,
                               data['truthHitAssignedZ'] + 1e-3)
            truePhi = calc_phi(data['truthHitAssignedX'], data['truthHitAssignedY'] + 1e-3)

            make_eta_phi_projection_truth_plot(plt, ax[2],
                                               data['truthHitAssignementIdx'],
                                               feats['recHitEnergy'],
                                               calc_eta(feats['recHitX'], feats['recHitY'], feats['recHitZ']),
                                               calc_phi(feats['recHitX'], feats['recHitY']),
                                               predEta,  # data['predEta']+feats['recHitEta'],
                                               predPhi,  # data['predPhi']+feats['recHitRelPhi'],
                                               trueEta,
                                               truePhi,
                                               data['truthHitAssignedEnergies'],
                                               data['predBeta'],
                                               data['predCCoords'],
                                               cmap=cmap,
                                               identified=identified,
                                               predEnergy=predEnergy
                                               )

            plt.tight_layout()
            fig.savefig(self.outputfile + str(self.keep_counter) + ".pdf")

            if self.publish is not None:
                temp_name = next(tempfile._get_candidate_names())
                temp_name = self.outputfile + temp_name + '.png'
                fig.savefig(temp_name)
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


class plotGravNetCoordinatesDuringTraining(PredictCallback): 
    '''
    Assumes 3-5 clustering dimensions
    '''
    def __init__(self, 
                 outputfile,
                 start_pred_index,
                 end_pred_index,
                 n_keep = 3,
                 cycle_colors=False,
                 **kwargs):
        super(plotGravNetCoordinatesDuringTraining, self).__init__(function_to_apply=self.make_plot,**kwargs)
        self.outputfile=outputfile
        self.cycle_colors=cycle_colors
        self.coordinates_a=[]
        self.coordinates_b=[]
        self.coordinates_c=[]
        self.ndims=end_pred_index-start_pred_index
        
        self.n_keep=n_keep
        self.keep_counter=0
        
        if self.ndims == 3:
            self.coordinates_a=[i+start_pred_index for i in range(self.ndims)]
            self.coordinates_b=[start_pred_index+1, start_pred_index+2, start_pred_index]
            self.coordinates_c=[start_pred_index, start_pred_index+2, start_pred_index+1]
            
        elif self.ndims == 4:
            self.coordinates_a=[i+start_pred_index for i in range(self.ndims-1)]
            self.coordinates_b=[start_pred_index+1, start_pred_index+2, start_pred_index+3]
            self.coordinates_c=[start_pred_index, start_pred_index+2, start_pred_index+3]
            
        elif self.ndims == 5:
            self.coordinates_a=[i+start_pred_index for i in range(self.ndims-2)]
            self.coordinates_b=[start_pred_index+1, start_pred_index+2, start_pred_index+3]
            self.coordinates_c=[start_pred_index+2, start_pred_index+3, start_pred_index+4]
            
        else: 
            raise ValueError("plotGravNetCoordinatesDuringTraining: max 5 dimensions latent space")
        
        #print(self.coordinates_a, self.coordinates_b, self.coordinates_c)
        if self.td.nElements()>1:
            raise ValueError("plotEventDuringTraining: only one event allowed")
        
        self.gs = gridspec.GridSpec(2, 2)
        self.plot_process=None
    
           
    def make_plot(self,counter,feat,predicted,truth):  
        
        self.keep_counter+=1
        if self.keep_counter > self.n_keep:
            self.keep_counter=0
            
        if self.plot_process is not None:
            self.plot_process.join()
            
        self.plot_process = Process(target=self._make_plot, args=(counter,feat,predicted,truth))
        self.plot_process.start()
        
    #def make_plot(self,counter,feat,predicted,truth):  
    #    self._make_plot(counter,feat,predicted,truth)
        
        
    def _make_plot(self,counter,feat,predicted,truth):

        #exception handling is weird for keras fit right now... explicitely print exceptions at the end
        try:
            pred = predicted[0]
            feat = feat[0] #remove row splits
            truth = truth[0]#remove row splits, anyway one event
            
            
            fig = plt.figure(figsize=(10,8))
            ax = [fig.add_subplot(self.gs[0,0], projection='3d'), 
                  fig.add_subplot(self.gs[0,1], projection='3d'), 
                  fig.add_subplot(self.gs[1,0], projection='3d'), 
                  fig.add_subplot(self.gs[1,1], projection='3d')]
            
            
            _, pred = split_feat_pred(pred)    
            data = create_index_dict(truth, pred, usetf=False)
            feats = create_feature_dict(feat)
            
            seed = truth.shape[0]
            if self.cycle_colors:
                seed += counter
            
            cmap = createRandomizedColors('jet',seed=seed)
            
            
            make_original_truth_shower_plot(plt, ax[0], 
                                            data['truthHitAssignementIdx'],                      
                                             feats['recHitEnergy'], 
                                             feats['recHitX'],
                                             feats['recHitY'],
                                             feats['recHitZ'],
                                             cmap=cmap)
            
            angle_in=counter+60.
            while angle_in>=360: angle_in-=360
            while angle_in<=-360: angle_in-=360
            ax[0].view_init(30, angle_in)
            
            
            make_original_truth_shower_plot(plt, ax[1], 
                                            data['truthHitAssignementIdx'],                      
                                             feats['recHitEnergy'], 
                                             pred[:,self.coordinates_a[0]],
                                             pred[:,self.coordinates_a[1]],
                                             pred[:,self.coordinates_a[2]],
                                             cmap=cmap)
            
            make_original_truth_shower_plot(plt, ax[2], 
                                            data['truthHitAssignementIdx'],                      
                                             feats['recHitEnergy'], 
                                             pred[:,self.coordinates_b[0]],
                                             pred[:,self.coordinates_b[1]],
                                             pred[:,self.coordinates_b[2]],
                                             cmap=cmap)
            
            make_original_truth_shower_plot(plt, ax[3], 
                                            data['truthHitAssignementIdx'],                      
                                             feats['recHitEnergy'], 
                                             pred[:,self.coordinates_c[0]],
                                             pred[:,self.coordinates_c[1]],
                                             pred[:,self.coordinates_c[2]],
                                             cmap=cmap)
            
            
            
            plt.tight_layout()
            fig.savefig(self.outputfile+str(self.keep_counter)+".pdf")
            fig.clear()
            plt.close(fig)
            plt.clf()
            plt.cla()
            plt.close() 
            
            
        
        except Exception as e:
            print(e)
            raise e
        
        