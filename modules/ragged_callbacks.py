

from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot, createRandomizedColors, make_eta_phi_projection_truth_plot
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from index_dicts import create_index_dict, create_feature_dict, split_feat_pred
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from multiprocessing import Process
import numpy as np

class plotEventDuringTraining(PredictCallback): 
    def __init__(self, 
                 outputfile,
                 log_energy=False,
                 cycle_colors=False,
                 **kwargs):
        super(plotEventDuringTraining, self).__init__(function_to_apply=self.make_plot,**kwargs)
        self.outputfile=outputfile
        self.cycle_colors=cycle_colors
        self.log_energy = log_energy
        if self.td.nElements()>1:
            raise ValueError("plotEventDuringTraining: only one event allowed")
        
        self.gs = gridspec.GridSpec(2, 2)
        self.plot_process=None
            
    def make_plot(self,counter,feat,predicted,truth):  
        if self.plot_process is not None:
            self.plot_process.join()
            
        self.plot_process = Process(target=self._make_plot, args=(counter,feat,predicted,truth))
        self.plot_process.start()
        #self._make_plot(counter,feat,predicted,truth)
        
    def _make_plot(self,counter,feat,predicted,truth):

        #exception handling is weird for keras fit right now... explicitely print exceptions at the end
        try:
            pred = predicted[0]
            feat = feat[0] #remove row splits
            truth = truth[0]#remove row splits, anyway one event
            
            
            fig = plt.figure(figsize=(10,8))
            ax = [fig.add_subplot(self.gs[0,0], projection='3d'), 
                  fig.add_subplot(self.gs[0,1]), 
                  fig.add_subplot(self.gs[1,:])]
            
            
            _, pred = split_feat_pred(pred)    
            data = create_index_dict(truth, pred, usetf=False)
            feats = create_feature_dict(feat)
            
            seed = truth.shape[0]
            if self.cycle_colors:
                seed += counter
            
            cmap = createRandomizedColors('jet',seed=seed)
            
            identified = make_cluster_coordinates_plot(plt, ax[1], 
                                          data['truthHitAssignementIdx'], #[ V  x 1] or [ V ]
                                          data['predBeta'],               #[ V  x 1] or [ V ]
                                          data['predCCoords'],
                                          cmap=cmap)
            
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
            
            predEnergy=data['predEnergy']
            if self.log_energy:
                predEnergy = np.exp(predEnergy) - 1
                
            make_eta_phi_projection_truth_plot(plt,ax[2],
                                               data['truthHitAssignementIdx'],                      
                                    feats['recHitEnergy'],                       
                                    feats['recHitEta'],                       
                                    feats['recHitRelPhi'], 
                                    data['predEta']+feats['recHitEta'],
                                    data['predPhi']+feats['recHitRelPhi'],
                                    data['truthHitAssignedEtas'],
                                    data['truthHitAssignedPhis'],
                                    data['truthHitAssignedEnergies'],
                                    data['predBeta'],               
                                    data['predCCoords'],            
                                    cmap=cmap,
                                    identified=identified,
                                    predEnergy=predEnergy
                                    )
            
            plt.tight_layout()
            fig.savefig(self.outputfile+str(counter)+".pdf")
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
                 cycle_colors=False,
                 **kwargs):
        super(plotGravNetCoordinatesDuringTraining, self).__init__(function_to_apply=self.make_plot,**kwargs)
        self.outputfile=outputfile
        self.cycle_colors=cycle_colors
        self.coordinates_a=[]
        self.coordinates_b=[]
        self.coordinates_c=[]
        self.ndims=end_pred_index-start_pred_index
        
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
            fig.savefig(self.outputfile+str(counter)+".pdf")
            fig.clear()
            plt.close(fig)
            plt.clf()
            plt.cla()
            plt.close() 
        
        except Exception as e:
            print(e)
            raise e
        
        