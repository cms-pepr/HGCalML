

from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from index_dicts import create_index_dict, create_feature_dict
import matplotlib.pyplot as plt


class plotEventDuringTraining(PredictCallback): 
    def __init__(self, 
                 outputfile,
                 **kwargs):
        super(plotEventDuringTraining, self).__init__(function_to_apply=self.make_plot,**kwargs)
        self.outputfile=outputfile
        if self.td.nElements()>1:
            raise ValueError("plotEventDuringTraining: only one event allowed")
        
    def make_plot(self,counter,feat,predicted,truth):

        pred = predicted[0]
        feat = feat[0] #remove row splits
        truth = truth[0]#remove row splits, anyway one event
        
        fig = plt.figure(figsize=(10,4))
        ax = [fig.add_subplot(1,2,1, projection='3d'), fig.add_subplot(1,2,2)]
            
        data = create_index_dict(truth, pred, usetf=False)
        feats = create_feature_dict(feat)
        
        make_cluster_coordinates_plot(plt, ax[1], 
                                      data['truthHitAssignementIdx'], #[ V ]
                                      data['predBeta'],               #[ V ]
                                      data['predCCoords'])
        
        
        
        make_original_truth_shower_plot(plt, ax[0], 
                                        data['truthHitAssignementIdx'],                      
                                         feats['recHitEnergy'], 
                                         feats['recHitX'],
                                         feats['recHitY'],
                                         feats['recHitZ'])
        
        
        angle_in=counter+80.
        while angle_in>=360: angle_in-=360
        while angle_in<=-360: angle_in-=360
        ax[0].view_init(30, angle_in)
        
        
        plt.tight_layout()
        fig.savefig(self.outputfile+str(counter)+".pdf")
        fig.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close() 

        
        