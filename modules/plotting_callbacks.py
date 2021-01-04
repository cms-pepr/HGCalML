
print(">>>> WARNING: THE MODULE", __name__ ,"IS MARKED FOR REMOVAL","move implementations still in use to callbacks")

from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from multiprocessing import Process
import numpy as np
from datastructures import TrainData_OC
import matplotlib.gridspec as gridspec
import os

import plotly.express as px
import pandas as pd

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

def publish(file_to_publish, publish_to_path):
    cpstring = 'cp -f '
    if "@" in publish_to_path:
        cpstring = 'scp '
    basefilename = os.path.basename(file_to_publish)
    os.system(cpstring + file_to_publish + ' ' + publish_to_path +'_'+basefilename+ ' 2>&1 > /dev/null') 

def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx"):
    ta = df[qualifier]
    unta = np.unique(ta)
    np.random.shuffle(unta)
    unta = unta[unta>-0.1]
    for i in range(len(unta)):
        df[qualifier][df[qualifier] ==unta[i]]=i

class plotClusteringDuringTraining(PredictCallback):
    def __init__(self,
                 outputfile,
                 cycle_colors=False,
                 publish=None,
                 n_keep=1,
                 use_backgather_idx=7,#and more,
                 batchsize=300000,
                 **kwargs):
        
        super(plotClusteringDuringTraining, self).__init__(function_to_apply=self.make_plot, batchsize=batchsize,**kwargs)
        self.outputfile = outputfile
        self.cycle_colors = cycle_colors
        assert n_keep>0
        self.n_keep = n_keep-1
        self.use_backgather_idx=use_backgather_idx
        self.publish = publish
        ## preparation
        os.system('mkdir -p '+os.path.dirname(outputfile))
        
        #internals
        self.keep_counter = 0
        if self.td.nElements() > 1:
            raise ValueError("plotEventDuringTraining: only one event allowed")

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
            feats = td.createFeatureDict(feat[0],addxycomb=False)
            backgather = predicted[self.use_backgather_idx]
            truths = td.createTruthDict(truth[0])
            
            data = {}
            data.update(feats)
            data.update(truths)
            
            if len(backgather.shape)<2:
                backgather = np.expand_dims(backgather,axis=1)
            
            data['recHitLogEnergy'] = np.log(data['recHitEnergy']+1)
            data['hitBackGatherIdx'] = backgather
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            
            shuffle_truth_colors(df)
            
            fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", color="truthHitAssignementIdx", size="recHitLogEnergy",
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.outputfile + str(self.keep_counter) + "_truth.html")
            
            bgfile = self.outputfile + str(self.keep_counter) + "_backgather.html"
            #now the cluster indices
            
            shuffle_truth_colors(df,"hitBackGatherIdx")
            
            fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", color="hitBackGatherIdx", size="recHitLogEnergy",
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(bgfile)
            
            if self.publish is not None:
                publish(bgfile, self.publish)

        except Exception as e:
            print(e)
            raise e


class plotEventDuringTraining(PredictCallback):
    def __init__(self,
                 outputfile,
                 log_energy=False,
                 cycle_colors=False,
                 publish=None,
                 n_keep=1,
                 beta_threshold=0.01,
                 **kwargs):
        super(plotEventDuringTraining, self).__init__(function_to_apply=self.make_plot, **kwargs)
        self.outputfile = outputfile
        os.system('mkdir -p '+os.path.dirname(outputfile))
        self.cycle_colors = cycle_colors
        self.log_energy = log_energy
        self.beta_threshold=beta_threshold
        assert n_keep>0
        self.n_keep = n_keep-1
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
            feats = td.createFeatureDict(feat[0],addxycomb=False)
            truths = td.createTruthDict(truth[0])
            
            predBeta = predicted[0]
            predCCoords = predicted[1]
            if not predCCoords.shape[-1] == 3:
                return #just for 3D ccoords
            
            #for later
            predEnergy = predicted[2]
            predX = predicted[3][:,0:1]
            predY = predicted[3][:,1:2]
            
            data = {}
            data.update(feats)
            data.update(truths)
            
            
            data['recHitLogEnergy'] = np.log(data['recHitEnergy']+1)
            data['predBeta'] = predBeta
            data['predBeta+0.05'] = predBeta+0.05 #so that the others don't disappear
            data['predCCoordsX'] = predCCoords[:,0:1]
            data['predCCoordsY'] = predCCoords[:,1:2]
            data['predCCoordsZ'] = predCCoords[:,2:3]
            data['predEnergy'] = predEnergy
            data['predX']=predX
            data['predY']=predY
            data['(predBeta+0.05)**2'] = data['predBeta+0.05']**2
            data['(thresh(predBeta)+0.05))**2'] = np.where(predBeta>self.beta_threshold ,data['(predBeta+0.05)**2'], 0.)
            
            #for k in data:
            #    print(k, data[k].shape)
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            
            #fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", color="truthHitAssignementIdx", size="recHitLogEnergy")
            #fig.write_html(self.outputfile + str(self.keep_counter) + "_truth.html")
            shuffle_truth_colors(df)
            #now the cluster indices
            fig = px.scatter_3d(df, x="predCCoordsX", y="predCCoordsY", z="predCCoordsZ", 
                                color="truthHitAssignementIdx", size="recHitLogEnergy",
                                hover_data=['predBeta','predEnergy', 'predX', 'predY', 'truthHitAssignementIdx', 
                                            'truthHitAssignedEnergies', 'truthHitAssignedX','truthHitAssignedY'],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_ccoords.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
            
            fig = px.scatter_3d(df, x="predCCoordsX", y="predCCoordsY", z="predCCoordsZ", 
                                color="truthHitAssignementIdx", size="(predBeta+0.05)**2",
                                hover_data=['predBeta','predEnergy', 'predX', 'predY', 'truthHitAssignementIdx', 
                                            'truthHitAssignedEnergies', 'truthHitAssignedX','truthHitAssignedY'],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_ccoords_betasize.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
                
            # thresholded
            fig = px.scatter_3d(df, x="predCCoordsX", y="predCCoordsY", z="predCCoordsZ", 
                                color="truthHitAssignementIdx", size="(thresh(predBeta)+0.05))**2",
                                hover_data=['predBeta','predEnergy', 'predX', 'predY', 'truthHitAssignementIdx', 
                                            'truthHitAssignedEnergies', 'truthHitAssignedX','truthHitAssignedY'],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_ccoords_betathresh.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
            
            


        except Exception as e:
            print(e)
            raise e

