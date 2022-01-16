
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from multiprocessing import Process
import numpy as np
from datastructures import TrainData_NanoML
import os

import plotly.express as px
import pandas as pd
from Layers import DictModel

from plotting_tools import publish, shuffle_truth_colors
       
       
class plotDuringTrainingBase(PredictCallback):
    def __init__(self,
                 outputfile="",
                 cycle_colors=False,
                 publish=None,
                 n_keep=1,
                 **kwargs):
        self.outputfile = outputfile
        os.system('mkdir -p '+os.path.dirname(outputfile))
        self.cycle_colors = cycle_colors
        assert n_keep>0
        self.n_keep = n_keep-1
        self.keep_counter = 0

        self.plot_process = None
        self.publish = publish
        
        super(plotDuringTrainingBase, self).__init__(function_to_apply=self.make_plot, **kwargs)
        if self.td.nElements() > 1:
            raise ValueError("plotDuringTrainingBase: only one event allowed")

    def make_plot(self, counter, feat, predicted, truth):
        if self.plot_process is not None:
            self.plot_process.join()

        self.keep_counter += 1
        if self.keep_counter > self.n_keep:
            self.keep_counter = 0

        self.plot_process = Process(target=self._make_plot, args=(counter, feat, predicted, truth))
        self.plot_process.start()
        # self._make_plot(counter,feat,predicted,truth)
     
    def _make_plot(self,counter, feat, predicted, truth):
        pass #implement in inheriting classes

class plotClusteringDuringTraining(plotDuringTrainingBase):
    def __init__(self,
                 use_backgather_idx=7,
                 **kwargs):
        
        self.use_backgather_idx=use_backgather_idx
        super(plotClusteringDuringTraining, self).__init__(**kwargs)
     
     
    def _make_plot(self, counter, feat, predicted, truth):#all these are lists and also include row splits
        try:
            td = TrainData_NanoML()#contains all dicts
            #row splits not needed
            feats = td.createFeatureDict(feat,addxycomb=False)
            backgather = predicted[self.use_backgather_idx]
            truths = td.createTruthDict(feat)
            
            
            
            data = {}
            data.update(feats)
            data.update(truths)
            
            if len(backgather.shape)<2:
                backgather = np.expand_dims(backgather,axis=1)
            
            data['recHitLogEnergy'] = np.log(data['recHitEnergy']+1)
            data['hitBackGatherIdx'] = backgather
            
            from globals import cluster_space as cs 
            removednoise = np.logical_and(data["recHitX"] == cs.noise_coord, 
                                          data["recHitY"] == cs.noise_coord)
            removednoise = np.logical_and(data["recHitZ"] == cs.noise_coord, 
                                          removednoise)
            #remove removed noise
            for k in data.keys():
                data[k] = data[k][not removednoise]
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            
            shuffle_truth_colors(df)
            
            fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", 
                                color="truthHitAssignementIdx", size="recHitLogEnergy",
                                symbol = "recHitID",
                                #template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.outputfile + str(self.keep_counter) + "_truth.html")
            
            bgfile = self.outputfile + str(self.keep_counter) + "_backgather.html"
            #now the cluster indices
            
            shuffle_truth_colors(df,"hitBackGatherIdx")
            
            fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", color="hitBackGatherIdx", size="recHitLogEnergy",
                                #template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(bgfile)
            
            if self.publish is not None:
                publish(bgfile, self.publish)

        except Exception as e:
            print(e)
            raise e


class plotEventDuringTraining(plotDuringTrainingBase):
    def __init__(self,
                 beta_threshold=0.01,
                 **kwargs):
        self.beta_threshold=beta_threshold
        super(plotEventDuringTraining, self).__init__(**kwargs)
        
        self.datastorage=None #keep this small


    def _make_plot(self, counter, feat, predicted, truth):

        try:
            '''
            [pred_beta, 
             pred_ccoords,
             pred_energy, 
             pred_pos, 
             pred_time, 
             pred_id
            '''
            td = TrainData_NanoML()#contains all dicts
            #row splits not needed
            feats = td.createFeatureDict(feat,addxycomb=False)
            truths = td.createTruthDict(feat)
            
            predBeta = predicted['pred_beta']
            
            print('>>>> plotting cluster coordinates... average beta',np.mean(predBeta), ' lowest beta ', 
                  np.min(predBeta), 'highest beta', np.max(predBeta))
            
            
            #for later
            predEnergy = predicted['pred_energy_corr_factor']
            predX = predicted['pred_pos'][:,0:1]
            predY = predicted['pred_pos'][:,1:2]
            predT = predicted['pred_time']
            predD = predicted['pred_dist']
            
            data = {}
            data.update(feats)
            data.update(truths)
            
            predCCoords = predicted['pred_ccoords']
            
            
            data['recHitLogEnergy'] = np.log(data['recHitEnergy']+1)
            data['predBeta'] = predBeta
            data['predBeta+0.05'] = predBeta+0.05 #so that the others don't disappear
            data['predEnergy'] = predEnergy
            data['predX']=predX
            data['predY']=predY
            data['predT']=predT
            data['predD']=predD
            data['(predBeta+0.05)**2'] = data['predBeta+0.05']**2
            data['(thresh(predBeta)+0.05))**2'] = np.where(predBeta>self.beta_threshold ,data['(predBeta+0.05)**2'], 0.)
            
            if not predCCoords.shape[-1] == 3:
                self.projection_plot(data, predCCoords)
                return
            
            
            data['predCCoordsX'] = predCCoords[:,0:1]
            data['predCCoordsY'] = predCCoords[:,1:2]
            data['predCCoordsZ'] = predCCoords[:,2:3]
            
            from globals import cluster_space as cs 
            removednoise = np.logical_and(data["predCCoordsX"] == cs.noise_coord, 
                                          data["predCCoordsY"] == cs.noise_coord)
            removednoise = np.logical_and(data["predCCoordsZ"] == cs.noise_coord, 
                                          removednoise)
            removednoise = np.where(removednoise[:,0],False,True)
            #remove removed noise
            
            for k in data.keys():
                data[k] = data[k][removednoise]
            
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            
            #fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY", color="truthHitAssignementIdx", size="recHitLogEnergy")
            #fig.write_html(self.outputfile + str(self.keep_counter) + "_truth.html")
            shuffle_truth_colors(df)
            #now the cluster indices
            
            hover_data=['predBeta','predD','predEnergy','truthHitAssignedEnergies',
                        'predT','truthHitAssignedT',
                        'predX', 'truthHitAssignedX',
                        'predY', 'truthHitAssignedY',
                        'truthHitAssignementIdx']
            
            fig = px.scatter_3d(df, x="predCCoordsX", y="predCCoordsY", z="predCCoordsZ", 
                                color="truthHitAssignementIdx", size="recHitLogEnergy",
                                symbol = "recHitID",
                                hover_data=hover_data,
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_ccoords.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
            
            fig = px.scatter_3d(df, x="predCCoordsX", y="predCCoordsY", z="predCCoordsZ", 
                                color="truthHitAssignementIdx", size="(predBeta+0.05)**2",
                                hover_data=hover_data,
                                symbol = "recHitID",
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_ccoords_betasize.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
                
            # thresholded
            fig = px.scatter_3d(df, x="recHitX", y="recHitZ", z="recHitY",
                                color="truthHitAssignementIdx", size="recHitLogEnergy",
                                symbol = "recHitID",
                                hover_data=['predBeta','predEnergy', 'predX', 'predY', 'truthHitAssignementIdx', 
                                            'truthHitAssignedEnergies', 'truthHitAssignedX','truthHitAssignedY'],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_truth.html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
            
            


        except Exception as e:
            print(e)
            raise e
        
    def projection_plot(self, data, predCCoords):
        
        import time
        thistime = time.time()
        from sklearn.manifold import TSNE
        tsne = TSNE(verbose=1,n_iter=250)#try with lowest default
        coords2D = tsne.fit_transform(predCCoords)
        
        #coords2D = predCCoords[:,0:2]
        
        
        data['predCCoordsX']=coords2D[:,0:1]
        data['predCCoordsY']=coords2D[:,1:2]
        
        
        df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
        shuffle_truth_colors(df)
        
        
        hover_data=['predBeta','predD','predEnergy','truthHitAssignedEnergies',
                        'predT','truthHitAssignedT',
                        'predX', 'truthHitAssignedX',
                        'predY', 'truthHitAssignedY',
                        'truthHitAssignementIdx']
            
        fig = px.scatter(df, x="predCCoordsX", y="predCCoordsY",
                            color="truthHitAssignementIdx", size="(predBeta+0.05)**2",
                            symbol = "recHitID",
                            hover_data=hover_data,
                            template='plotly_dark',
                color_continuous_scale=px.colors.sequential.Rainbow)
        fig.update_traces(marker=dict(line=dict(width=0)))
        ccfile = self.outputfile + str(self.keep_counter) + "_proj_ccoords_betasize.html"
        fig.write_html(ccfile)
        
        
        if self.publish is not None:
            publish(ccfile, self.publish)
        
        
            
class plotClusterSummary(PredictCallback):        
    def __init__(self,
                 outputfile="",
                 nevents=20,
                 publish=None,
                 **kwargs):
        self.outputfile = outputfile
        os.system('mkdir -p '+os.path.dirname(outputfile))
        self.publish = publish
        self.plot_process=None
        super(plotClusterSummary, self).__init__(function_to_apply=self.make_plot, 
                                                 batchsize=1,
                                                 use_event=-1, 
                                                 **kwargs)
        
        self.td=self.td.getSlice(0,min(nevents,self.td.nElements()))


    def subdict(self, d, sel):
        o={}
        for k in d.keys():
            o[k] = d[k][sel]
        return o
    
    def make_plot(self, counter, feat, predicted, truth):
        if self.plot_process is not None:
            self.plot_process.join(60)#safety margin to not block training if plotting took too long
            try:
                self.plot_process.terminate()#got enough time
            except:
                pass

        self.plot_process = Process(target=self._make_plot, args=(counter, feat, predicted, truth))
        self.plot_process.start()
        
    def _make_plot(self, counter, feat, predicted, truth):
        
        td = TrainData_NanoML()
        preddict=predicted
        
        cdata=td.createTruthDict(feat)
        cdata['predBeta'] = preddict['pred_beta']
        cdata['predCCoords'] = preddict['pred_ccoords']
        cdata['predD'] = preddict['pred_dist']
        rs = feat[-1]#last one has to be row splits
        # this will not work, since it will be adapted by batch, and not anymore the right tow splits
        #rs = preddict['row_splits']
        
        eid=0
        eids=[]
        #make event id
        for i in range(len(rs)-1):
            eids.append( np.zeros( (rs[i+1,0]-rs[i,0], ) ,dtype='int64') +eid )
            eid+=1
        cdata['eid'] = np.concatenate(eids, axis=0)
        
        pids=[]
        vdtom=[]
        did=[]
        for i in range(eid):
            a,b,pid = self.run_per_event(self.subdict(cdata,i==cdata['eid']))
            vdtom.append(a)
            did.append(b)
            pids.append(pid)
            
        vdtom = np.concatenate(vdtom, axis=0)
        did = np.concatenate(did,axis=0)
        pids = np.concatenate(pids,axis=0)[:,0]
        upids = np.unique(pids).tolist()
        upids.append(0)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print(upids)
        for p in upids:
            svdtom=vdtom
            sdid=did
            if p:
                svdtom=vdtom[pids==p]
                sdid=did[pids==p]

            if not len(svdtom):
                continue
            
            fig = plt.figure()
            plt.hist(svdtom,bins=51,color='tab:blue',alpha = 0.5,label='same')
            plt.hist(sdid,bins=51,color='tab:orange',alpha = 0.5,label='other')
            plt.yscale('log')
            plt.xlabel('normalised distance')
            plt.ylabel('A.U.')
            plt.legend()
            ccfile=self.outputfile+str(p)+'_cluster.pdf'
            plt.savefig(ccfile)
            plt.cla()
            plt.clf()
            plt.close(fig)
            if self.publish is not None:
                publish(ccfile, self.publish)
     
    def run_per_event(self,data):
            
        tidx = data['truthHitAssignementIdx'][:,0]# V x 1
        utidx = np.unique(tidx)
        
        overflowat=6
        
        vtpid=[]
        vdtom = []
        did = []
        for uidx in utidx:
            if uidx < 0:
                continue
            thiscoords,thisbeta,thisdist = data['predCCoords'][tidx==uidx],data['predBeta'][tidx==uidx],data['predD'][tidx==uidx]
            ak = np.argmax(thisbeta)
            coords_ak = thiscoords[ak]
            dist_ak = thisdist[ak]
            
            pid = np.abs(data['truthHitAssignedPIDs'][tidx==uidx])
            
            dtom = np.sqrt(np.sum( (thiscoords-coords_ak)**2, axis=-1)) / (np.abs(dist_ak)+1e-3)
            dtom[dtom>overflowat]=overflowat
            #dtom = np.expand_dims(dtom,axis=1)
            dother =  np.sqrt(np.sum((data['predCCoords'][tidx!=uidx]-coords_ak )**2,axis=-1))/ (np.abs(dist_ak)+1e-3)
            
            dother[dother>overflowat]=overflowat
            #dother = np.expand_dims(dother,axis=1)
            dother = dother[np.random.choice(len(dother), size=int(min(len(dother), len(dtom))), replace=False)]#restrict
            
            vtpid.append(pid)
            vdtom.append(dtom)
            did.append(dother)
        
        vtpid = np.concatenate(vtpid,axis=0)
        vdtom = np.concatenate(vdtom,axis=0)
        did = np.concatenate(did,axis=0)
        return vdtom,did,vtpid
        


class plotGravNetCoordsDuringTraining(plotDuringTrainingBase):
    def __init__(self,
                 use_prediction_idx=16,
                 **kwargs):
        
        super(plotGravNetCoordsDuringTraining, self).__init__(**kwargs)
        self.use_prediction_idx=use_prediction_idx
     
     
    def _make_plot(self, counter, feat, predicted, truth):
        try:
            td = TrainData_NanoML()#contains all dicts
            truths = td.createTruthDict(feat)
            feats = td.createFeatureDict(feat,addxycomb=False)
                                        
            data = {}
            data.update(truths)
            data.update(feats)
            data['recHitLogEnergy'] = np.log(data['recHitEnergy']+1)
            
            coords = predicted[self.use_prediction_idx]
            if not coords.shape[-1] == 3:
                print("plotGravNetCoordsDuringTraining only supports 3D coordinates") #2D and >3D TBI
                return #not supported
                
            data['coord A'] = coords[:,0:1]
            data['coord B'] = coords[:,1:2]
            data['coord C'] = coords[:,2:3]
            
            df = pd.DataFrame (np.concatenate([data[k] for k in data],axis=1), columns = [k for k in data])
            shuffle_truth_colors(df)
            
            fig = px.scatter_3d(df, x="coord A", y="coord B", z="coord C", 
                                color="truthHitAssignementIdx", size="recHitLogEnergy",
                                symbol = "recHitID",
                                #hover_data=[],
                                template='plotly_dark',
                    color_continuous_scale=px.colors.sequential.Rainbow)
            fig.update_traces(marker=dict(line=dict(width=0)))
            ccfile = self.outputfile + str(self.keep_counter) + "_coords_"+ str(self.use_prediction_idx) +".html"
            fig.write_html(ccfile)
            
            
            if self.publish is not None:
                publish(ccfile, self.publish)
                
            
        except Exception as e:
            print(e)
            raise e
        
        
        
############ from running_full_validation_callbacks


import os
import mgzip
import tensorflow as tf

import graph_functions
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
from plotting_tools import publish
from importlib import reload


graph_functions = reload(graph_functions)


class RunningFullValidation(tf.keras.callbacks.Callback):
    def __init__(self, after_n_batches, predictor, analyzer, optimizer=None, test_on_points=None,
                 database_manager=None, pdfs_path=None, min_batch=0, table_prefix='gamma_full_validation',
                 run_optimization_loop_for=80, optimization_loop_num_init_points=5, 
                 limit_endcaps = -1,#all endcaps in file
                 limit_endcaps_by_time = 600,#in seconds, don't spend more than 10 minutes on this
                 trial_batch=10):
        """
        :param analyzer:
        :param after_n_batches: Run callback after n batches
        :param predictor: Predictor class for HGCal
        :param optimizer: Bayesian optimizer class for HGCal (OCHyperParamOptimizer)
        :param database_manager: Database writing manager, where to write the data, leave None without proper understanding
        of underlying database operations
        :param pdfs_path: Where to
        :param table_prefix: Adds this prefix to storage tables, leave None without proper understanding of underlying
        database operations
        """
        super().__init__()
        self.after_n_batches = after_n_batches
        self.predictor = predictor
        self.optimizer = optimizer
        self.batch_idx = 0
        self.hyper_param_points=test_on_points
        self.limit_endcaps = limit_endcaps
        self.limit_endcaps_by_time = limit_endcaps_by_time
        self.database_manager = database_manager
        self.table_prefix = table_prefix
        self.pdfs_path = pdfs_path
        self.run_optimization_loop_for = run_optimization_loop_for
        self.optimization_loop_num_init_points = optimization_loop_num_init_points
        self.trial_batch = trial_batch
        self.analyzer = analyzer

        if database_manager is None and pdfs_path is None:
            raise RuntimeError("Set either database manager or pdf output path")

        if optimizer is None != test_on_points is None: # Just an xor
            raise RuntimeError("Can either do optimization or run at specific points")

    #there is a lot that can go wrong but should not kill the training
    def on_train_batch_end(self, batch, logs=None):
        try:
            self._on_train_batch_end(batch, logs)
        except Exception as e:
            print('encountered the following exception when running RunningFullValidation callback:')
            print(e)
        
    def _on_train_batch_end(self, batch, logs=None):
        
        runcallback = False
        
        if self.trial_batch > 0:
            print('RunningFullValidation: trial run in...',self.trial_batch)
            self.trial_batch -= 1
            runcallback = False
        elif self.trial_batch==0:
            self.trial_batch -= 1
            print("Running full validation test to see if it fits resource requirements.")
            runcallback = True
            
        if self.batch_idx and (not self.batch_idx % self.after_n_batches):
            runcallback = True
        
        self.batch_idx+=1
        if not runcallback:
            return
            
        print("\n\nGonna run callback to do full validation...\n\n")

        try:
            all_data = self.predictor.predict(model=self.model, output_to_file=False)
        except FileNotFoundError:
            print("Model file not found. Will skip.")
            return

        if self.optimizer is not None:
            max_point = self.optimizer.optimize(all_data, num_iterations=self.run_optimization_loop_for, init_points=self.optimization_loop_num_init_points)
            b = max_point['params']['b']
            d = max_point['params']['d']
            test_on = [(b,d)]
        else:
            test_on = self.hyper_param_points


        #TBI: this should be sent to a thread and not block main execution
        for b, d in test_on:
            graphs, metadata = self.analyzer.analyse_from_data(all_data, b, d, 
                                                               limit_endcaps=self.limit_endcaps,
                                                               limit_endcaps_by_time=self.limit_endcaps_by_time)
            if self.optimizer is not None:
                self.optimizer.remove_data() # To save memory


            plotter = HGCalAnalysisPlotter()
            tags = dict()
            if self.optimizer is not None:
                tags['iteration'] = int(self.optimizer.iteration)
            else:
                tags['iteration'] = -1

            tags['training_iteration'] = int(self.batch_idx)

            plotter.add_data_from_analysed_graph_list(graphs, metadata, additional_tags=tags)

            if self.database_manager is not None:
                plotter.write_data_to_database(self.database_manager, self.table_prefix)
                print("Written to database")

            if self.pdfs_path is not None:
                plotter.write_to_pdf(pdfpath=os.path.join(self.pdfs_path, 'validation_results_%07d_%.2f_%.2f.pdf'%(self.batch_idx, b,d)))

        print('finished full validation callback, proceeding with training.')
        
        
        
        
        
        
        
