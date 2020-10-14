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
from keras.callbacks import Callback
from DeepJetCore.TrainData import TrainData
from DeepJetCore.dataPipeline import TrainDataGenerator
from LayersRagged import RaggedConstructTensor
import tensorflow as tf
from ragged_plotting_tools import analyse_one_window_cut, make_running_plots
from obc_data import append_window_dict_to_dataset_dict, build_window_visualization_dict, build_dataset_analysis_dict
import copy


class plotRunningPerformanceMetrics(Callback):
    def __init__(self,
                 samplefile,
                 accumulate_after_batches=5,
                 plot_after_batches=50,
                 batchsize=10,
                 beta_threshold=0.6,
                 distance_threshold=0.6,
                 iou_threshold=0.1,
                 n_windows_for_plots=5,
                 n_windows_for_scalar_metrics=5000000,
                 outputdir=None,
                 publish = None,
                 n_ccoords=None
                 ):
        """

        :param samplefile: the file to pick validation data from
        :param accumulate_after_batches: run performance metrics after n batches (a good value is 5)
        :param plot_after_batches: update and upload plots after n batches
        :param batchsize: batch size
        :param beta_threshold: beta threshold for running prediction on obc
        :param distance_threshold: distance threshold for running prediction on obc
        :param iou_threshold: iou threshold to use to match both for obc and for ticl
        :param n_windows_for_plots: how many windows to average to do running performance plots
        :param n_windows_for_scalar_metrics: the maximum windows to store data for scalar performance metrics as a function of iteration
        :param outputdir: the output directory where to store results
        :param publish: where to publish, could be ssh'able path
        :param n_ccoords: n coords for plots
        """
        super(plotRunningPerformanceMetrics, self).__init__()
        self.samplefile = samplefile
        self.counter = 0
        self.call_counter = 0
        self.decay_function = None
        self.outputdir = outputdir
        self.n_ccords=n_ccoords
        self.publish=publish

        self.accumulate_after_batches = accumulate_after_batches
        self.plot_after_batches = plot_after_batches
        self.run_on_epoch_end = False

        if self.run_on_epoch_end and self.accumulate_after_batches >= 0:
            print('PredictCallback: can only be used on epoch end OR after n batches, falling back to epoch end')
            self.accumulate_after_batches = 0

        td = TrainData()
        td.readFromFile(samplefile)
        # td_selected = td.split(self.n_events)  # check if this works in ragged out of the box
        # if use_event >= 0:
        #     if use_event < td.nElements():
        #         td.skim(use_event)
        #     else:
        #         td.skim(use_event % td.nElements())
        self.batchsize = batchsize
        self.td = td
        self.gen = TrainDataGenerator()
        self.gen.setBatchSize(self.batchsize)
        self.gen.setSkipTooLargeBatches(False)
        self.gen.setBuffer(td)

        self.n_batches=self.gen.getNBatches()


        with tf.device('/CPU:0'):
            self.ragged_constructor = RaggedConstructTensor()
        self.window_id = 0
        self.window_analysis_dicts = []
        self.n_windows_for_plots = n_windows_for_plots
        self.n_windows_for_scalar_metrics = n_windows_for_scalar_metrics
        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold

        self.scalar_metrics = dict()
        self.scalar_metrics['efficiency'] = []
        self.scalar_metrics['efficiency_ticl'] = []
        self.scalar_metrics['fake_rate'] = []
        self.scalar_metrics['fake_rate_ticl'] = []
        self.scalar_metrics['var_response'] = []
        self.scalar_metrics['var_response_ticl'] = []
        self.scalar_metrics['iteration'] = []

        self.plot_process = None


    def reset(self):
        self.call_counter = 0

    def predict_and_call(self, counter):
        feat, truth = next(self.gen.feedNumpyData())  # this is  [ [features],[truth],[None] ]

        if self.gen.lastBatch():
            self.gen.setBuffer(self.td)
            # self.gen.prepareNextEpoch()

        def dummy_gen():
            yield (feat, truth)

        # predicted = self.model.predict([feat, truth])


        #
        predicted = self.model.predict_generator(dummy_gen(),
                                                 steps=1,
                                                 max_queue_size=1,
                                                 use_multiprocessing=False,
                                                 verbose=2)

        #
        # if not isinstance(predicted, list):
        #     predicted = [predicted]
        #
        # print(len(predicted), len(predicted[0]), len(predicted[1]), len(self.td.copyFeatureListToNumpy()[0]))
        #
        self.accumulate(self.counter, feat,
                               predicted, truth)
        self.call_counter += 1

    def on_epoch_end(self, epoch, logs=None):
        self.counter = 0
        if not self.run_on_epoch_end: return
        self.predict_and_call(epoch)

    def on_batch_end(self, batch, logs=None):
        if self.accumulate_after_batches <= 0: return
        if self.counter % self.accumulate_after_batches == 0:
            self.predict_and_call(batch)
        if self.plot_after_batches > 0:
            if self.counter % self.plot_after_batches == 0:
                self.plot()
        self.counter += 1

    def plot(self):
        if self.plot_process is not None:
            self.plot_process.join()

        self.plot_process = Process(target=self._plot, args=(copy.deepcopy(self.window_analysis_dicts), copy.deepcopy(self.scalar_metrics)))
        self.plot_process.start()

    def _plot(self, window_analysis_dicts, scalar_metrics):
        with tf.device('/CPU:0'):
            if len(window_analysis_dicts) == self.n_windows_for_plots:
                print("Plotting and publishing")
                dataset_analysis_dict = build_dataset_analysis_dict()
                dataset_analysis_dict['beta_threshold'] = self.beta_threshold
                dataset_analysis_dict['distance_threshold'] = self.distance_threshold
                dataset_analysis_dict['iou_threshold'] = self.iou_threshold
                for x in window_analysis_dicts:
                    dataset_analysis_dict = append_window_dict_to_dataset_dict(dataset_analysis_dict, x)

                make_running_plots(self.outputdir, dataset_analysis_dict, scalar_metrics)

                if self.publish is not None:
                    for f in os.listdir(self.outputdir):
                        if f.endswith('.png'):
                            f_full = os.path.join(self.outputdir, f)
                            cpstring = 'cp -f '
                            if "@" in self.publish:
                                cpstring = 'scp '
                            s = (cpstring + f_full + ' ' + self.publish + f+' > /dev/null')
                            os.system(s)

    def accumulate(self, counter, feat, predicted, truth):
        print("Accumulating")

        with tf.device('/CPU:0'):
            new_window_analysis_dicts = self.analyse_one_file(feat, predicted, truth)
            self.window_analysis_dicts += new_window_analysis_dicts

            for i, wdict in enumerate(new_window_analysis_dicts):
                efficiency = float(wdict['window_num_found_showers']) / wdict['window_num_truth_showers']
                efficiency_ticl = float(wdict['window_num_found_showers_ticl']) / wdict['window_num_truth_showers']

                fake_rate = float(wdict['window_num_fake_showers']) / wdict['window_num_pred_showers']
                fake_rate_ticl = float(wdict['window_num_fake_showers_ticl']) / wdict['window_num_ticl_showers']

                truth_shower_energy = np.array(wdict['truth_shower_energy'])
                pred_shower_energy = np.array(wdict['truth_shower_matched_energy_regressed'])
                ticl_shower_energy = np.array(wdict['truth_shower_matched_energy_regressed_ticl'])

                filter = pred_shower_energy!=-1
                filter_ticl = ticl_shower_energy!=-1

                var_res = pred_shower_energy[filter]/truth_shower_energy[filter]
                var_res = np.std(var_res) / np.mean(var_res)
                var_res_ticl = ticl_shower_energy[filter_ticl] / truth_shower_energy[filter_ticl]
                var_res_ticl = np.std(var_res_ticl) / np.mean(var_res_ticl)

                iteration = counter + float((i +1)) / float(len(new_window_analysis_dicts))

                self.scalar_metrics['efficiency'].append(efficiency)
                self.scalar_metrics['efficiency_ticl'].append(efficiency_ticl)
                self.scalar_metrics['fake_rate'].append(fake_rate)
                self.scalar_metrics['fake_rate_ticl'].append(fake_rate_ticl)
                self.scalar_metrics['var_response'].append(var_res)
                self.scalar_metrics['var_response_ticl'].append(var_res_ticl)
                self.scalar_metrics['iteration'].append(iteration)

            while len(self.window_analysis_dicts) > self.n_windows_for_plots:
                self.window_analysis_dicts.pop(0)

            while len(self.scalar_metrics['iteration']) > self.n_windows_for_scalar_metrics:
                self.n_windows_for_scalar_metrics.pop(0)


    def analyse_one_file(self, features, predictions, truth_in, soft=False):
        predictions = tf.constant(predictions[0])

        row_splits = features[1][:, 0]

        features, _ = self.ragged_constructor((features[0], row_splits))
        truth, _ = self.ragged_constructor((truth_in[0], row_splits))

        hit_assigned_truth_id, row_splits = self.ragged_constructor((truth_in[0][:, 0][..., tf.newaxis], row_splits))

        # make 100% sure the cast doesn't hit the fan
        hit_assigned_truth_id = tf.where(hit_assigned_truth_id < -0.1, hit_assigned_truth_id - 0.1,
                                         hit_assigned_truth_id + 0.1)
        hit_assigned_truth_id = tf.cast(hit_assigned_truth_id[:, 0], tf.int32)

        num_unique = []
        shower_sizes = []

        # here ..._s refers to quantities per window/segment
        #

        window_analysis_dicts = []
        for i in range(len(row_splits) - 1):
            hit_assigned_truth_id_s = hit_assigned_truth_id[row_splits[i]:row_splits[i + 1]].numpy()
            features_s = features[row_splits[i]:row_splits[i + 1]].numpy()
            truth_s = truth[row_splits[i]:row_splits[i + 1]].numpy()
            prediction_s = predictions[row_splits[i]:row_splits[i + 1]].numpy()

            window_analysis_dict = analyse_one_window_cut(hit_assigned_truth_id_s, features_s, truth_s,
                                                          prediction_s,
                                                          self.beta_threshold, self.distance_threshold, self.iou_threshold, self.window_id, False,
                                                          soft=soft)

            window_analysis_dicts.append(window_analysis_dict)

            # append_window_dict_to_dataset_dict(dataset_analysis_dict, window_analysis_dict)
            # num_visualized_segments += 1
            self.window_id += 1

        return window_analysis_dicts


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
        
        