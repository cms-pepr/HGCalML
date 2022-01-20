
print("MODULE OBSOLETE?",__name__)
raise ImportError("MODULE",__name__,"will be removed")

import traceback

import numpy as np
import tensorflow as tf
import threading
import queue
import graph_functions
from callbacks import publish
from importlib import reload

from training_metrics_plots import TrainingMetricPlots

graph_functions = reload(graph_functions)

from Layers import DictModel


class Worker(threading.Thread):
    def __init__(self, q, tensorboard_manager, analyzer, database_manager=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.tensorboard_manager = tensorboard_manager
        self.analyzer=analyzer
        self.database_manager = database_manager


    def _run(self, data):
        with tf.device('/cpu:0'):
            step, feat_dict, pred_dict, truth_dict, loss_value = data
            analysed_graphs, metadata = self.analyzer.analyse_single_endcap(feat_dict, truth_dict, pred_dict)

        dic = dict()
        dic['efficiency'] = metadata['efficiency']
        dic['fake_rate'] = metadata['fake_rate']
        dic['response'] = metadata['response_mean']
        dic['sum_response'] = metadata['response_sum_mean']
        dic['beta_threshold'] = metadata['beta_threshold']
        dic['distance_threshold'] = metadata['distance_threshold']
        dic['iteration'] = step

        dic['loss'] = loss_value

        dic['f_score_energy'] = metadata['reco_score']

        if self.database_manager is not None:
            self.database_manager.insert_experiment_data('training_performance_metrics', dic)

        dic = dic.copy()
        dic['precision_without_energy'] = 0
        dic['recall_without_energy'] = 0
        dic['f_score_without_energy'] = 0
        dic['precision_energy'] = metadata['pred_energy_percentage_matched']
        dic['recall_energy'] = metadata['truth_energy_percentage_matched']

        dic['num_truth_showers'] = metadata['num_truth_showers']
        dic['num_pred_showers'] = metadata['num_pred_showers']


        if self.database_manager is not None:
            self.database_manager.insert_experiment_data('training_performance_metrics_extended', dic)

        if self.tensorboard_manager is not None:
            self.tensorboard_manager.step(step, dic)

    def run(self):
        while True:
            try:
                data = self.q.get(timeout=3)  # 3s timeout

            except queue.Empty:
                continue

            self._run(data)
            self.q.task_done()



class RunningMetricsDatabaseAdditionCallback(tf.keras.callbacks.Callback):
    def __init__(self, td, tensorboard_manager=None, analyzer=None, database_manager=None):
        """Initialize intermediate variables for batches and lists."""
        super().__init__()
        self.td = td
        q = queue.Queue()
        self.q = q

        if analyzer is None:
            raise RuntimeError("Please pass an analyzer, can't run without it")

        if tensorboard_manager is None and database_manager is None:
            raise RuntimeError("Both tensorboard manager and database managers are None. No where to write to.")

        self. worker = Worker(self.q, tensorboard_manager, database_manager=database_manager, analyzer=analyzer)
        self.thread = self.worker.start()


    def add(self, data):
        if self.q.empty():
            # print("Putting on")
            self.q.put(data)

    def on_train_batch_end(self, batch, logs=None):
        # y_pred = logs['y_pred']
        # x = logs['x']
        
        if not isinstance(self.model , DictModel):
            raise ValueError("RunningMetricsDatabaseAdditionCallback: requires DictModel")

        x = self.model.data_x
        y_pred = self.model.data_y_pred



        beta = y_pred[0]
        cc = y_pred[1]
        pred_energy = y_pred[2]



        feat, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = self.td.interpretAllModelInputs(x)
        #
        # print(t_fully_contained.shape)
        # 0/0
        #
        predictions_dict = y_pred #DictModel
        truth_dict = dict()
        truth_dict['truthHitAssignementIdx'] = t_idx
        truth_dict['truthHitAssignedEnergies'] = t_energy
        truth_dict['truthHitAssignedT'] = t_time
        truth_dict['truthHitAssignedPIDs'] = t_pid
        truth_dict['truthHitAssignedDepEnergies'] = t_energy*0
        truth_dict['truthHitAssignedEta'] = t_pos[:, 0:1]
        truth_dict['truthHitAssignedPhi'] = t_pos[:, 1:2]
        truth_dict['truthHitAssignedX'] = t_energy*0
        truth_dict['truthHitAssignedY'] = t_energy*0
        truth_dict['truthHitAssignedZ'] = t_energy*0

        feat_dict = self.td.createFeatureDict(feat)


        loss_value = logs['loss']


        feat_dict_numpy = dict()
        pred_dict_numpy = dict()
        truth_dict_numpy = dict()
        for key, value in feat_dict.items():
            if type(value) is not np.ndarray:
                feat_dict_numpy[key] = value[0:row_splits[1,0]].numpy()
            else:
                feat_dict_numpy[key] = value[0:row_splits[1,0]].numpy()

        for key, value in predictions_dict.items():
            if type(value) is not np.ndarray:
                pred_dict_numpy[key] = value[0:row_splits[1,0]].numpy()
            else:
                pred_dict_numpy[key] = value[0:row_splits[1,0]].numpy()

        for key, value in truth_dict.items():
            if type(value) is not np.ndarray:
                truth_dict_numpy[key] = value[0:row_splits[1,0]].numpy()
            else:
                truth_dict_numpy[key] = value[0:row_splits[1,0]].numpy()


        data = self.model.num_train_step, feat_dict_numpy, pred_dict_numpy, truth_dict_numpy, loss_value

        self.add(data)
        # self.worker._run(data)



class RunningMetricsPlotterCallback(tf.keras.callbacks.Callback):
    def __init__(self, after_n_batches, database_reading_manager, output_html_location, average_over=100, publish=None):
        super().__init__()
        self.after_n_batches = after_n_batches
        self.database_reading_manager = database_reading_manager
        self.output_location = output_html_location
        self.average_over = average_over
        self.plotter = TrainingMetricPlots(database_reading_manager, experiment_name=None, ignore_cache=True)
        self.publish = publish

    def on_train_batch_end(self, batch, logs=None):
        if self.model.num_train_step > 0 and self.model.num_train_step % self.after_n_batches==0:
            pass
        else:
            return
        print("Gonna run callback to make htmls for loss and more")
        try:
            try:
                self.plotter.do_plot_to_html(self.output_location, average_over=self.average_over)
            except TrainingMetricPlots.ExperimentNotFoundError as e:
                print("Possible experiment name problem, otherwise maybe data isn't yet written. Skipping writing HTML file.")
            if self.publish is not None:
                print("Publishing")
                publish(self.output_location, self.publish)
        except Exception as e:
            print(e.args)
            print(e)
            traceback.print_exc()


class RunningMetricsCallback(RunningMetricsDatabaseAdditionCallback):
    def __init__(self, *args, **kwargs):
        print("Obselete: Please change the name to RunningMetricsDatabaseAdditionCallback instead of RunningMetricsCallback."
              "This is to avoid confusion with RunningMetricsPlotterCallback. This class will be removed in the future.")
        super(RunningMetricsCallback, self).__init__(*args, **kwargs)
