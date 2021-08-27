import numpy as np
import tensorflow as tf
import tensorboard_manager as tm
import threading
import queue
import graph_functions
from plotting_callbacks import publish
from importlib import reload

from training_metrics_plots import TrainingMetricPlots

graph_functions = reload(graph_functions)
from tensorflow.keras import backend as K
import tensorflow.keras.callbacks.experimental
from matching_and_analysis import analyse_hgcal_endcap
from scalar_metrics import compute_scalar_metrics


class Worker(threading.Thread):
    def __init__(self, q, tensorboard_manager, beta_threshold=0.5, dist_threshold=0.5, database_manager=None, with_local_distance_scaling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.tensorboard_manager = tensorboard_manager
        self.beta_threshold = beta_threshold
        self.dist_threshold = dist_threshold
        self.database_manager = database_manager
        self.with_local_distance_scaling = with_local_distance_scaling

    def run(self):
        while True:
            try:
                data = self.q.get(timeout=3)  # 3s timeout

            except queue.Empty:
                continue

            with tf.device('/cpu:0'):
                step, feat_dict, pred_dict, truth_dict, loss_value = data

                endcap_analysis_result = analyse_hgcal_endcap(feat_dict, truth_dict, pred_dict, self.beta_threshold, self.dist_threshold, iou_threshold=0.1, endcap_id=10, with_local_distance_scaling=False) # endcap id is not needed, so its just some bogus value

            try:
                eff = endcap_analysis_result['endcap_num_found_showers'] / float(endcap_analysis_result['endcap_num_truth_showers'])
            except ZeroDivisionError:
                eff = 0
            try:
                fake_rate = endcap_analysis_result['endcap_num_fake_showers'] / float(endcap_analysis_result['endcap_num_pred_showers'])
            except ZeroDivisionError:
                fake_rate = 0
            dic = dict()
            dic['efficiency'] = eff
            dic['fake_rate'] = fake_rate

            filter = np.array(endcap_analysis_result['truth_shower_found_or_not'])


            if len(filter) != 0:
                response_2 = np.array(endcap_analysis_result['truth_shower_matched_energy_regressed'])[filter] / np.array(endcap_analysis_result['truth_shower_energy'])[filter]
                sum_response_2 = np.array(endcap_analysis_result['truth_shower_matched_energy_sum'])[filter] / np.array(endcap_analysis_result['truth_shower_energy'])[filter]
                response_2 = np.mean(response_2).item()
                sum_response_2 = np.mean(sum_response_2).item()
            else:
                response_2 = 0
                sum_response_2 = 0


            dic['response'] = response_2
            dic['sum_response'] = sum_response_2


            dic['beta_threshold'] = self.beta_threshold
            dic['distance_threshold'] = self.dist_threshold
            dic['iteration'] = step

            precision, recall, f_score, precision_energy, recall_energy, f_score_energy = compute_scalar_metrics(result=endcap_analysis_result)
            dic['loss'] = loss_value

            dic['f_score_energy'] = f_score_energy

            if self.database_manager is not None:
                print("Adding new values to database ne")
                print(dic.keys())
                self.database_manager.insert_experiment_data('training_performance_metrics', dic)

            dic = dic.copy()
            dic['precision_without_energy'] = precision
            dic['recall_without_energy'] = recall
            dic['f_score_without_energy'] = f_score
            dic['precision_energy'] = precision_energy
            dic['recall_energy'] = recall_energy
            dic['num_truth_showers'] = endcap_analysis_result['endcap_num_truth_showers']
            dic['num_pred_showers'] = endcap_analysis_result['endcap_num_pred_showers']

            if self.database_manager is not None:
                print("Adding new values to database e")
                print(dic.keys())
                self.database_manager.insert_experiment_data('training_performance_metrics_extended', dic)

            if self.tensorboard_manager is not None:
                self.tensorboard_manager.step(step, dic)

            self.q.task_done()



class RunningMetricsDatabaseAdditionCallback(tf.keras.callbacks.Callback):
    def __init__(self, td, tensorboard_manager=None, beta_threshold=0.5, dist_threshold=0.5, with_local_distance_scaling=False, database_manager=None):
        """Initialize intermediate variables for batches and lists."""
        super().__init__()
        self.td = td
        q = queue.Queue()
        self.q = q

        if tensorboard_manager is None and database_manager is None:
            raise RuntimeError("Both tensorboard manager and database managers are None. No where to write to.")

        self.thread = Worker(self.q, tensorboard_manager, beta_threshold=beta_threshold, dist_threshold=dist_threshold, database_manager=database_manager, with_local_distance_scaling=with_local_distance_scaling).start()


    def add(self, data):
        if self.q.empty():
            # print("Putting on")
            self.q.put(data)

    def on_train_batch_end(self, batch, logs=None):
        # y_pred = logs['y_pred']
        # x = logs['x']

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
        predictions_dict = self.model.convert_output_to_dict(y_pred)
        truth_dict = dict()
        truth_dict['truthHitAssignementIdx'] = t_idx
        truth_dict['truthHitAssignedEnergies'] = t_energy
        truth_dict['truthHitAssignedT'] = t_time
        truth_dict['truthHitAssignedPIDs'] = t_pid
        truth_dict['truthHitAssignedDepEnergies'] = t_energy*0
        truth_dict['truthHitAssignedEta'] = t_pos[:, 0:1]
        truth_dict['truthHitAssignedPhi'] = t_pos[:, 1:2]

        feat_dict = self.td.createFeatureDict(feat)


        loss_value = logs['loss']
        print('\n\nBeginning x test')

        print("Loss value", loss_value, self.model.num_train_step, batch)


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

        # data_2  = self.model.num_train_step, cc.numpy(), beta[:, 0].numpy(), t_idx[:, 0].numpy(), feat_dict['recHitEnergy'][:, 0].numpy(), t_energy[:, 0].numpy(), pred_energy[:, 0].numpy(), row_splits[:, 0].numpy()

        self.add(data)



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
        self.plotter.do_plot_to_html(self.output_location, average_over=self.average_over)
        if self.publish is not None:
            publish(self.output_location, self.publish)


class RunningMetricsCallback(RunningMetricsDatabaseAdditionCallback):
    def __init__(self, *args, **kwargs):
        print("Obselete: Please change the name to RunningMetricsDatabaseAdditionCallback instead of RunningMetricsCallback."
              "This is to avoid confusion with RunningMetricsPlotterCallback. This class will be removed in the future.")
        super(RunningMetricsCallback, self).__init__(*args, **kwargs)
