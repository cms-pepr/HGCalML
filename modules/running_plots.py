import numpy as np
import tensorflow as tf
import tensorboard_manager as tm
import threading
import queue
import graph_functions
from DeepJetCore.training.DeepJet_callbacks import PredictCallback
from importlib import reload
graph_functions = reload(graph_functions)
from tensorflow.keras import backend as K
import tensorflow.keras.callbacks.experimental


class Worker(threading.Thread):
    def __init__(self, q, tensorboard_manager, beta_threshold=0.5, dist_threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.tensorboard_manager = tensorboard_manager
        self.beta_threshold = beta_threshold
        self.dist_threshold = dist_threshold

    def run(self):
        while True:
            try:
                data = self.q.get(timeout=3)  # 3s timeout

            except queue.Empty:
                continue

            with tf.device('/cpu:0'):
                step, cc, beta, truth_sid, rechit_energy, truth_energy, pred_energy, row_splits = data
                first_end = int(row_splits[1])

                truth_sid = truth_sid[0:first_end]
                cc = cc[0:first_end, :]
                beta = beta[0:first_end]
                rechit_energy = rechit_energy[0:first_end]
                truth_energy = truth_energy[0:first_end]
                pred_energy = pred_energy[0:first_end]

                _pred_sid, pred_shower_idx = graph_functions.reconstruct_showers(cc, beta, return_alpha_indices=True, beta_threshold=self.beta_threshold, dist_threshold=self.dist_threshold)
                pred_sid = graph_functions.match(truth_sid, _pred_sid, rechit_energy)
                eff, fake_rate = graph_functions.compute_efficiency_and_fake_rate(pred_sid, truth_sid)


                response, sum_response = graph_functions.compute_response_mean(pred_sid, truth_sid, rechit_energy, truth_energy, pred_energy, beta)

                print("\n\nY test start")
                print("XYZ", response, sum_response, eff, fake_rate)
                print("Y test end\n\n")

            dic = dict()
            dic['efficiency'] = eff
            dic['fake_rate'] = fake_rate
            dic['response'] = response
            dic['sum_response'] = sum_response

            self.tensorboard_manager.step(step, dic)

            self.q.task_done()


class RunningEfficiencyFakeRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, td, tensorboard_manager, beta_threshold=0.5, dist_threshold=0.5):
        """Initialize intermediate variables for batches and lists."""
        super().__init__()
        self.td = td
        q = queue.Queue()
        self.q = q
        self.thread = Worker(self.q, tensorboard_manager, beta_threshold=beta_threshold, dist_threshold=dist_threshold).start()

    def add(self, data):
        if self.q.empty():
            print("Putting on")
            self.q.put(data)

    def on_train_batch_end(self, batch, logs=None):
        y_pred = logs['y_pred']
        x = logs['x']

        beta = y_pred[0]
        cc = y_pred[1]
        pred_energy = y_pred[2]

        feat, t_idx, t_energy, t_pos, t_time, t_pid, row_splits = self.td.interpretAllModelInputs(x)

        feat_dict = self.td.createFeatureDict(feat)

        data = batch, cc, beta[:, 0], t_idx[:, 0], feat_dict['recHitEnergy'][:, 0], t_energy[:, 0], pred_energy[:, 0], row_splits[:, 0]

        # print("\n\nX test start")
        # for x in data[1:]:
        #     print(x.shape)
        # print("X test end\n\n")

        self.add(data)


