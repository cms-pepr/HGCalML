import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from hplots.efficiency_fo_truth_energy_plot import EfficiencyFoTruthEnergyPlot
# from hplots.fake_rate_fo_predicted_energy_plot import FakeRateFoPredEnergyPlot
# from hplots.response_fo_truth_energy_plot import ResponseFoTruthEnergy
from hplots.hgcal_analysis_plotter import convert_dataset_dict_elements_to_numpy
from plotting_tools import base_plotter, plotter_3d
from matplotlib.patches import Patch
from ragged_plotting_tools import make_original_truth_shower_plot, make_cluster_coordinates_plot, make_eta_phi_projection_truth_plot
from ragged_plotting_tools import createRandomizedColors
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Process, Queue, Pool
from importlib import reload
import tensorflow as tf
from graph_functions import *
import threading

def build_dataset_analysis_dict():
    data_dict = dict()

    data_dict['beta_threshold'] = -1
    data_dict['distance_threshold'] = -1
    data_dict['iou_threshold'] = -1

    data_dict['truth_shower_energy'] = []
    data_dict['truth_shower_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum'] = []
    data_dict['truth_shower_matched_iou_pred'] = []
    data_dict['truth_shower_matched_iou_ticl'] = []
    data_dict['truth_shower_matched_energy_sum_ticl'] = []
    data_dict['truth_shower_eta'] = []
    data_dict['truth_shower_local_density'] = []
    data_dict['truth_shower_closest_particle_distance'] = []
    data_dict['truth_shower_found_or_not'] = []
    data_dict['truth_shower_found_or_not_ticl'] = []
    data_dict['truth_shower_sample_id'] = []
    data_dict['truth_shower_sid'] = []
    data_dict['truth_shower_matched_energy_regressed'] = []
    data_dict['truth_shower_matched_energy_regressed_ticl'] = []
    data_dict['truth_shower_num_rechits'] = []

    data_dict['endcap_num_rechits'] = []

    data_dict['endcap_total_energy_pred'] = []
    data_dict['endcap_total_energy_ticl'] = []
    data_dict['endcap_total_energy_truth'] = []

    data_dict['pred_shower_regressed_energy'] = []
    data_dict['pred_shower_matched_energy'] = []
    data_dict['pred_shower_energy_sum'] = []
    data_dict['pred_shower_matched_energy_sum'] = []
    data_dict['pred_shower_regressed_phi'] = []
    data_dict['pred_shower_matched_iou'] = []
    data_dict['pred_shower_matched_phi'] = []
    data_dict['pred_shower_regressed_eta'] = []
    data_dict['pred_shower_matched_eta'] = []
    data_dict['pred_shower_sid'] = []
    data_dict['pred_shower_sid_merged'] = []
    data_dict['pred_shower_sample_id'] = []


    data_dict['ticl_shower_regressed_energy'] = []
    data_dict['ticl_shower_matched_energy'] = []
    data_dict['ticl_shower_energy_sum'] = []
    data_dict['ticl_shower_matched_energy_sum'] = []
    data_dict['ticl_shower_regressed_phi'] = []
    data_dict['ticl_shower_matched_phi'] = []
    data_dict['ticl_shower_matched_iou'] = []
    data_dict['ticl_shower_regressed_eta'] = []
    data_dict['ticl_shower_matched_eta'] = []
    data_dict['ticl_shower_sid'] = []
    data_dict['ticl_shower_sid_merged'] = []
    data_dict['ticl_shower_sample_id'] = []

    data_dict['endcap_num_truth_showers'] = []

    data_dict['endcap_num_pred_showers'] = []
    data_dict['endcap_num_found_showers'] = []
    data_dict['endcap_num_missed_showers'] = []
    data_dict['endcap_num_fake_showers'] = []

    data_dict['endcap_num_ticl_showers'] = []
    data_dict['endcap_num_found_showers_ticl'] = []
    data_dict['endcap_num_missed_showers_ticl'] = []
    data_dict['endcap_num_fake_showers_ticl'] = []

    data_dict['visualized_segments'] = []

    return data_dict


def build_endcap_analysis_dict():
    data_dict = dict()

    data_dict['truth_shower_energy'] = []
    data_dict['truth_shower_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum_ticl'] = []
    data_dict['truth_shower_matched_energy_regressed'] = []
    data_dict['truth_shower_matched_energy_regressed_ticl'] = []
    data_dict['truth_shower_matched_iou_pred'] = []
    data_dict['truth_shower_matched_iou_ticl'] = []

    data_dict['truth_shower_eta'] = []
    data_dict['truth_shower_local_density'] = []
    data_dict['truth_shower_closest_particle_distance'] = []
    data_dict['truth_shower_found_or_not'] = []
    data_dict['truth_shower_found_or_not_ticl'] = []
    data_dict['truth_shower_sid'] = []
    data_dict['truth_shower_sample_id'] = []

    data_dict['endcap_total_energy_pred'] = -1
    data_dict['endcap_total_energy_ticl'] = -1
    data_dict['endcap_total_energy_truth'] = -1

    data_dict['truth_shower_num_rechits'] = []
    data_dict['endcap_num_rechits'] = -1
    data_dict['num_showers_per_endcap'] = -1

    data_dict['pred_shower_regressed_energy'] = []
    data_dict['pred_shower_matched_energy'] = []
    data_dict['pred_shower_energy_sum'] = []
    data_dict['pred_shower_matched_energy_sum'] = []
    data_dict['pred_shower_regressed_phi'] = []
    data_dict['pred_shower_matched_phi'] = []
    data_dict['pred_shower_matched_iou'] = []
    data_dict['pred_shower_regressed_eta'] = []
    data_dict['pred_shower_matched_eta'] = []
    data_dict['pred_shower_sid'] = []
    data_dict['pred_shower_sid_merged'] = []
    data_dict['pred_shower_sample_id'] = []


    data_dict['ticl_shower_regressed_energy'] = []
    data_dict['ticl_shower_matched_energy'] = []
    data_dict['ticl_shower_energy_sum'] = []
    data_dict['ticl_shower_matched_energy_sum'] = []
    data_dict['ticl_shower_regressed_phi'] = []
    data_dict['ticl_shower_matched_phi'] = []
    data_dict['ticl_shower_matched_iou'] = []
    data_dict['ticl_shower_regressed_eta'] = []
    data_dict['ticl_shower_matched_eta'] = []
    data_dict['ticl_shower_sid'] = []
    data_dict['ticl_shower_sid_merged'] = []
    data_dict['ticl_shower_sample_id'] = []

    data_dict['found_showers_predicted_truth_rotational_difference'] = []
    data_dict['endcap_num_truth_showers'] = -1
    data_dict['endcap_num_pred_showers'] = -1
    data_dict['endcap_num_found_showers'] = -1
    data_dict['endcap_num_missed_showers'] = -1
    data_dict['endcap_num_fake_showers'] = -1

    data_dict['endcap_num_ticl_showers'] = -1
    data_dict['endcap_num_found_showers_ticl'] = -1
    data_dict['endcap_num_missed_showers_ticl'] = -1
    data_dict['endcap_num_fake_showers_ticl'] = -1

    data_dict['visualization_data'] = -1

    return data_dict

def append_endcap_dict_to_dataset_dict(dataset_dict, endcap_dict):
    dataset_dict['truth_shower_energy'] += endcap_dict['truth_shower_energy']

    dataset_dict['truth_shower_energy_sum'] += endcap_dict['truth_shower_energy_sum']
    dataset_dict['truth_shower_matched_energy_sum'] += endcap_dict['truth_shower_matched_energy_sum']
    dataset_dict['truth_shower_matched_energy_sum_ticl'] += endcap_dict['truth_shower_matched_energy_sum_ticl']
    dataset_dict['truth_shower_matched_iou_pred'] += endcap_dict['truth_shower_matched_iou_pred']
    dataset_dict['truth_shower_matched_iou_ticl'] += endcap_dict['truth_shower_matched_iou_ticl']


    dataset_dict['truth_shower_matched_energy_regressed'] += endcap_dict['truth_shower_matched_energy_regressed']
    dataset_dict['truth_shower_matched_energy_regressed_ticl'] += endcap_dict[
        'truth_shower_matched_energy_regressed_ticl']



    dataset_dict['truth_shower_eta'] += endcap_dict['truth_shower_eta']

    dataset_dict['truth_shower_local_density'] += endcap_dict['truth_shower_local_density']
    dataset_dict['truth_shower_closest_particle_distance'] += endcap_dict['truth_shower_closest_particle_distance']

    dataset_dict['truth_shower_found_or_not'] += endcap_dict['truth_shower_found_or_not']
    dataset_dict['truth_shower_found_or_not_ticl'] += endcap_dict['truth_shower_found_or_not_ticl']
    dataset_dict['truth_shower_sid'] += endcap_dict['truth_shower_sid']
    dataset_dict['truth_shower_sample_id'] += endcap_dict['truth_shower_sample_id']

    dataset_dict['truth_shower_num_rechits'] += endcap_dict['truth_shower_num_rechits']
    dataset_dict['endcap_num_rechits'].append(endcap_dict['endcap_num_rechits'])

    dataset_dict['pred_shower_regressed_energy'] += endcap_dict['pred_shower_regressed_energy']
    dataset_dict['pred_shower_matched_energy'] += endcap_dict['pred_shower_matched_energy']
    dataset_dict['pred_shower_energy_sum'] += endcap_dict['pred_shower_energy_sum']
    dataset_dict['pred_shower_matched_energy_sum'] += endcap_dict['pred_shower_matched_energy_sum']
    dataset_dict['pred_shower_regressed_phi'] += endcap_dict['pred_shower_regressed_phi']
    dataset_dict['pred_shower_matched_phi'] += endcap_dict['pred_shower_matched_phi']
    dataset_dict['pred_shower_matched_iou'] += endcap_dict['pred_shower_matched_iou']
    dataset_dict['pred_shower_regressed_eta'] += endcap_dict['pred_shower_regressed_eta']
    dataset_dict['pred_shower_matched_eta'] += endcap_dict['pred_shower_matched_eta']
    dataset_dict['pred_shower_sid'] += endcap_dict['pred_shower_sid']
    dataset_dict['pred_shower_sid_merged'] += endcap_dict['pred_shower_sid_merged']
    dataset_dict['pred_shower_sample_id'] += endcap_dict['pred_shower_sample_id']

    dataset_dict['ticl_shower_regressed_energy'] += endcap_dict['ticl_shower_regressed_energy']
    dataset_dict['ticl_shower_matched_energy'] += endcap_dict['ticl_shower_matched_energy']
    dataset_dict['ticl_shower_energy_sum'] += endcap_dict['ticl_shower_energy_sum']
    dataset_dict['ticl_shower_matched_energy_sum'] += endcap_dict['ticl_shower_matched_energy_sum']
    dataset_dict['ticl_shower_regressed_phi'] += endcap_dict['ticl_shower_regressed_phi']
    dataset_dict['ticl_shower_matched_phi'] += endcap_dict['ticl_shower_matched_phi']
    dataset_dict['ticl_shower_matched_iou'] += endcap_dict['ticl_shower_matched_iou']
    dataset_dict['ticl_shower_regressed_eta'] += endcap_dict['ticl_shower_regressed_eta']
    dataset_dict['ticl_shower_matched_eta'] += endcap_dict['ticl_shower_matched_eta']
    dataset_dict['ticl_shower_sid'] += endcap_dict['ticl_shower_sid']
    dataset_dict['ticl_shower_sid_merged'] += endcap_dict['ticl_shower_sid_merged']
    dataset_dict['ticl_shower_sample_id'] += endcap_dict['ticl_shower_sample_id']

    dataset_dict['endcap_num_truth_showers'].append(endcap_dict['endcap_num_truth_showers'])
    dataset_dict['endcap_num_pred_showers'].append(endcap_dict['endcap_num_pred_showers'])
    dataset_dict['endcap_num_found_showers'].append(endcap_dict['endcap_num_found_showers'])
    dataset_dict['endcap_num_missed_showers'].append(endcap_dict['endcap_num_missed_showers'])
    dataset_dict['endcap_num_fake_showers'].append(endcap_dict['endcap_num_fake_showers'])

    dataset_dict['endcap_num_ticl_showers'].append(endcap_dict['endcap_num_ticl_showers'])
    dataset_dict['endcap_num_found_showers_ticl'].append(endcap_dict['endcap_num_found_showers_ticl'])
    dataset_dict['endcap_num_missed_showers_ticl'].append(endcap_dict['endcap_num_missed_showers_ticl'])
    dataset_dict['endcap_num_fake_showers_ticl'].append(endcap_dict['endcap_num_fake_showers_ticl'])

    dataset_dict['endcap_total_energy_pred'].append(endcap_dict['endcap_total_energy_pred'])
    dataset_dict['endcap_total_energy_ticl'].append(endcap_dict['endcap_total_energy_ticl'])
    dataset_dict['endcap_total_energy_truth'].append(endcap_dict['endcap_total_energy_truth'])

    if endcap_dict['visualization_data'] != -1:
        dataset_dict['visualized_segments'].append(endcap_dict['visualization_data'])

    return dataset_dict


def build_endcap_visualization_dict():
    vis_dict = dict()
    vis_dict['truth_sid'] = -1

    vis_dict['pred_and_truth_dict'] = -1
    vis_dict['feature_dict'] = -1

    vis_dict['pred_sid'] = -1
    vis_dict['ticl_sid'] = -1
    vis_dict['coords_representatives'] = -1
    vis_dict['identified_vertices'] = -1

    return vis_dict


def find_uniques_from_betas(betas, coords, dist_threshold, beta_threshold, soft):
    # print('find_uniques_from_betas')

    with tf.device('/cpu:0'):
        from condensate_op import BuildCondensates
        '''
    
        .Output("asso_idx: int32")
        .Output("is_cpoint: int32")
        .Output("n_condensates: int32");
        '''
        print("here 1")
        row_splits = np.array([0, len(betas)], dtype='int32')
        print("here 2")
        asso_idx, is_cpoint, _ = BuildCondensates(coords, betas, row_splits,
                                                  radius=dist_threshold, min_beta=beta_threshold, soft=soft)

        allidxs = np.array(range(len(betas)))
        print("here 2.1")
        # allidxs = np.arange(len(betas))
        representative_indices = allidxs[is_cpoint > 0]

        print("here 3")
        # this should be fixed downstream. and index to an array of indicies to get a vertex is not a good way to code this!
        labels = asso_idx.numpy().copy()

        print("here 4")
        for i in range(len(representative_indices)):
            ridx = representative_indices[i]
            labels[asso_idx == ridx] = i

        print("here 5")

        return labels, representative_indices, asso_idx.numpy()


def calculate_iou_main_thread(queue):
    while True:
        try:
            data = queue.get()  # 3s timeout
            if data is None:
                break

            data, send_address = data

            result = calculate_iou_tf(data[0], data[1], data[2], data[3], data[4], data[5])

            send_address.send(result)

        except queue.Empty:
            continue


# @tf.function
def calculate_iou_tf(truth_sid,
                     pred_sid,
                     truth_shower_sid,
                     pred_shower_sid,
                     hit_weight,
                     iou_threshold):
    with tf.device('/cpu:0'):
        # print("1")
        truth_sid = tf.cast(tf.convert_to_tensor(truth_sid), tf.int32)
        pred_sid = tf.cast(tf.convert_to_tensor(pred_sid), tf.int32)
        hit_weight = tf.cast(tf.convert_to_tensor(hit_weight), tf.float32)
        # print("2")

        truth_shower_sid = tf.cast(tf.convert_to_tensor(truth_shower_sid), tf.int32)
        pred_shower_sid = tf.cast(tf.convert_to_tensor(pred_shower_sid), tf.int32)
        len_pred_showers = len(pred_shower_sid)
        len_truth_showers = len(truth_shower_sid)

        # print("3")
        truth_idx_2 = tf.zeros_like(truth_sid)
        pred_idx_2 = tf.zeros_like(pred_sid)
        hit_weight_2 = tf.zeros_like(hit_weight)

        # print("3.1")

        for i in range(len(pred_shower_sid)):
            # print("dum dum dum")
            pred_idx_2 = tf.where(pred_sid == pred_shower_sid[i], i, pred_idx_2)
            # print("dum dum dum 2")

        # print("3.2")

        for i in range(len(truth_shower_sid)):
            truth_idx_2 = tf.where(truth_sid == truth_shower_sid[i], i, truth_idx_2)

        # print("3.3")
        one_hot_pred = tf.one_hot(pred_idx_2, depth=len_pred_showers)
        one_hot_truth = tf.one_hot(truth_idx_2, depth=len_truth_showers)

        intersection_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)
        # cross_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], tf.ones_like(one_hot_truth),
        #                                 transpose_a=True)

        pred_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], tf.ones_like(one_hot_truth),
                                        transpose_a=True)

        truth_sum_matrix = tf.linalg.matmul(
            tf.ones_like(one_hot_pred) * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)

        union_sum_matrix = pred_sum_matrix + truth_sum_matrix - intersection_sum_matrix


        overlap_matrix = (intersection_sum_matrix / union_sum_matrix).numpy()
        pred_shower_sid = pred_shower_sid.numpy()
        truth_shower_sid = truth_shower_sid.numpy()

        # print(overlap_matrix.shape)
        # 0/0

        all_iou = []
        for i in range(len_pred_showers):
            for j in range(len_truth_showers):

                overlap = overlap_matrix[i, j]

                if overlap > iou_threshold:
                    if pred_shower_sid[i] == -1 or truth_shower_sid[j] == -1:
                        continue
                    all_iou.append((pred_shower_sid[i], truth_shower_sid[j], overlap))
                    # G.add_edge('p%d'%index_shower_prediction, 't%d'%unique_showers_this_segment[i], weight=overlap)
        # print("6")
        return all_iou, overlap_matrix, pred_sum_matrix.numpy(), truth_sum_matrix.numpy(), intersection_sum_matrix.numpy()



class HGCalAnalyzer:
    def __init__(self, features, truth, predictions, beta_threshold,
                 distance_threshold, iou_threshold, endcap_id, should_return_visualization_data=False, soft=False, main_thread=None, use_ticl=False, with_local_distance_scaling=False):

        self.with_local_distance_scaling=with_local_distance_scaling
        self.truth_dict = truth
        # self.pred_dict = index_dicts.create_ragged_cal_pred_dict(prediction, n_ccoords=3)
        self.feat_dict = features
        self.predictions_dict = predictions

        truth_sid = self.truth_dict['truthHitAssignementIdx'][:, 0]

        self.endcap_id = endcap_id

        self.truth_shower_sid, unique_truth_shower_hit_idx = np.unique(truth_sid, return_index=True)

        unique_truth_shower_hit_idx = unique_truth_shower_hit_idx[self.truth_shower_sid!=-1]
        self.should_return_visualization_data = should_return_visualization_data
        self.truth_shower_sid = self.truth_shower_sid[self.truth_shower_sid!=-1]

        self.sid_to_truth_shower_sid_idx = dict()
        for i in range(len(self.truth_shower_sid)):
            self.sid_to_truth_shower_sid_idx[self.truth_shower_sid[i]] = i

        self.truth_dep_energy = self.truth_dict['truthHitAssignedDepEnergies'][:, 0]  # Flatten it

        self.truth_shower_energy = self.truth_dict['truthHitAssignedEnergies'][:, 0][unique_truth_shower_hit_idx]
        self.truth_shower_eta = self.truth_dict['truthHitAssignedEta'][unique_truth_shower_hit_idx][:, 0]
        self.truth_shower_phi = self.truth_dict['truthHitAssignedPhi'][unique_truth_shower_hit_idx][:, 0]

        self.hit_energy = self.feat_dict['recHitEnergy'][:, 0]

        self.pred_beta = self.predictions_dict['pred_beta'][:, 0]
        self.pred_energy = self.predictions_dict['pred_energy'][:, 0]

        # self.pred_x = (self.pred_dict['predX'] + self.feat_dict["recHitX"])[:, 0]
        # self.pred_y = (self.pred_dict['predY'] + self.feat_dict["recHitY"])[:, 0]

        self.pred_position = self.predictions_dict['pred_pos']
        self.pred_ccoords = self.predictions_dict['pred_ccoords']


        self.pred_dist = self.predictions_dict['pred_dist']

        self.truth_energy = self.truth_dict['truthHitAssignedEnergies'][:, 0]

        # self.pred_and_truth_dict = pred_and_truth_dict
        self.beta_threshold = beta_threshold
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.is_soft = soft
        self.truth_sid = truth_sid
        self.results_dict = build_endcap_analysis_dict()
        self.use_ticl = use_ticl

        self.main_thread = main_thread

    def match_to_truth(self, truth_sid, pred_sid, pred_shower_sid, hit_weight, truth_shower_energy, pred_shower_energy, obc=False):
        pred_shower_sid = np.array(pred_shower_sid, np.int32)

        # print("Y", pred_shower_sid.shape)
        pred_shower_sid_to_energy = {}
        truth_shower_sid_to_energy = {}
        truth_shower_sid_to_truth_shower_idx = {}
        pred_shower_sid_to_pred_shower_idx = {}

        truth_shower_sid_to_pred_shower_sid = {}
        truth_shower_sid = np.unique(truth_sid)
        for i,x in enumerate(truth_shower_sid):
            truth_shower_sid_to_pred_shower_sid[x] = -1
            truth_shower_sid_to_truth_shower_idx[x] = i
            if x != -1:
                truth_shower_sid_to_energy[x] = truth_shower_energy[i-1]

        pred_shower_sid_to_truth_shower_sid = {}
        pred_shower_sid_to_iou = {}

        for i,x in enumerate(pred_shower_sid):
            pred_shower_sid_to_truth_shower_sid[x] = -1
            pred_shower_sid_to_pred_shower_idx[x] = i
            pred_shower_sid_to_energy[x] = pred_shower_energy[i]
            pred_shower_sid_to_iou[x] =  -1


        G = nx.Graph()
        # print("Around here")

        if self.main_thread is None:
            all_iou, iou_matrix, pred_sum_matrix, truth_sum_matrix, intersection_matrix = calculate_iou_tf(truth_sid, pred_sid,
                                                                                                       truth_shower_sid, pred_shower_sid, hit_weight, self.iou_threshold)
        else:
            tf_connection, executor_connection = multiprocessing.Pipe()
            self.main_thread.put(( (truth_sid, pred_sid, truth_shower_sid, pred_shower_sid, hit_weight, self.iou_threshold), tf_connection))
            all_iou, iou_matrix, pred_sum_matrix, truth_sum_matrix, intersection_matrix = executor_connection.recv()

        # print("Done")
        for iou in all_iou:
            # print(iou)
            G.add_edge('p%d' % iou[0], 't%d' % iou[1], weight=iou[2])


        X = nx.algorithms.max_weight_matching(G)
        for x, y in X:
            if x[0] == 'p':
                prediction_index = int(x[1:])
                truth_index = int(y[1:])

            else:
                truth_index = int(x[1:])
                prediction_index = int(y[1:])
            _iou = G.get_edge_data(x, y)['weight']
            # print(x,y, truth_index, prediction_index)

            truth_shower_sid_to_pred_shower_sid[truth_index] = prediction_index
            pred_shower_sid_to_truth_shower_sid[prediction_index] = truth_index
            pred_shower_sid_to_iou[prediction_index] = _iou

        new_indicing = np.max(truth_shower_sid) + 1
        pred_sid_2 = np.zeros_like(pred_sid, np.int32) - 1


        pred_shower_sid_2 = []
        pred_shower_sid_3 = []
        pred_shower_matched_iou = []

        for k in pred_shower_sid:
            v = pred_shower_sid_to_truth_shower_sid[k]
            if v != -1:
                pred_sid_2[pred_sid == k] = v
                pred_shower_sid_2.append(v)
                pred_shower_matched_iou.append(pred_shower_sid_to_iou[k])
            else:
                pred_sid_2[pred_sid == k] = new_indicing
                pred_shower_sid_2.append(new_indicing)
                new_indicing += 1
                pred_shower_matched_iou.append(-1.)


        ########################################################################
        ################### INVESTIGATION PART #################################

        pred_sid_3 = np.zeros_like(pred_sid, np.int32) - 1
        for idx_B, k in enumerate(pred_shower_sid):
            v = pred_shower_sid_to_truth_shower_sid[k]
            energy_B = pred_shower_sid_to_energy[k]

            if v == -1:
                # print()
                iou_row = iou_matrix[idx_B]

                idx_T = np.argmax(iou_row)
                next_best_sid = truth_shower_sid[idx_T]

                if next_best_sid != -1:
                    energy_truth = truth_shower_sid_to_energy[next_best_sid]

                    sid_A = truth_shower_sid_to_pred_shower_sid[next_best_sid]
                    if sid_A!=-1:
                        energy_A = pred_shower_sid_to_energy[sid_A]
                        idx_A = pred_shower_sid_to_pred_shower_idx[sid_A]
                        old_iou = iou_matrix[idx_A, idx_T]

                        # print(intersection_matrix.shape, truth_sum_matrix.shape, pred_sum_matrix.shape, idx_A, idx_B, idx_T, len(truth_shower_sid), len(pred_shower_sid))
                        new_iou = (intersection_matrix[idx_B, idx_T] + intersection_matrix[idx_A, idx_T])
                        new_iou = new_iou / (truth_sum_matrix[0, idx_T] +pred_sum_matrix[idx_A, 0] +pred_sum_matrix[idx_B, 0] - intersection_matrix[idx_A, idx_T] - intersection_matrix[idx_B, idx_T])

                        if new_iou > old_iou and (energy_truth - energy_A)**2 > (energy_truth - energy_A - energy_B) **2:
                            v = next_best_sid

                    #     # print(energy_truth, energy_A, energy_B, old_iou, new_iou)
                    # else:
                    #     print(energy_truth, energy_B)


                    iou_row = np.sort(iou_row)
                    # print(iou_row[iou_row>0])
                    max_fake_iou = np.max(iou_row)

            if v == -1:
                pred_shower_sid_3.append(pred_shower_sid_2[idx_B])
                pred_sid_3[pred_sid == k] = pred_shower_sid_2[idx_B]

            else:
                pred_shower_sid_3.append(v)
                pred_sid_2[pred_sid == k] = v

        ########################################################################

        return pred_sid_2, pred_shower_sid_2, pred_shower_matched_iou, pred_sid_3, pred_shower_sid_3

    def compute_and_match_predicted_showers(self):

        # pred_shower_representative_hit_idxx = np.arange(len(self.pred_beta))
        # clustering_coords_all_filtered = self.pred_ccoords[self.pred_beta>self.beta_threshold]
        # beta_fil = self.pred_beta[self.pred_beta>self.beta_threshold]
        # pred_shower_representative_hit_idxx = pred_shower_representative_hit_idxx[self.pred_beta>self.beta_threshold]

        # print("Gathering")

        # pred_sidx, pred_shower_representative_hit_idxx, assoidx = find_uniques_from_betas(self.pred_beta,
        #                                                                                 self.pred_ccoords,
        #                                                                                 dist_threshold=self.distance_threshold,
        #                                                                                 beta_threshold=self.beta_threshold,
        #                                                                                 soft=self.is_soft)


        # if 'pred_dist' in self.predictions_dict:
        if self.with_local_distance_scaling:
            # print("Doing with pred dist")
            pred_sid, pred_shower_representative_hit_idx = reconstruct_showers(self.pred_ccoords, self.pred_beta, self.beta_threshold, self.distance_threshold, return_alpha_indices=True, limit=1000, pred_dist=self.predictions_dict['pred_dist'][:, 0])
        else:
            pred_sid, pred_shower_representative_hit_idx = reconstruct_showers(self.pred_ccoords, self.pred_beta, self.beta_threshold, self.distance_threshold, return_alpha_indices=True, limit=1000)
        # else:
        #     0/0 # TODO: remove
        #     pred_sid, pred_shower_representative_hit_idx = reconstruct_showers(self.pred_ccoords, self.pred_beta, self.beta_threshold, self.distance_threshold, return_alpha_indices=True, limit=1000)


        # print("Gathered")

        # pred_shower_representative_hit_idx = np.array([int(pred_shower_representative_hit_idxx[i]) for i in pred_shower_representative_hit_idx])
        # pred_sid = assign_prediction_labels_to_full_unfiltered_vertices(self.pred_beta, self.pred_ccoords, pred_sid,
        #                                                                       clustering_coords_all_filtered,
        #                                                                       beta_fil,
        #                                                                       distance_threshold=self.distance_threshold)

        self.pred_shower_representative_hit_idx = pred_shower_representative_hit_idx
        self.pred_shower_energy = [self.pred_energy[x] for x in pred_shower_representative_hit_idx]

        pred_shower_sid = [pred_sid[x] for x in pred_shower_representative_hit_idx]

        # print("XYZ", np.array(pred_shower_sid).shape, pred_sid.shape, pred_shower_representative_hit_idx.shape)
        pred_representative_coords = np.array([self.pred_ccoords[x] for x in pred_shower_representative_hit_idx])

        self.pred_sid, self.pred_shower_sid, self.pred_shower_matched_iou, self._pred_sid_merged, self.pred_shower_sid_merged = self.match_to_truth(self.truth_sid, pred_sid, pred_shower_sid,
                                                                                                                      self.hit_energy, self.truth_shower_energy, self.pred_shower_energy, obc=True)

        # print(num_total_showers, num_total_fakes)

        self.pred_shower_representative_coords = pred_representative_coords
        self.sid_to_pred_shower_sid_idx = dict()

        for i in range(len(self.pred_shower_sid)):
            self.sid_to_pred_shower_sid_idx[self.pred_shower_sid[i]] = i

    def match_ticl_showers(self):
        ticl_sid = self.truth_dict['ticlHitAssignementIdx'][:, 0]
        ticl_energy = self.truth_dict['ticlHitAssignedEnergies'][:, 0]


        ticl_sid[ticl_energy > 10 * self.truth_dep_energy] = 0
        ticl_energy[ticl_energy > 10 * self.truth_dep_energy] = 0

        ticl_shower_sid, ticl_shower_idx = np.unique(ticl_sid[ticl_sid >= 0], return_index=True)
        ticl_energy_2 = ticl_energy[ticl_sid >= 0]

        self.ticl_shower_energy = [ticl_energy_2[x] for x in ticl_shower_idx]

        self.ticl_sid, self.ticl_shower_sid, self.ticl_shower_matched_iou, self._ticl_sid_merged, self.ticl_shower_sid_merged = self.match_to_truth(self.truth_sid, ticl_sid, ticl_shower_sid,
                                                                                                                      self.hit_energy, self.truth_shower_energy, self.ticl_shower_energy)
        self.sid_to_ticl_shower_sid_idx = dict()
        for i in range(len(self.ticl_shower_sid)):
            self.sid_to_ticl_shower_sid_idx[self.ticl_shower_sid[i]] = i

    def gather_truth_matchings(self):
        num_found = 0.
        num_found_ticl = 0.
        num_missed = 0.
        num_missed_ticl = 0.
        num_gt_showers = 0.

        predicted_truth_total = 0

        for i in range(len(self.truth_shower_sid)):
            sid = self.truth_shower_sid[i]

            self.results_dict['truth_shower_sid'].append(sid)
            self.results_dict['truth_shower_energy'].append(self.truth_shower_energy[i])
            sum = self.truth_energy[self.truth_sid == sid]
            self.results_dict['truth_shower_energy_sum'].append(sum)
            predicted_truth_total += self.truth_shower_energy[i]

            self.results_dict['truth_shower_eta'].append(self.truth_shower_eta[i])

            try:
                distances_with_other_truths = np.sqrt((self.truth_shower_eta[i] - self.truth_shower_eta) ** 2 + (
                    np.arctan2(np.sin(self.truth_shower_phi[i] - self.truth_shower_phi),
                               np.cos(self.truth_shower_phi[i] - self.truth_shower_phi))) ** 2)
                distances_with_other_truths_excluding_me = distances_with_other_truths[distances_with_other_truths != 0]
                distance_to_closest_truth = np.min(distances_with_other_truths_excluding_me)
                density = self.truth_shower_energy[i] / (
                            np.sum(self.truth_shower_energy[distances_with_other_truths < 0.5]) + 1e-6)
            except ValueError:
                density = 0
                distance_to_closest_truth = 0
            self.results_dict['truth_shower_local_density'].append(density)
            self.results_dict['truth_shower_closest_particle_distance'].append(distance_to_closest_truth)

            pred_match_shower_idx = self.sid_to_pred_shower_sid_idx[
                sid] if sid in self.sid_to_pred_shower_sid_idx else -1
            if pred_match_shower_idx > -1:
                predicted_energy = self.pred_shower_energy[pred_match_shower_idx]

                self.results_dict['truth_shower_found_or_not'].append(True)
                self.results_dict['truth_shower_matched_energy_sum'].append(
                    np.sum(self.hit_energy[self.pred_sid == sid]))
                self.results_dict['truth_shower_matched_energy_regressed'].append(predicted_energy)
                self.results_dict['truth_shower_matched_iou_pred'].append(self.pred_shower_matched_iou[pred_match_shower_idx])

                num_found += 1
            else:
                self.results_dict['truth_shower_found_or_not'].append(False)
                self.results_dict['truth_shower_matched_energy_sum'].append(-1)
                self.results_dict['truth_shower_matched_energy_regressed'].append(-1)
                self.results_dict['truth_shower_matched_iou_pred'].append(-1)

                num_missed += 1


            if self.use_ticl:
                ticl_match_shower_idx = self.sid_to_ticl_shower_sid_idx[
                    sid] if sid in self.sid_to_ticl_shower_sid_idx else -1
                if ticl_match_shower_idx > -1:
                    predicted_energy = self.ticl_shower_energy[ticl_match_shower_idx]

                    self.results_dict['truth_shower_found_or_not_ticl'].append(True)
                    self.results_dict['truth_shower_matched_energy_sum_ticl'].append(
                        np.sum(self.hit_energy[self.ticl_sid == sid]))
                    self.results_dict['truth_shower_matched_energy_regressed_ticl'].append(predicted_energy)

                    self.results_dict['truth_shower_matched_iou_ticl'].append(self.ticl_shower_matched_iou[ticl_match_shower_idx])

                    num_found_ticl += 1
                else:
                    self.results_dict['truth_shower_found_or_not_ticl'].append(False)
                    self.results_dict['truth_shower_matched_energy_sum_ticl'].append(-1)
                    self.results_dict['truth_shower_matched_energy_regressed_ticl'].append(-1)
                    self.results_dict['truth_shower_matched_iou_ticl'].append(-1)


                    num_missed_ticl += 1

            num_gt_showers += 1
            self.results_dict['truth_shower_num_rechits'].append(len(self.truth_sid[self.truth_sid == sid]))

        self.results_dict['endcap_num_rechits'] = len(self.truth_sid)
        # self.results_dict['num_showers_per_endcap']
        #
        # global endcap_id
        self.results_dict['truth_shower_sample_id'] = (
                self.endcap_id + 0 * np.array(self.results_dict['truth_shower_energy'])).tolist()

        self.results_dict['endcap_num_truth_showers'] = num_gt_showers
        self.results_dict['endcap_num_found_showers'] = num_found
        self.results_dict['endcap_num_missed_showers'] = num_missed

        if self.use_ticl:
            self.results_dict['endcap_num_found_showers_ticl'] = num_found_ticl
            self.results_dict['endcap_num_missed_showers_ticl'] = num_missed_ticl

        self.results_dict['endcap_total_energy_truth'] = predicted_truth_total

    def gather_prediction_matchings(self):

        predicted_total_obc = 0

        num_fakes = 0
        num_predicted_showers = 0

        for i in range(len(self.pred_shower_sid)):
            sid = self.pred_shower_sid[i]
            sid2 = self.pred_shower_sid_merged[i]
            rep_idx = self.pred_shower_representative_hit_idx[i]

            shower_energy_predicted = self.pred_energy[rep_idx]
            predicted_total_obc += shower_energy_predicted

            shower_eta_predicted = 0 #self.pred_x[rep_idx]
            shower_phi_predicted = 0 #self.pred_y[rep_idx]
            shower_energy_sum_predicted = np.sum(self.hit_energy[self.pred_sid == sid])

            self.results_dict['pred_shower_sid'].append(sid)
            # print(self.results_dict)
            self.results_dict['pred_shower_sid_merged'].append(sid2)
            self.results_dict['pred_shower_regressed_energy'].append(shower_energy_predicted)
            self.results_dict['pred_shower_energy_sum'].append(shower_energy_sum_predicted)
            self.results_dict['pred_shower_regressed_phi'].append(shower_phi_predicted)
            self.results_dict['pred_shower_regressed_eta'].append(shower_eta_predicted)
            self.results_dict['pred_shower_matched_iou'].append(self.pred_shower_matched_iou[i])

            truth_match_shower_idx = self.sid_to_truth_shower_sid_idx[
                sid] if sid in self.sid_to_truth_shower_sid_idx else -1

            if truth_match_shower_idx > -1:
                shower_energy_truth = self.truth_shower_energy[truth_match_shower_idx]
                shower_eta_truth = self.truth_shower_eta[truth_match_shower_idx]
                shower_phi_truth = self.truth_shower_phi[truth_match_shower_idx]
                # print(self.truth_shower_sid[truth_match_shower_idx], sid)

                shower_energy_sum_truth = np.sum(self.hit_energy[self.truth_sid == sid])

                self.results_dict['pred_shower_matched_energy'].append(shower_energy_truth)
                self.results_dict['pred_shower_matched_energy_sum'].append(shower_energy_sum_truth)
                self.results_dict['pred_shower_matched_phi'].append(shower_phi_truth)
                self.results_dict['pred_shower_matched_eta'].append(shower_eta_truth)

                # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

            else:
                num_fakes += 1
                self.results_dict['pred_shower_matched_energy'].append(-1)
                self.results_dict['pred_shower_matched_energy_sum'].append(-1)
                self.results_dict['pred_shower_matched_phi'].append(-1)
                self.results_dict['pred_shower_matched_eta'].append(-1)

            self.results_dict['pred_shower_sample_id'] = (
                    self.endcap_id + 0 * np.array(self.results_dict['pred_shower_regressed_energy'])).tolist()
            num_predicted_showers += 1

            self.results_dict['endcap_num_fake_showers'] = num_fakes
            self.results_dict['endcap_num_pred_showers'] = num_predicted_showers
            self.results_dict['endcap_total_energy_pred'] = predicted_total_obc

    def gather_visualization_data(self):
        vis_dict = build_endcap_visualization_dict()

        print("Returning vis data")
        vis_dict['truth_sid'] = self.truth_sid # replace(truth_id, replace_dictionary)

        # replace_dictionary = copy.deepcopy(pred_shower_sid_to_truth_shower_sid)
        # replace_dictionary_2 = copy.deepcopy(replace_dictionary)
        # start_secondary_indicing_from = np.max(truth_sid) + 1
        # for k, v in replace_dictionary.items():
        #     if v == -1 and k != -1:
        #         replace_dictionary_2[k] = start_secondary_indicing_from
        #         start_secondary_indicing_from += 1
        # replace_dictionary = replace_dictionary_2
        #
        # # print("===============")
        # # print(np.unique(pred_sid), replace_dictionary)
        vis_dict['pred_sid'] = self.pred_sid
        if self.use_ticl:
            vis_dict['ticl_sid'] = self.ticl_sid

        # print(predicted_showers_found)

        # print(np.mean((pred_sid==-1)).astype(np.float))

        # if use_ticl:
        #     replace_dictionary = copy.deepcopy(ticl_shower_sid_to_truth_shower_sid)
        #     replace_dictionary_2 = copy.deepcopy(replace_dictionary)
        #     start_secondary_indicing_from = np.max(truth_sid) + 1
        #     for k, v in replace_dictionary.items():
        #         if v == -1 and k != -1:
        #             replace_dictionary_2[k] = start_secondary_indicing_from
        #             start_secondary_indicing_from += 1
        #     replace_dictionary = replace_dictionary_2
        #
        #     # print(np.unique(ticl_sid), replace_dictionary)
        #     vis_dict['ticl_showers'] = replace(ticl_sid, replace_dictionary)
        # else:
        #     vis_dict['ticl_showers'] = pred_sid * 10

        vis_dict['pred_and_truth_dict'] = self.pred_and_truth_dict
        vis_dict['feature_dict'] = self.feature_dict

        vis_dict['coords_representatives'] = self.pred_shower_representative_coords
        vis_dict['identified_vertices'] = self.pred_shower_representative_hit_idx

        self.results_dict['visualization_data'] = vis_dict

    def gather_ticl_matchings(self):
        predicted_total_ticl = 0

        num_fakes_ticl = 0
        num_predicted_showers_ticl = 0

        for i in range(len(self.ticl_shower_sid)):
            sid = self.ticl_shower_sid[i]
            sid2 = self.ticl_shower_sid_merged[i]
            shower_energy_predicted = self.ticl_shower_energy[i]
            predicted_total_ticl += shower_energy_predicted

            shower_energy_sum_predicted = np.sum(self.hit_energy[self.ticl_sid == sid])

            self.results_dict['ticl_shower_sid'].append(sid)
            self.results_dict['ticl_shower_sid_merged'].append(sid2)
            self.results_dict['ticl_shower_regressed_energy'].append(shower_energy_predicted)
            self.results_dict['ticl_shower_energy_sum'].append(shower_energy_sum_predicted)
            self.results_dict['ticl_shower_regressed_phi'].append(0)
            self.results_dict['ticl_shower_regressed_eta'].append(0)
            self.results_dict['ticl_shower_matched_iou'].append(self.ticl_shower_matched_iou[i])

            truth_match_shower_idx = self.sid_to_truth_shower_sid_idx[
                sid] if sid in self.sid_to_truth_shower_sid_idx else -1

            if truth_match_shower_idx > -1:
                shower_energy_truth = self.truth_shower_energy[truth_match_shower_idx]
                shower_eta_truth = self.truth_shower_eta[truth_match_shower_idx]
                shower_phi_truth = self.truth_shower_phi[truth_match_shower_idx]

                shower_energy_sum_truth = np.sum(self.hit_energy[self.truth_sid == sid])

                self.results_dict['ticl_shower_matched_energy'].append(shower_energy_truth)
                self.results_dict['ticl_shower_matched_energy_sum'].append(shower_energy_sum_truth)
                self.results_dict['ticl_shower_matched_phi'].append(shower_phi_truth)
                self.results_dict['ticl_shower_matched_eta'].append(shower_eta_truth)

                # print(shower_eta_predicted, shower_eta_truth, shower_phi_predicted, shower_phi_truth)

            else:
                num_fakes_ticl += 1
                self.results_dict['ticl_shower_matched_energy'].append(-1)
                self.results_dict['ticl_shower_matched_energy_sum'].append(-1)
                self.results_dict['ticl_shower_matched_phi'].append(-1)
                self.results_dict['ticl_shower_matched_eta'].append(-1)

            self.results_dict['ticl_shower_sample_id'] = (
                    self.endcap_id + 0 * np.array(self.results_dict['ticl_shower_regressed_energy'])).tolist()
            num_predicted_showers_ticl += 1

            self.results_dict['endcap_num_fake_showers_ticl'] = num_fakes_ticl
            self.results_dict['endcap_num_ticl_showers'] = num_predicted_showers_ticl
            self.results_dict['endcap_total_energy_ticl'] = predicted_total_ticl

    def analyse(self):
        # print("hello there")
        self.compute_and_match_predicted_showers()

        # print("hello there again")
        if self.use_ticl:
            self.match_ticl_showers()

        # print("what's up")
        self.gather_truth_matchings()
        # print("hehe hehehe")
        self.gather_prediction_matchings()
        if self.use_ticl:
            self.gather_ticl_matchings()
        if self.should_return_visualization_data:
            self.gather_visualization_data()

        return self.results_dict




def make_truth_energy_histogram(plt, ax, truth_energies):
    plt.figure()
    plt.hist(truth_energies, bins=50, histtype='step')
    plt.xlabel("Truth shower energy")
    plt.ylabel("Frequency")
    plt.title('Truth energies')


def histogram_total_endcap_resolution(plt, ax, total_truth_energies, total_obc_energies, total_ticl_energies,
                                      energy_filter=0):
    if energy_filter > 0:
        filter = total_truth_energies > energy_filter

        total_truth_energies = total_truth_energies[filter]
        total_obc_energies = total_obc_energies[filter]

        if total_ticl_energies is not None:
            total_ticl_energies = total_ticl_energies[filter]

    if total_ticl_energies is not None:
        response_ticl = total_ticl_energies / total_truth_energies

    response_obc = total_obc_energies / total_truth_energies

    # bins = np.linspace(0,3.001,40)
    if total_ticl_energies is not None:
        response_ticl[response_ticl > 3] = 3
    response_obc[response_obc > 3] = 3

    plt.subplots(figsize=(7, 5.4))
    plt.axvline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
    plt.hist(response_obc, bins=20, histtype='step', label='Object condensation', density=True)
    if total_ticl_energies is not None:
        plt.hist(response_ticl, bins=20, histtype='step', label='ticl', density=True)
    plt.xlabel("Response (predicted showers' energy sum / truth showers' energy sum)")
    plt.ylabel("Frequency")

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter

    plt.title('Total energy response over a endcap' + sx)
    plt.legend()


def make_fake_energy_regressed_histogram(plt, ax, regressed_energy, ticl=False):
    plt.figure()
    plt.hist(regressed_energy, bins=50, histtype='step')
    plt.xlabel("Fake shower energy regressed")
    plt.ylabel("Frequency")
    plt.title('Fakes energy histogram' + (' - ticl' if ticl else ''))


def make_fake_energy_sum_histogram(plt, ax, predicted_energy_sum, ticl=False):
    predicted_energy_sum[predicted_energy_sum > 60] = 60
    plt.figure()
    plt.hist(predicted_energy_sum, bins=50, histtype='step')
    plt.xlabel("Fake shower energy rechit sum")
    plt.ylabel("Frequency")
    plt.title('Fakes energy histogram' + (' - ticl' if ticl else ''))


def make_response_histograms(plt, ax, found_showers_predicted_sum, found_showers_truth_sum,
                             found_showers_predicted_energies, found_showers_target_energies, ticl=False):
    found_showers_predicted_sum = np.array(found_showers_predicted_sum)
    found_showers_truth_sum = np.array(found_showers_truth_sum)
    found_showers_target_energies = np.array(found_showers_target_energies)
    found_showers_predicted_energies = np.array(found_showers_predicted_energies)

    response_rechit_sum_energy = found_showers_predicted_sum / found_showers_truth_sum
    response_rechit_sum_energy[response_rechit_sum_energy > 3] = 3

    response_energy_predicted = found_showers_predicted_energies / found_showers_target_energies
    response_energy_predicted[response_energy_predicted > 3] = 3
    response_energy_predicted[response_energy_predicted < 0.1] = 0.1

    data_dict = {}
    plt.figure()
    plt.hist(response_rechit_sum_energy, bins=20, histtype='step')
    plt.hist(response_energy_predicted, bins=20, histtype='step')
    plt.legend(['predicted shower sum / truth shower sum', 'predicted energy / target energy'])
    plt.xlabel("Predicted/truth")
    plt.ylabel("Frequency")
    plt.title('Response curves' + (' - ticl' if ticl else ''))


def make_response_histograms_energy_segmented(plt, ax, _found_showers_predicted_sum, _found_showers_truth_sum,
                                              _found_showers_predicted_energies, _found_showers_target_energies,
                                              ticl=False):
    energy_segments = [0, 5, 10, 20, 30, 50, 100, 200, 300, 3000]
    names = ['0-5', '5-10', '10-20', '20-30', '30-50', '50-100', '100-200', '200-300', '300+']
    if ticl:
        names = ['Energy = %s Gev - ticl' % s for s in names]
    else:
        names = ['Energy = %s Gev' % s for s in names]

    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(3, 3)

    ax = [[fig.add_subplot(gs[0, 0]),
           fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[0, 2]), ],

          [fig.add_subplot(gs[1, 0]),
           fig.add_subplot(gs[1, 1]),
           fig.add_subplot(gs[1, 2]), ],

          [fig.add_subplot(gs[2, 0]),
           fig.add_subplot(gs[2, 1]),
           fig.add_subplot(gs[2, 2]), ]]

    _found_showers_predicted_sum = np.array(_found_showers_predicted_sum)
    _found_showers_truth_sum = np.array(_found_showers_truth_sum)
    _found_showers_target_energies = np.array(_found_showers_target_energies)
    _found_showers_predicted_energies = np.array(_found_showers_predicted_energies)

    for i in range(9):
        c = int(i / 3)
        r = i % 3
        n = names[i]
        l = energy_segments[i]
        h = energy_segments[i + 1]

        condition = np.logical_and(_found_showers_truth_sum > l, _found_showers_truth_sum < h)

        found_showers_predicted_sum = _found_showers_predicted_sum[condition]
        found_showers_truth_sum = _found_showers_truth_sum[condition]
        found_showers_target_energies = _found_showers_target_energies[condition]
        found_showers_predicted_energies = _found_showers_predicted_energies[condition]

        response_rechit_sum_energy = found_showers_predicted_sum / found_showers_truth_sum
        response_rechit_sum_energy[response_rechit_sum_energy > 3] = 3

        response_energy_predicted = found_showers_predicted_energies / found_showers_target_energies
        response_energy_predicted[response_energy_predicted > 3] = 3
        response_energy_predicted[response_energy_predicted < 0.1] = 0.1

        # spread_sum = np.sqrt(np.var(response_rechit_sum_energy)/np.alen(response_rechit_sum_energy)).round(5)
        # spread_pred = np.sqrt(np.var(response_energy_predicted)/np.alen(response_energy_predicted)).round(5)
        spread_sum = np.std(response_rechit_sum_energy).round(3)
        spread_pred = np.std(response_energy_predicted).round(3)
        mean_sum = np.mean(response_rechit_sum_energy).round(3)
        mean_pred = np.mean(response_energy_predicted).round(3)
        error_sum = np.sqrt(np.var(response_rechit_sum_energy) / np.alen(response_rechit_sum_energy)).round(5)
        error_pred = np.sqrt(np.var(response_energy_predicted) / np.alen(response_energy_predicted)).round(5)

        data_dict = {}
        # plt.figure()
        ax[c][r].hist(response_rechit_sum_energy, bins=20, histtype='step')
        ax[c][r].hist(response_energy_predicted, bins=20, histtype='step')
        ax[c][r].legend(['predicted shower sum / truth shower sum\nmean ' + str(mean_sum) + ' ± ' + str(
            error_sum) + ' spread ' + str(spread_sum),
                         'predicted energy / target energy\nmean ' + str(mean_pred) + ' ± ' + str(
                             error_pred) + ' spread ' + str(spread_pred)])
        ax[c][r].set_xlabel("Predicted/truth")
        ax[c][r].set_ylabel("Frequency")
        ax[c][r].set_title(n)
        # ax[c][r].text(n)


def make_truth_predicted_rotational_distance_histogram(plt, ax, rotational_distance_data):
    rotational_distance_data = np.array(rotational_distance_data)
    rotational_distance_data[rotational_distance_data > 0.2] = 0.2

    plt.figure()
    plt.hist(rotational_distance_data, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')


def make_truth_predicted_rotational_distance_histogram(plt, ax, eta_predicted, eta_truth, phi_predicted, phi_truth):
    eta_predicted = np.array(eta_predicted)
    eta_truth = np.array(eta_truth)
    phi_predicted = np.array(phi_predicted)
    phi_truth = np.array(phi_truth)

    rotational_distance_data = np.sqrt((eta_predicted - eta_truth) ** 2 + (phi_predicted - phi_truth) ** 2)

    rotational_distance_data = np.array(rotational_distance_data)
    rotational_distance_data[rotational_distance_data > 0.2] = 0.2

    plt.figure()
    plt.hist(rotational_distance_data, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')


def make_found_showers_plot_as_function_of_energy(plt, ax, energies, found_or_not, ticl=False):
    e_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_energies = np.argwhere(np.logical_and(energies > l, energies < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l + 5)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Shower energy')
    plt.ylabel('% found')
    plt.title('Function of energy' + (' - ticl' if ticl else ''))


def make_energy_response_curve_as_a_function_of_truth_energy(plt, ax, energies, predicted_energies, ticl=False):
    e_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    predicted_energies = np.array(predicted_energies)

    energies = energies[predicted_energies != -1]
    predicted_energies = predicted_energies[predicted_energies != -1]

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_energies_indices = np.argwhere(np.logical_and(energies > l, energies < h))

        this_energies = energies[this_energies_indices]
        this_predicted = predicted_energies[this_energies_indices].astype(np.float)
        response = this_predicted / this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append(l + 5)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Shower energy')
    plt.ylabel('Response ($e_{predicted}/e_{true}$)')
    plt.title('Function of energy' + (' - ticl' if ticl else ''))


def make_energy_response_curve_as_a_function_of_local_energy_density(plt, ax, local_energy_densities, truth_energies,
                                                                     predicted_energies, ticl=False):
    e_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    centers = []
    mean = []
    std = []

    local_energy_densities = np.array(local_energy_densities)
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_energies)

    truth_energies = truth_energies[predicted_energies != -1]
    local_energy_densities = local_energy_densities[predicted_energies != -1]
    predicted_energies = predicted_energies[predicted_energies != -1]

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_density_indices = np.argwhere(np.logical_and(local_energy_densities > l, local_energy_densities < h))

        this_energies = truth_energies[this_density_indices]
        this_predicted = predicted_energies[this_density_indices].astype(np.float)
        response = this_predicted / this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append((l + h) / 2)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Local energy densities')
    plt.ylabel('Response')
    plt.title('Function of local energy density' + (' - ticl' if ticl else ''))


def make_energy_response_curve_as_a_function_of_closest_particle_distance(plt, ax, closest_particle_distance,
                                                                          truth_energies, predicted_energies,
                                                                          ticl=False):
    e_bins = [0., 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 10]

    centers = []
    mean = []
    std = []

    closest_particle_distance = np.array(closest_particle_distance)
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_energies)

    truth_energies = truth_energies[predicted_energies != -1]
    closest_particle_distance = closest_particle_distance[predicted_energies != -1]
    predicted_energies = predicted_energies[predicted_energies != -1]

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_density_indices = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

        this_energies = truth_energies[this_density_indices]
        this_predicted = predicted_energies[this_density_indices].astype(np.float)
        response = this_predicted / this_energies

        m = np.mean(response)
        mean.append(m)
        std.append(np.std(response))
        centers.append(l + 0.3)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Closest particle distance ($\Delta R$)')
    plt.ylabel('Response')
    plt.title('Function of closest particle distance' + (' - ticl' if ticl else ''))


def make_found_showers_plot_as_function_of_closest_particle_distance(plt, ax, cloest_particle_distance, found_or_not,
                                                                     ticl=False):
    e_bins = [0., 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 10]

    centers = []
    mean = []
    std = []

    cloest_particle_distance = np.array(cloest_particle_distance)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_energies = np.argwhere(np.logical_and(cloest_particle_distance > l, cloest_particle_distance < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l + 0.03)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Closest particle distance (eta, phi, sqrt)')
    plt.ylabel('% found')
    plt.title('Function of closest particle distance' + (' - ticl' if ticl else ''))


def make_found_showers_plot_as_function_of_local_density(plt, ax, local_densities, found_or_not, ticl=False):
    e_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    centers = []
    mean = []
    std = []

    local_densities = np.array(local_densities)
    found_or_not = np.array(found_or_not)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        this_energies = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append((l + h) / 2)

    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Local particle density')
    plt.ylabel('% found')
    plt.title('Function of local particle density' + (' - ticl' if ticl else ''))


def efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, ax, _local_densities, _found_or_not,
                                                                   _found_or_not_ticl, _truth_energies, energy_filter=0,
                                                                   make_segments=False):
    segments_low = [0, 5, 10, 30]
    segments_high = [5, 10, 30, 300]

    count_segments = 4 if make_segments else 1

    use_ticl = _found_or_not_ticl is not None

    if make_segments:
        ax_array = [0, 0, 0, 0]
        fig, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2, 2, figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)


    else:
        fig, ax_array = plt.subplots(1, 1, figsize=(8, 6))

    if energy_filter > 0:
        _local_densities = _local_densities[_truth_energies > energy_filter]
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        if use_ticl:
            _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        local_densities = _local_densities
        found_or_not = _found_or_not
        if use_ticl:
            found_or_not_ticl = _found_or_not_ticl
        truth_energies = _truth_energies

    if energy_filter != 0:
        assert make_segments == False

    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax_array

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number],
                                    _truth_energies < segments_high[segment_number])
            local_densities = _local_densities[filter]
            found_or_not = _found_or_not[filter]
            if use_ticl:
                found_or_not_ticl = _found_or_not_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
        # e_bins = [0, .01, .02, .03, .04, .06, .08, .10, .15, .20, .30, .40, .50, .60, .70, .80, .90, 1]
        e_bins = [0, .01, .02, .03, .04, .05, .06,0.07, .08,.09, .10, .11,.12,.13,.14,.15,.16,.18, .20,.25, .30, .40, .50, .60, .70, .80, .90, 1]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []
        std = []

        local_densities = np.array(local_densities)
        found_or_not = np.array(found_or_not)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))
            filtered_found = found_or_not[filter].astype(np.float)
            if use_ticl:
                filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)

            m = np.mean(filtered_found)
            mean.append(m)

            if use_ticl:
                mt = np.mean(filtered_found_ticl)
                mean_ticl.append(mt)

            std.append(np.std(filtered_found))
            centers.append((l + h) / 2)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        #
        # print(len(mean), len(centers))
        #
        # 0/0

        ax2 = ax1.twinx()
        hist_values, _ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Fraction of number of showers')
        ax2.set_ylim(0, np.max(hist_values) * 1.3)
        ax1.set_title('Efficiency comparison ' + sx)

        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        if use_ticl:
            ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax1.set_ylabel('Reconstruction efficiency')
        ax1.legend(loc='center right')
        ax1.set_ylim(0, 1.04)





def compute_fake_rate_plot_data(predicted_energies, matched_energies):

    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    centers = []
    mean = []

    lows = []
    highs = []

    fake_energies = predicted_energies[matched_energies == -1]
    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0

        mean.append(m)
        lows.append(l)
        highs.append(h)


    hist_values, _ = np.histogram(predicted_energies, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    print(hist_values, mean)

    return np.array(lows), np.array(highs), np.array(hist_values), np.array(mean)

def compute_response_curve_data(energy_truth, energy_predicted):
    mean = []
    var = []

    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())


    energy_truth = np.array(energy_truth)
    energy_predicted = np.array(energy_predicted)

    fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4)


    lows = []
    highs = []

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy = energy_predicted[filter].astype(np.float)

        m = np.mean(filtered_predicted_energy / filtered_truth_energy)
        mean.append(m)
        var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)

        lows.append(l)
        highs.append(h)


    hist_values, _ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()


    return np.array(lows), np.array(highs), np.array(hist_values), np.array(mean), np.array(var)








def fake_rate_plot_beta(plt, ax, predicted_energies, matched_energies):
    lows, highs, hist_values, mean = compute_fake_rate_plot_data(predicted_energies, matched_energies)
    hist_values = hist_values.tolist()
    mean = mean.tolist()

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    ax2 = ax1.twinx()

    e_bins = np.concatenate(([lows[0]], highs), axis=0)

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Fraction of number of showers')
    ax2.set_ylim(0, np.max(hist_values) * 1.3)
    ax1.set_title('Fake rate comparison ')

    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.set_xlabel('Predicted energy [GeV]')
    ax1.set_ylabel('Fake rate')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, 1.04)


def response_curve_beta(plt, ax, energy_truth, energy_predicted):
    lows, highs, hist_values, mean, var = compute_response_curve_data(energy_truth, energy_predicted)
    hist_values = hist_values.tolist()
    mean = mean.tolist()
    var = var.tolist()



    fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4)

    ax2 = ax1.twinx()

    e_bins = np.concatenate(([lows[0]], highs), axis=0)

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Fraction of number of showers')
    ax2.set_ylim(0, np.max(hist_values) * 1.3)

    ax1.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.set_xlabel('Truth energy [GeV]')
    ax1.set_ylabel('$<e_{pred} / e_{true}>$')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, ax1.get_ylim()[1])

    ax22_twin = ax22.twinx()

    ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)
    ax22_twin.set_ylim(0, np.max(hist_values) * 1.3)

    ax22_twin.set_ylabel('Fraction of number of showers')

    ax22.step(e_bins, [var[0]] + var, label='Object condensation')
    ax22.set_xlabel('Truth energy [GeV]')
    ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
    ax22.legend(loc='center right')



def efficiency_comparison_plot_with_distribution_fo_truth_energy(plt, ax, _found_or_not, _found_or_not_ticl,
                                                                 _truth_energies, energy_filter=0):
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

    use_ticl = _found_or_not_ticl is not None

    if energy_filter > 0:
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    found_or_not = _found_or_not
    if use_ticl:
        found_or_not_ticl = _found_or_not_ticl
    truth_energies = _truth_energies

    # e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    centers = []
    mean = []
    mean_ticl = []
    std = []

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        filter = np.argwhere(np.logical_and(truth_energies > l, truth_energies < h))
        filtered_found = found_or_not[filter].astype(np.float)
        if use_ticl:
            filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)

        m = np.mean(filtered_found)
        mean.append(m)
        # print(np.sum(filtered_found), len(filtered_found), m, l, h)

        if use_ticl:
            mt = np.mean(filtered_found_ticl)
            mean_ticl.append(mt)
        std.append(np.std(filtered_found))
        centers.append((l + h) / 2)

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter  #
    # print(len(mean), len(centers))
    #
    # 0/0

    ax2 = ax1.twinx()
    hist_values, _ = np.histogram(truth_energies, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Fraction of number of showers')
    ax2.set_ylim(0, np.max(hist_values) * 1.3)
    ax1.set_title('Efficiency comparison ' + sx)

    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    if use_ticl:
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Truth energy [GeV]')
    ax1.set_ylabel('Reconstruction efficiency')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, 1.04)


def efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, ax, _closest_particle_distance,
                                                                              _found_or_not, _found_or_not_ticl,
                                                                              _truth_energies, energy_filter=0,
                                                                              make_segments=False):
    segments_low = [0, 5, 10, 30]
    segments_high = [5, 10, 30, 300]

    count_segments = 4 if make_segments else 1

    use_ticl = _found_or_not_ticl is not None

    if make_segments:
        ax_array = [0, 0, 0, 0]
        fig, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2, 2, figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
    else:
        fig, ax_array = plt.subplots(1, 1, figsize=(8, 6))

    if energy_filter > 0:
        _closest_particle_distance = _closest_particle_distance[_truth_energies > energy_filter]
        _found_or_not = _found_or_not[_truth_energies > energy_filter]
        if use_ticl:
            _found_or_not_ticl = _found_or_not_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        closest_particle_distance = _closest_particle_distance
        found_or_not = _found_or_not
        if use_ticl:
            found_or_not_ticl = _found_or_not_ticl
        truth_energies = _truth_energies

    if energy_filter != 0:
        assert make_segments == False

    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax_array

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number],
                                    _truth_energies < segments_high[segment_number])
            closest_particle_distance = _closest_particle_distance[filter]
            found_or_not = _found_or_not[filter]
            if use_ticl:
                found_or_not_ticl = _found_or_not_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.07, 0.14, 0.21, 0.28, 0.35, .42, .49, .56, .63, .7]
        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.055, 0.07, 0.09, 0.11, 0.13, 0.14, 0.175, 0.21, 0.28, 0.35, .42, .49,
                  .56,
                  .63, .7]

        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,0.08, 0.09,0.1, 0.11,0.12, 0.13, 0.14, 0.175, 0.21, 0.28, 0.35, .42, .49,
                  .56,
                  .63, .7]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []
        std = []

        closest_particle_distance = np.array(closest_particle_distance)
        found_or_not = np.array(found_or_not)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))
            filtered_found = found_or_not[filter].astype(np.float)
            if use_ticl:
                filtered_found_ticl = found_or_not_ticl[filter].astype(np.float)

            m = np.mean(filtered_found)
            mean.append(m)
            if use_ticl:
                mt = np.mean(filtered_found_ticl)
                mean_ticl.append(mt)
            std.append(np.std(filtered_found))
            centers.append((l + h) / 2)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter  #
        # print(len(mean), len(centers))
        #
        # 0/0

        ax2 = ax1.twinx()
        hist_values, _ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Fraction of number of showers')
        ax2.set_ylim(0, np.max(hist_values) * 1.3)
        ax1.set_title('Efficiency comparison ' + sx)

        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        if use_ticl:
            ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Closest particle distance ($\Delta R$)')
        ax1.set_ylabel('Reconstruction efficiency')
        ax1.legend(loc='center right')
        ax1.set_ylim(0, 1.04)


def response_comparison_plot_with_distribution_fo_local_fraction(plt, ax, _local_densities, _energy_predicted,
                                                                 _energy_predicted_ticl, _truth_energies,
                                                                 energy_filter=0, make_segments=False):
    segments_low = [0, 5, 10, 30]
    segments_high = [5, 10, 30, 300]

    count_segments = 4 if make_segments else 1

    use_ticl = _energy_predicted_ticl is not None

    if make_segments:
        ax_array = [0, 0, 0, 0]
        ax_array_res = [0, 0, 0, 0]
        fig1, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2, 2, figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)

        fig2, ((ax_array_res[0], ax_array_res[1]), (ax_array_res[2], ax_array_res[3])) = plt.subplots(2, 2,
                                                                                                      figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)


    else:
        fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(16, 6))
        plt.subplots_adjust(wspace=0.4)

    if energy_filter > 0:
        _local_densities = _local_densities[_truth_energies > energy_filter]
        _energy_predicted = _energy_predicted[_truth_energies > energy_filter]
        if use_ticl:
            _energy_predicted_ticl = _energy_predicted_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        local_densities = _local_densities
        energy_predicted = _energy_predicted
        if use_ticl:
            energy_predicted_ticl = _energy_predicted_ticl
        truth_energies = _truth_energies

    if energy_filter != 0:
        assert make_segments == False

    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax1
        ax22 = ax_array_res[segment_number] if make_segments else ax22

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number],
                                    _truth_energies < segments_high[segment_number])
            local_densities = _local_densities[filter]
            energy_predicted = _energy_predicted[filter]
            if use_ticl:
                energy_predicted_ticl = _energy_predicted_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
        e_bins = [0, .01, .02, .03, .04, .06, .08, .10, .15, .20, .30, .40, .50, .60, .70, .80, .90, 1]
        e_bins = [0, .01, .02, .03, .04, .05, .06,0.07, .08,.09, .10, .11,.12,.13,.14,.15,.16,.18, .20,.25, .30, .40, .50, .60, .70, .80, .90, 1]

        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []
        var = []
        var_ticl = []

        local_densities = np.array(local_densities)
        energy_predicted = np.array(energy_predicted)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy = energy_predicted[filter].astype(np.float)

            second_filter = filtered_predicted_energy >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy = filtered_predicted_energy[second_filter]

            m = np.mean(filtered_predicted_energy / filtered_truth_energy)
            mean.append(m)

            var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)

            centers.append((l + h) / 2)
        if use_ticl:
            for i in range(len(e_bins) - 1):
                l = e_bins[i]
                h = e_bins[i + 1]

                filter = np.argwhere(np.logical_and(local_densities > l, local_densities < h))

                filtered_truth_energy = truth_energies[filter].astype(np.float)
                filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)

                second_filter = filtered_predicted_energy_ticl >= 0
                filtered_truth_energy = filtered_truth_energy[second_filter]
                filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]

                mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
                mean_ticl.append(mt)

                var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - m) / m)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax1.set_title('Response comparison ' + sx)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax22.set_title('Response comparison ' + sx)
        #
        # print(len(mean), len(centers))
        #
        # 0/0

        ax2 = ax1.twinx()
        hist_values, _ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Fraction of number of showers')
        ax2.set_ylim(0, np.max(hist_values) * 1.3)

        ax1.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')

        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        if use_ticl:
            ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax1.set_ylabel('$<e_{pred} / e_{true}>$')
        ax1.legend(loc='center right')

        ax1.set_ylim(0, ax1.get_ylim()[1])

        ax22_twin = ax22.twinx()
        hist_values, _ = np.histogram(local_densities, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax22_twin.set_ylabel('Fraction of number of showers')

        ax22_twin.set_ylim(0, np.max(hist_values) * 1.3)

        ax22.step(e_bins, [var[0]] + var, label='Object condensation')
        if use_ticl:
            ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
        ax22.set_xticks(e_bins_ticks)
        ax22.set_xlabel('Local shower energy fraction ($\\frac{e_s}{\\sum_{i}^{} e_i \mid \Delta R(s, i) < 0.5 }$)')
        ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
        ax22.legend(loc='center right')

    if make_segments:
        return fig1, fig2


def response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, ax, _closest_particle_distance,
                                                                            _energy_predicted, _energy_predicted_ticl,
                                                                            _truth_energies, energy_filter=0,
                                                                            make_segments=False):
    segments_low = [0, 5, 10, 30]
    segments_high = [5, 10, 30, 300]

    count_segments = 4 if make_segments else 1

    use_ticl = _energy_predicted_ticl is not None

    if make_segments:
        ax_array = [0, 0, 0, 0]
        ax_array_res = [0, 0, 0, 0]
        fig1, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2, 2, figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)

        fig2, ((ax_array_res[0], ax_array_res[1]), (ax_array_res[2], ax_array_res[3])) = plt.subplots(2, 2,
                                                                                                      figsize=(16, 10))

        plt.subplots_adjust(hspace=0.3, wspace=0.4)
    else:
        fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(16, 6))
        plt.subplots_adjust(wspace=0.4)

    if energy_filter > 0:
        _closest_particle_distance = _closest_particle_distance[_truth_energies > energy_filter]
        _energy_predicted = _energy_predicted[_truth_energies > energy_filter]
        if use_ticl:
            _energy_predicted_ticl = _energy_predicted_ticl[_truth_energies > energy_filter]
        _truth_energies = _truth_energies[_truth_energies > energy_filter]

    if not make_segments:
        closest_particle_distance = _closest_particle_distance
        energy_predicted = _energy_predicted
        if use_ticl:
            energy_predicted_ticl = _energy_predicted_ticl
        truth_energies = _truth_energies

    if energy_filter != 0:
        assert make_segments == False

    for segment_number in range(count_segments):
        ax1 = ax_array[segment_number] if make_segments else ax1
        ax22 = ax_array_res[segment_number] if make_segments else ax22

        if make_segments:
            filter = np.logical_and(_truth_energies > segments_low[segment_number],
                                    _truth_energies < segments_high[segment_number])
            closest_particle_distance = _closest_particle_distance[filter]
            energy_predicted = _energy_predicted[filter]
            if use_ticl:
                energy_predicted_ticl = _energy_predicted_ticl[filter]
            truth_energies = _truth_energies[filter]

        e_bins_ticks = [0, 0.07, 0.14, 0.21, 0.28, 0.35, .42, .49, .56, .63, .7]
        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.055, 0.07, 0.1, 0.13, 0.21, 0.28, 0.35, 0.49, .7]

        e_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,0.08, 0.09,0.1, 0.11,0.12, 0.13, 0.14, 0.175, 0.21, 0.28, 0.35, .42, .49,
                  .56,
                  .63, .7]
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        mean_ticl = []

        var = []
        var_ticl = []

        closest_particle_distance = np.array(closest_particle_distance)
        energy_predicted = np.array(energy_predicted)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

            filtered_truth_energy = truth_energies[filter].astype(np.float)
            filtered_predicted_energy = energy_predicted[filter].astype(np.float)

            second_filter = filtered_predicted_energy >= 0
            filtered_truth_energy = filtered_truth_energy[second_filter]
            filtered_predicted_energy = filtered_predicted_energy[second_filter]

            m = np.mean(filtered_predicted_energy / filtered_truth_energy)
            mean.append(m)
            var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)

        if use_ticl:
            for i in range(len(e_bins) - 1):
                l = e_bins[i]
                h = e_bins[i + 1]

                filter = np.argwhere(np.logical_and(closest_particle_distance > l, closest_particle_distance < h))

                filtered_truth_energy = truth_energies[filter].astype(np.float)
                filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)

                second_filter = filtered_predicted_energy_ticl >= 0
                filtered_truth_energy = filtered_truth_energy[second_filter]
                filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]

                mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
                mean_ticl.append(mt)
                var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - mt) / mt)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax1.set_title('Response comparison ' + sx)

        if make_segments:
            sx = '(%.2f GeV - %.2f GeV)' % (segments_low[segment_number], segments_high[segment_number])
        else:
            sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
        ax22.set_title('Response comparison ' + sx)
        #
        # print(len(mean), len(centers))
        #
        # 0/0

        ax2 = ax1.twinx()
        hist_values, _ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax2.set_ylabel('Fraction of number of showers')

        ax2.set_ylim(0, np.max(hist_values) * 1.3)

        ax1.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
        if use_ticl:
            ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
        ax1.set_xticks(e_bins_ticks)
        ax1.set_xlabel('Closest particle distance ($\Delta R$)')
        ax1.set_ylabel('$<e_{pred} / e_{true}>$')
        ax1.legend(loc='center right')
        ax1.set_ylim(0, ax1.get_ylim()[1])

        ax22_twin = ax22.twinx()
        hist_values, _ = np.histogram(closest_particle_distance, bins=e_bins)
        hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values = (hist_values / np.sum(hist_values)).tolist()

        ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
        ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

        ax22_twin.set_ylabel('Fraction of number of showers')

        ax22_twin.set_ylim(0, np.max(hist_values) * 1.3)

        ax22.step(e_bins, [var[0]] + var, label='Object condensation')
        if use_ticl:
            ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
        ax22.set_xticks(e_bins_ticks)
        ax22.set_xlabel('Closest particle distance ($\Delta R$)')
        ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
        ax22.legend(loc='center right')

    if make_segments:
        return fig1, fig2





def response_curve_comparison_with_distribution_fo_energy(plt, ax, energy_truth, energy_predicted,
                                                          energy_predicted_ticl, energy_filter=0):

    use_ticl = energy_predicted_ticl is not None
    # e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    if energy_filter > 0:
        energy_predicted = energy_predicted[energy_truth > energy_filter]
        if use_ticl:
            energy_predicted_ticl = energy_predicted_ticl[energy_truth > energy_filter]
        energy_truth = energy_truth[energy_truth > energy_filter]

    centers = []
    mean = []
    mean_ticl = []
    var = []
    var_ticl = []

    energy_truth = np.array(energy_truth)
    energy_predicted = np.array(energy_predicted)

    fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy = energy_predicted[filter].astype(np.float)

        m = np.mean(filtered_predicted_energy / filtered_truth_energy)
        mean.append(m)
        var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)

    if use_ticl:
        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

            filtered_truth_energy = energy_truth[filter].astype(np.float)
            filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)

            mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
            mean_ticl.append(mt)

            var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - mt) / mt)

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax1.set_title('Response comparison ' + sx)

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax22.set_title('Response comparison ' + sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0

    ax2 = ax1.twinx()
    hist_values, _ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Fraction of number of showers')
    ax2.set_ylim(0, np.max(hist_values) * 1.3)

    ax1.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    if use_ticl:
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Truth energy [GeV]')
    ax1.set_ylabel('$<e_{pred} / e_{true}>$')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, ax1.get_ylim()[1])

    ax22_twin = ax22.twinx()
    hist_values, _ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)
    ax22_twin.set_ylim(0, np.max(hist_values) * 1.3)

    ax22_twin.set_ylabel('Fraction of number of showers')

    ax22.step(e_bins, [var[0]] + var, label='Object condensation')
    if use_ticl:
        ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
    ax22.set_xticks(e_bins_ticks)
    ax22.set_xlabel('Truth energy [GeV]')
    ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
    ax22.legend(loc='center right')


def response_curve_comparison_with_distribution_fo_energy_with_multiple_matches(plt, ax, pred_sample_id, pred_shower_sid,
                            pred_shower_energy, truth_sample_id, truth_shower_sid, truth_shower_energy,
                            ticl_sample_id, ticl_shower_sid, ticl_shower_energy, energy_filter=0):
    # e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    energy_truth = []
    energy_predicted=[]
    energy_predicted_ticl = []


    # _num_showers_per_segment = []
    max_endcap_id = int(np.max(truth_sample_id))
    for i in range(max_endcap_id):
        _pred_shower_sid = pred_shower_sid[pred_sample_id==i]
        _pred_shower_energy = pred_shower_energy[pred_sample_id==i]

        _ticl_shower_sid = ticl_shower_sid[ticl_sample_id==i]
        _ticl_shower_energy = ticl_shower_energy[ticl_sample_id==i]

        _truth_shower_sid = truth_shower_sid[truth_sample_id==i]
        _truth_shower_energy = truth_shower_energy[truth_sample_id==i]


        truth_shower_sid_unique = np.unique(_truth_shower_sid)

        for idx, sid in enumerate(truth_shower_sid_unique):
            this_truth_energy = _truth_shower_energy[idx]
            energy_truth.append(this_truth_energy)

            this_pred_energy = _pred_shower_energy[_pred_shower_sid==sid]
            if len(this_pred_energy) ==0:
                energy_predicted.append(-1)
            else:
                energy_predicted.append(np.sum(this_pred_energy))

            this_ticl_energy = _ticl_shower_energy[_ticl_shower_sid==sid]
            if len(this_ticl_energy) ==0:
                energy_predicted_ticl.append(-1)
            else:
                energy_predicted_ticl.append(np.sum(this_ticl_energy))

    energy_truth = np.array(energy_truth)
    energy_predicted = np.array(energy_predicted)
    energy_predicted_ticl = np.array(energy_predicted_ticl)

    if energy_filter > 0:
        energy_predicted = energy_predicted[energy_truth > energy_filter]
        energy_predicted_ticl = energy_predicted_ticl[energy_truth > energy_filter]
        energy_truth = energy_truth[energy_truth > energy_filter]

    centers = []
    mean = []
    mean_ticl = []
    var = []
    var_ticl = []

    energy_truth = np.array(energy_truth)
    energy_predicted = np.array(energy_predicted)

    fig, (ax1, ax22) = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy = energy_predicted[filter].astype(np.float)

        second_filter = filtered_predicted_energy >= 0
        filtered_truth_energy = filtered_truth_energy[second_filter]
        filtered_predicted_energy = filtered_predicted_energy[second_filter]

        m = np.mean(filtered_predicted_energy / filtered_truth_energy)
        mean.append(m)
        var.append(np.std(filtered_predicted_energy / filtered_truth_energy - m) / m)

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))

        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_predicted_energy_ticl = energy_predicted_ticl[filter].astype(np.float)

        second_filter = filtered_predicted_energy_ticl >= 0
        filtered_truth_energy = filtered_truth_energy[second_filter]
        filtered_predicted_energy_ticl = filtered_predicted_energy_ticl[second_filter]

        mt = np.mean(filtered_predicted_energy_ticl / filtered_truth_energy)
        mean_ticl.append(mt)

        var_ticl.append(np.std(filtered_predicted_energy_ticl / filtered_truth_energy - mt) / mt)

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax1.set_title('Response comparison with multiple matches' + sx)

    sx = '' if energy_filter == 0 else ' - truth energy > %.2f GeV' % energy_filter
    ax22.set_title('Response comparison with multiple matches' + sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0

    ax2 = ax1.twinx()
    hist_values, _ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    ax2.set_ylabel('Fraction of number of showers')
    ax2.set_ylim(0, np.max(hist_values) * 1.3)

    ax1.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Truth energy [GeV]')
    ax1.set_ylabel('$<e_{pred} / e_{true}>$')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, ax1.get_ylim()[1])

    ax22_twin = ax22.twinx()
    hist_values, _ = np.histogram(energy_truth, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    ax22_twin.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax22_twin.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)
    ax22_twin.set_ylim(0, np.max(hist_values) * 1.3)

    ax22_twin.set_ylabel('Fraction of number of showers')

    ax22.step(e_bins, [var[0]] + var, label='Object condensation')
    ax22.step(e_bins, [var_ticl[0]] + var_ticl, label='ticl')
    ax22.set_xticks(e_bins_ticks)
    ax22.set_xlabel('Truth energy [GeV]')
    ax22.set_ylabel('$\\frac{\sigma (e_{pred} / e_{true})}{<e_{pred} / e_{true}>}$')
    ax22.legend(loc='center right')


def fake_rate_comparison_with_distribution_fo_energy(plt, ax, predicted_energies, matched_energies,
                                                     predicted_energies_ticl, matched_energies_ticl, energy_filter=0):
    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    use_ticl = predicted_energies_ticl is not None

    if energy_filter > 0:
        matched_energies = matched_energies[predicted_energies > energy_filter]
        predicted_energies = predicted_energies[predicted_energies > energy_filter]

        if use_ticl:
            matched_energies_ticl = matched_energies_ticl[predicted_energies_ticl > energy_filter]
            predicted_energies_ticl = predicted_energies_ticl[predicted_energies_ticl > energy_filter]

    centers = []
    mean = []
    mean_ticl = []
    std = []

    predicted_energies = np.array(predicted_energies)
    matched_energies = np.array(matched_energies)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # print("OBC")
    # print(len(predicted_energies), len(matched_energies[matched_energies==-1]))
    # print("TICL")
    # print(len(predicted_energies_ticl), len(matched_energies_ticl[matched_energies_ticl==-1]))
    # 0/0

    fake_energies = predicted_energies[matched_energies == -1]
    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0

        mean.append(m)

    if use_ticl:
        fake_energies_ticl = predicted_energies_ticl[matched_energies_ticl == -1]
        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            fake_energies_interval = np.argwhere(np.logical_and(fake_energies_ticl > l, fake_energies_ticl < h))
            total_energies_interval = np.argwhere(np.logical_and(predicted_energies_ticl > l, predicted_energies_ticl < h))

            try:
                mt = len(fake_energies_interval) / float(len(total_energies_interval))
            except ZeroDivisionError:
                mt = 0

            mean_ticl.append(mt)

    # mean_ticl = mean_ticl / (e_bins[1:] - e_bins[:-1])

    sx = '' if energy_filter == 0 else ' - predicted energy > %.2f GeV' % energy_filter
    plt.title('Fake rate comparison' + sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0

    ax2 = ax1.twinx()
    hist_values, _ = np.histogram(predicted_energies, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    if use_ticl:
        hist_values_ticl, _ = np.histogram(predicted_energies_ticl, bins=e_bins)
        hist_values_ticl = (hist_values_ticl / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values_ticl = (hist_values_ticl / np.sum(hist_values_ticl)).tolist()

    if use_ticl:
        ax2.set_ylim(0, max(np.max(hist_values_ticl), np.max(hist_values)) * 1.3)
    else:
        ax2.set_ylim(0, np.max(hist_values) * 1.3)
    #
    # hist_values[hist_values == 0] = 10
    # hist_values[hist_values_ticl == 0] = 10

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    if use_ticl:
        ax2.step(e_bins, [hist_values_ticl[0]] + hist_values_ticl, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values_ticl[0]] + hist_values_ticl, step="pre", alpha=0.2)

    if not use_ticl:
        legend_elements = [Patch(facecolor='#1f77b4', label='Object condensation', alpha=0.2),
                           Patch(facecolor='#ff7f0e', label='ticl', alpha=0.2)]
    else:
        legend_elements = [Patch(facecolor='#1f77b4', label='Object condensation', alpha=0.2)]

    ax2.set_ylabel('Fraction of number of predicted showers')

    ax2.legend(handles=legend_elements, loc=(0.675, 0.34))

    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    if use_ticl:
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Predicted energy [GeV]')
    ax1.set_ylabel('Fake rate')
    ax1.legend(loc='center right')
    ax1.set_ylim(0, 1.04)


def fake_rate_comparison_with_distribution_fo_energy_with_multiple_matches(plt, ax, pred_sample_id, pred_shower_sid_merged, pred_shower_sid,
                                                                           pred_shower_energy, truth_sample_id, truth_shower_sid, truth_shower_energy,
                                                                           ticl_sample_id, ticl_shower_sid_merged, ticl_shower_sid, ticl_shower_energy):
    e_bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200]
    # e_bins = [0, 1., 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]
    e_bins_n = np.array(e_bins)
    e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

    use_ticl = ticl_shower_energy is not None

    predicted_energies_merged = []
    predicted_energies_ticl_merged=[]
    matched_energies_merged = []
    matched_energies_ticl_merged = []
    # _num_showers_per_segment = []
    max_endcap_id = int(np.max(truth_sample_id))
    for i in range(max_endcap_id):
        _pred_shower_sid = pred_shower_sid_merged[pred_sample_id == i]
        _pred_shower_energy = pred_shower_energy[pred_sample_id==i]

        if use_ticl:
            _ticl_shower_sid = ticl_shower_sid_merged[ticl_sample_id == i]
            _ticl_shower_energy = ticl_shower_energy[ticl_sample_id==i]

        _truth_shower_sid = truth_shower_sid[truth_sample_id==i]
        _truth_shower_energy = truth_shower_energy[truth_sample_id==i]

        pred_shower_sid_unique = np.unique(_pred_shower_sid)
        for sid in pred_shower_sid_unique:
            this_pred_energy = np.sum(_pred_shower_energy[_pred_shower_sid==sid])
            predicted_energies_merged.append(this_pred_energy)
            truth_shower_energy_matched = _truth_shower_energy[_truth_shower_sid==sid]
            if len(truth_shower_energy_matched) > 0:
                matched_energies_merged.append(np.sum(truth_shower_energy_matched))
            else:
                matched_energies_merged.append(-1)

        if use_ticl:
            ticl_shower_sid_unique = np.unique(_ticl_shower_sid)
            for sid in ticl_shower_sid_unique:
                this_ticl_energy = np.sum(_ticl_shower_energy[_ticl_shower_sid==sid])
                predicted_energies_ticl_merged.append(this_ticl_energy)
                truth_shower_energy_matched = _truth_shower_energy[_truth_shower_sid==sid]
                if len(truth_shower_energy_matched) > 0:
                    matched_energies_ticl_merged.append(np.sum(truth_shower_energy_matched))
                else:
                    matched_energies_ticl_merged.append(-1)
    predicted_energies_merged = np.array(predicted_energies_merged)
    matched_energies_merged = np.array(matched_energies_merged)

    if use_ticl:
        predicted_energies_ticl_merged=np.array(predicted_energies_ticl_merged)
        matched_energies_ticl_merged = np.array(matched_energies_ticl_merged)

    predicted_energies = []
    predicted_energies_ticl = []
    matched_energies = []
    matched_energies_ticl = []
    # _num_showers_per_segment = []
    max_endcap_id = int(np.max(truth_sample_id))
    for i in range(max_endcap_id):
        _pred_shower_sid = pred_shower_sid[pred_sample_id == i]
        _pred_shower_energy = pred_shower_energy[pred_sample_id == i]

        if use_ticl:
            _ticl_shower_sid = ticl_shower_sid[ticl_sample_id == i]
            _ticl_shower_energy = ticl_shower_energy[ticl_sample_id == i]

        _truth_shower_sid = truth_shower_sid[truth_sample_id == i]
        _truth_shower_energy = truth_shower_energy[truth_sample_id == i]

        pred_shower_sid_unique = np.unique(_pred_shower_sid)
        for sid in pred_shower_sid_unique:
            this_pred_energy = np.sum(_pred_shower_energy[_pred_shower_sid == sid])
            predicted_energies.append(this_pred_energy)
            truth_shower_energy_matched = _truth_shower_energy[_truth_shower_sid == sid]
            if len(truth_shower_energy_matched) > 0:
                matched_energies.append(np.sum(truth_shower_energy_matched))
            else:
                matched_energies.append(-1)

        if use_ticl:
            ticl_shower_sid_unique = np.unique(_ticl_shower_sid)
            for sid in ticl_shower_sid_unique:
                this_ticl_energy = np.sum(_ticl_shower_energy[_ticl_shower_sid == sid])
                predicted_energies_ticl.append(this_ticl_energy)
                truth_shower_energy_matched = _truth_shower_energy[_truth_shower_sid == sid]
                if len(truth_shower_energy_matched) > 0:
                    matched_energies_ticl.append(np.sum(truth_shower_energy_matched))
                else:
                    matched_energies_ticl.append(-1)
    predicted_energies = np.array(predicted_energies)
    matched_energies = np.array(matched_energies)
    predicted_energies_ticl = np.array(predicted_energies_ticl)
    matched_energies_ticl = np.array(matched_energies_ticl)

    mean = []
    mean_ticl = []

    mean_merged = []
    mean_ticl_merged = []
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # print("OBC")
    # print(len(predicted_energies), len(matched_energies[matched_energies==-1]))
    # print("TICL")
    # print(len(predicted_energies_ticl), len(matched_energies_ticl[matched_energies_ticl==-1]))
    # 0/0

    fake_energies = predicted_energies_merged[matched_energies_merged == -1]
    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies_merged > l, predicted_energies_merged < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0

        mean_merged.append(m)

    if use_ticl:
        fake_energies_ticl = predicted_energies_ticl_merged[matched_energies_ticl_merged == -1]
        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            fake_energies_interval = np.argwhere(np.logical_and(fake_energies_ticl > l, fake_energies_ticl < h))
            total_energies_interval = np.argwhere(np.logical_and(predicted_energies_ticl_merged > l, predicted_energies_ticl_merged < h))

            try:
                mt = len(fake_energies_interval) / float(len(total_energies_interval))
            except ZeroDivisionError:
                mt = 0

            mean_ticl_merged.append(mt)


    fake_energies = predicted_energies[matched_energies == -1]
    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0

        mean.append(m)


    if use_ticl:
        fake_energies_ticl = predicted_energies_ticl[matched_energies_ticl == -1]
        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            fake_energies_interval = np.argwhere(np.logical_and(fake_energies_ticl > l, fake_energies_ticl < h))
            total_energies_interval = np.argwhere(np.logical_and(predicted_energies_ticl > l, predicted_energies_ticl < h))

            try:
                mt = len(fake_energies_interval) / float(len(total_energies_interval))
            except ZeroDivisionError:
                mt = 0

            mean_ticl.append(mt)

    # mean_ticl = mean_ticl / (e_bins[1:] - e_bins[:-1])

    sx = ''
    plt.title('Fake rate comparison' + sx)
    #
    # print(len(mean), len(centers))
    #
    # 0/0

    ax2 = ax1.twinx()
    hist_values, _ = np.histogram(predicted_energies_merged, bins=e_bins)
    hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
    hist_values = (hist_values / np.sum(hist_values)).tolist()

    if use_ticl:
        hist_values_ticl, _ = np.histogram(predicted_energies_ticl_merged, bins=e_bins)
        hist_values_ticl = (hist_values_ticl / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        hist_values_ticl = (hist_values_ticl / np.sum(hist_values_ticl)).tolist()

    ax2.set_ylim(0, np.max(hist_values) * 1.3)

    ax2.step(e_bins, [hist_values[0]] + hist_values, color='tab:gray', alpha=0)
    ax2.fill_between(e_bins, [hist_values[0]] + hist_values, step="pre", alpha=0.2)

    if use_ticl:
        ax2.step(e_bins, [hist_values_ticl[0]] + hist_values_ticl, color='tab:gray', alpha=0)
        ax2.fill_between(e_bins, [hist_values_ticl[0]] + hist_values_ticl, step="pre", alpha=0.2)

    if not use_ticl:
        legend_elements = [Patch(facecolor='#1f77b4', label='Object condensation', alpha=0.2)]
    else:
        legend_elements = [Patch(facecolor='#1f77b4', label='Object condensation', alpha=0.2),
                           Patch(facecolor='#ff7f0e', label='ticl', alpha=0.2)]

    ax2.set_ylabel('Fraction of number of predicted showers')

    ax2.legend(handles=legend_elements, loc=(0.675, 0.34))

    ax1.step(e_bins, [mean[0]] + mean, label='Object condensation')
    if use_ticl:
        ax1.step(e_bins, [mean_ticl[0]] + mean_ticl, label='ticl')

    ax1.step(e_bins, [mean_merged[0]] + mean_merged, label='Object condensation - non-remergable fakes', c='#1f77b4', ls='--')
    if use_ticl:
        ax1.step(e_bins, [mean_ticl_merged[0]] + mean_ticl_merged, label='ticl -  non-remergable fakes', c='#ff7f0e', ls='--')
    ax1.set_xticks(e_bins_ticks)
    ax1.set_xlabel('Predicted energy [GeV]')
    ax1.set_ylabel('Fake rate')
    ax1.legend(loc=(0.39, 0.445))
    ax1.set_ylim(0, 1.04)


def make_fake_rate_plot_as_function_of_fake_energy(plt, ax, predicted_energies, matched_energies, is_sum, ticl=False):
    predicted_energies = np.array(predicted_energies)
    matched_energies = np.array(matched_energies)
    e_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    centers = []
    mean = []
    std = []

    fake_energies = predicted_energies[matched_energies == -1]

    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]

        fake_energies_interval = np.argwhere(np.logical_and(fake_energies > l, fake_energies < h))
        total_energies_interval = np.argwhere(np.logical_and(predicted_energies > l, predicted_energies < h))

        try:
            m = len(fake_energies_interval) / float(len(total_energies_interval))
        except ZeroDivisionError:
            m = 0
        mean.append(m)
        # std.append(np.std(filtered_found))
        centers.append(l + 5)

    # plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.plot(centers, mean, linewidth=0.7, marker='o', ls='--', markersize=3)
    plt.xticks(centers)
    plt.xlabel('Fake energy sum' if is_sum else 'Fake energy regressed')
    plt.ylabel('% fake')
    plt.title('Function of fake energy' + (' - ticl' if ticl else ''))


def make_energy_hists(plt, ax, predicted_energies, matched_energies, truth_shower_energies, truth_showers_found_or_not,
                      is_sum, ticl=False):
    predicted_energies = np.array(predicted_energies)
    truth_energies = np.array(truth_shower_energies)
    fake_energies = predicted_energies[matched_energies == -1]
    missed_energies = truth_shower_energies[truth_showers_found_or_not == 0]

    bins = np.linspace(0, 200, 30)

    plt.hist(predicted_energies, bins=bins, histtype='step', log=True)
    plt.hist(truth_energies, bins=bins, histtype='step', log=True)
    plt.hist(fake_energies, bins=bins, histtype='step', log=True)
    plt.hist(missed_energies, bins=bins, histtype='step', log=True)

    plt.xlabel('Energy (GeV')
    plt.ylabel('Frequency')

    plt.legend(['Predicted', 'Truth', 'Fake', 'Missed'])

    # plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    # plt.xlabel('Fake energy sum' if is_sum else 'Fake energy regressed')
    # plt.ylabel('% fake')
    plt.title('Energy histograms' + (' - ticl' if ticl else ''))


def make_found_showers_plot_as_function_of_pt(plt, ax, energies, eta, found_or_not, ticl=False):
    pt_bins = [0, 10, 30, 70, 100, 150, 250, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    pt_bins = np.linspace(0, 800, 15)
    pt_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800]

    centers = []
    mean = []
    std = []

    energies = np.array(energies)
    eta = np.array(eta)
    found_or_not = np.array(found_or_not)

    pt = np.cosh(eta) * energies

    for i in range(len(pt_bins) - 1):
        l = pt_bins[i]
        h = pt_bins[i + 1]

        this_energies = np.argwhere(np.logical_and(pt > l, pt < h))

        filtered_found = found_or_not[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        std.append(np.std(filtered_found))
        centers.append(l + 5)

    plt.errorbar(centers, mean, std, linewidth=0.7, marker='o', ls='--', markersize=3, capsize=3)
    plt.xticks(centers)
    plt.xlabel('Shower pT')
    plt.ylabel('% found')
    plt.title('Function of pT' + (' - ticl' if ticl else ''))


def make_real_predicted_number_of_showers_histogram(plt, ax, num_real_showers, num_predicted_showers, ticl=False):
    plt.hist(num_real_showers, bins=np.arange(0, 50), histtype='step')
    plt.hist(num_predicted_showers, bins=np.arange(0, 70), histtype='step')
    plt.xlabel('Num showers')
    plt.ylabel('Frequency')
    plt.legend(['Real showers', 'Predicted showers'])
    plt.title('Histogram of predicted/real number of showers' + (' - ticl' if ticl else ''))


def make_histogram_of_number_of_rechits_per_shower(plt, ax, num_rechits_per_shower):
    plt.hist(num_rechits_per_shower, histtype='step')
    plt.xlabel('Num rechits per shower')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per endcap','Num rechits per shower'])
    plt.title('Distribution of number of rechits')


def make_histogram_of_number_of_rechits_per_segment(plt, ax, num_rechits_per_segment):
    plt.hist(num_rechits_per_segment, histtype='step')
    plt.xlabel('Num rechits per calorimeter')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per endcap','Num rechits per shower'])
    plt.title('Distribution of number of rechits')


def make_histogram_of_number_of_showers_per_segment(plt, ax, endcap_sample_ids):
    _num_showers_per_segment = []
    max_endcap_id = int(np.max(endcap_sample_ids))
    for i in range(max_endcap_id):
        _num_showers_per_segment.append(len(endcap_sample_ids[endcap_sample_ids==i]))

    plt.hist(_num_showers_per_segment, histtype='step')
    # plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers per calorimeter')
    plt.ylabel('Frequency')
    # plt.legend(['Real showers','Predicted showers'])
    plt.title('Distribution of number of showers')


def visualize_the_segment(plt, truth_showers_this_segment, truth_and_pred_dict, feature_dict, ticl_showers, labels,
                          coords_representative_predicted_showers, distance_threshold):
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(3, 2)

    ax = [fig.add_subplot(gs[0, 0], projection='3d'),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0], projection='3d'),
          fig.add_subplot(gs[1, 1], projection='3d'),
          fig.add_subplot(gs[2, 0], projection='3d'), ]

    # wrt ground truth colors

    ax[0].set_xlabel('z (cm)')
    ax[0].set_ylabel('y (cm)')
    ax[0].set_zlabel('x (cm)')
    ax[0].set_title('Input data')

    ax[1].set_xlabel('Clustering dimension 1')
    ax[1].set_ylabel('Clustering dimension 2')
    ax[1].set_title('Colors = truth showers')

    ax[2].set_xlabel('z (cm)')
    ax[2].set_ylabel('y (cm)')
    ax[2].set_zlabel('x (cm)')
    ax[2].set_title('Colors = truth showers')

    ax[3].set_xlabel('z (cm)')
    ax[3].set_ylabel('y (cm)')
    ax[3].set_zlabel('x (cm)')
    ax[3].set_title('Colors = predicted showers')

    ax[4].set_xlabel('z (cm)')
    ax[4].set_ylabel('y (cm)')
    ax[4].set_zlabel('x (cm)')
    ax[4].set_title('Colors = ticl showers')

    cmap = createRandomizedColors('jet')

    make_original_truth_shower_plot(plt, ax[0], truth_showers_this_segment * 0, feature_dict['recHitEnergy'][:, 0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:, 0],
                                    feature_dict['recHitZ'][:, 0],
                                    cmap=plt.get_cmap('Wistia'))
    make_cluster_coordinates_plot(plt, ax[1], truth_showers_this_segment, truth_and_pred_dict['predBeta'][:, 0],
                                  truth_and_pred_dict['predCCoords'],
                                  identified_coords=coords_representative_predicted_showers, cmap=cmap,
                                  distance_threshold=distance_threshold)
    #
    # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
    # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))

    # wrt predicted colors

    np.set_printoptions(threshold=np.inf)

    # print(np.array(truth_showers_this_segment))
    # print(np.array(labels))

    xmax = max(np.max(truth_showers_this_segment), np.max(labels))
    rgbcolor_truth = cmap(truth_showers_this_segment / xmax)[:, :-1]
    rgbcolor_labels = cmap(labels / xmax)[:, :-1]
    rgbcolor_ticl = cmap(ticl_showers / xmax)[:, :-1]

    make_original_truth_shower_plot(plt, ax[2], truth_showers_this_segment, feature_dict['recHitEnergy'][:, 0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:, 0],
                                    feature_dict['recHitZ'][:, 0],
                                    cmap=cmap, rgbcolor=rgbcolor_truth)
    make_original_truth_shower_plot(plt, ax[3], labels, feature_dict['recHitEnergy'][:, 0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:, 0],
                                    feature_dict['recHitZ'][:, 0],
                                    cmap=cmap, rgbcolor=rgbcolor_labels)
    make_original_truth_shower_plot(plt, ax[4], ticl_showers, feature_dict['recHitEnergy'][:, 0],
                                    feature_dict['recHitX'][:, 0], feature_dict['recHitY'][:, 0],
                                    feature_dict['recHitZ'][:, 0],
                                    cmap=cmap, rgbcolor=rgbcolor_ticl)

    # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)


# def visualize_the_segment_separate(pdf2, plt, truth_showers_this_segment, truth_and_pred_dict, feature_dict, ticl_showers, labels,
#                           coords_representative_predicted_showers, distance_threshold):
#
#     # gs = plt.GridSpec(3, 2)
#
#     # ax = [fig.add_subplot(gs[0, 0], projection='3d'),
#     #       fig.add_subplot(gs[0, 1]),
#     #       fig.add_subplot(gs[1, 0], projection='3d'),
#     #       fig.add_subplot(gs[1, 1], projection='3d'),
#     #       fig.add_subplot(gs[2, 0], projection='3d'), ]
#     ax = []
#     fig1 = plt.figure()
#     ax.append(fig1.add_subplot(111, projection='3d'))
#     fig2 = plt.figure()
#     ax.append(fig2.gca())
#     fig3 = plt.figure()
#     ax.append(fig3.add_subplot(111, projection='3d'))
#     fig4 = plt.figure()
#     ax.append(fig4.add_subplot(111, projection='3d'))
#     fig5 = plt.figure()
#     ax.append(fig5.add_subplot(111, projection='3d'))
#
#
#     # wrt ground truth colors
#
#     ax[0].set_xlabel('z (cm)')
#     ax[0].set_ylabel('y (cm)')
#     ax[0].set_zlabel('x (cm)')
#     ax[0].set_title('Input data')
#
#     font = {'family': 'sans-serif',
#             'color': 'black',
#             'weight': 'bold',
#             'size': 10,
#             }
#
#     font2 = {'family': 'sans-serif',
#             'color': 'black',
#             'weight': 'normal',
#             'size': 10,
#             }
#
#     fig1.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig1.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig2.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig2.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig3.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig3.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig4.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig4.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#     fig5.text(x=0.1, y=0.9, s='CMS', fontdict=font)
#     fig5.text(x=0.165, y=0.9, s='Phase-2 Simulation Preliminary', fontdict=font2, fontstyle='italic')
#
#
#     ax[1].set_xlabel('Clustering dimension 1')
#     ax[1].set_ylabel('Clustering dimension 2')
#     # ax[1].set_title('Colors = truth showers')
#
#
#
#     ax[2].set_xlabel('z (cm)')
#     ax[2].set_ylabel('y (cm)')
#     ax[2].set_zlabel('x (cm)')
#     ax[2].set_title('Colors = truth showers')
#
#     ax[3].set_xlabel('z (cm)')
#     ax[3].set_ylabel('y (cm)')
#     ax[3].set_zlabel('x (cm)')
#     ax[3].set_title('Colors = predicted showers')
#
#     ax[4].set_xlabel('z (cm)')
#     ax[4].set_ylabel('y (cm)')
#     ax[4].set_zlabel('x (cm)')
#     ax[4].set_title('Colors = ticl showers')
#
#     cmap = createRandomizedColors('jet')
#
#     make_original_truth_shower_plot(plt, ax[0], truth_showers_this_segment * 0, x_this_segment[:, 0],
#                                     x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7],
#                                     cmap=plt.get_cmap('Wistia'))
#     make_cluster_coordinates_plot(plt, ax[1], truth_showers_this_segment, pred_this_segment[:, -6],
#                                   pred_this_segment[:, -2:],
#                                   identified_coords=coords_representative_predicted_showers, cmap=cmap,
#                                   distance_threshold=distance_threshold)
#     #
#     # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
#     # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
#
#     # wrt predicted colors
#
#     np.set_printoptions(threshold=np.inf)
#
#     # print(np.array(truth_showers_this_segment))
#     # print(np.array(labels))
#
#     xmax = max(np.max(truth_showers_this_segment), np.max(labels))
#     rgbcolor_truth = cmap(truth_showers_this_segment/xmax)[:,:-1]
#     rgbcolor_labels = cmap(labels/xmax)[:,:-1]
#     rgbcolor_ticl = cmap(ticl_showers/xmax)[:,:-1]
#
#
#     make_original_truth_shower_plot(plt, ax[2], truth_showers_this_segment, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_truth)
#     make_original_truth_shower_plot(plt, ax[3], labels, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_labels)
#     make_original_truth_shower_plot(plt, ax[4], ticl_showers, x_this_segment[:, 0], x_this_segment[:, 5],
#                                     x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap, rgbcolor=rgbcolor_ticl)
#
#     # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)
#
#     pdf2.savefig(fig1)
#     pdf2.savefig(fig2)
#     pdf2.savefig(fig3)
#     pdf2.savefig(fig4)
#     pdf2.savefig(fig5)


def make_found_showers_plot_as_function_of_number_of_truth_showers(plt, ax, num_real, num_found, num_missed, num_fakes,
                                                                   num_predicted, ticl=False):
    num_real = np.array(num_real)
    num_found = np.array(num_found)
    num_missed = np.array(num_missed)
    num_fakes = np.array(num_fakes)
    num_predicted = np.array(num_predicted)

    x_num_real = []
    mean_fraction_found = []
    mean_fraction_missed = []
    mean_fraction_fakes = []

    var_fraction_found = []
    var_fraction_missed = []
    var_fraction_fakes = []

    fraction_found_by_real = num_found / num_real
    fraction_missed_by_real = num_missed / num_real
    fraction_fake_by_predicted = num_fakes / num_predicted

    for i in np.sort(np.unique(num_real)):
        if i <= 0:
            continue

        x_num_real.append(i)
        mean_fraction_found.append(np.mean(fraction_found_by_real[num_real == i]))
        mean_fraction_missed.append(np.mean(fraction_missed_by_real[num_real == i]))
        mean_fraction_fakes.append(np.mean(fraction_fake_by_predicted[num_real == i]))

        var_fraction_found.append(np.std(fraction_found_by_real[num_real == i]))
        var_fraction_missed.append(np.std(fraction_missed_by_real[num_real == i]))
        var_fraction_fakes.append(np.std(fraction_fake_by_predicted[num_real == i]))

    x_num_real = np.array(x_num_real)
    mean_fraction_found = np.array(mean_fraction_found)
    mean_fraction_missed = np.array(mean_fraction_missed)
    mean_fraction_fakes = np.array(mean_fraction_fakes)

    var_fraction_found = np.array(var_fraction_found)
    var_fraction_missed = np.array(var_fraction_missed)
    var_fraction_fakes = np.array(var_fraction_fakes)

    plt.errorbar(x_num_real, mean_fraction_found, var_fraction_found, linewidth=0.7, marker='o', ls='--', markersize=3,
                 capsize=3)
    plt.errorbar(x_num_real, mean_fraction_fakes, var_fraction_fakes, linewidth=0.7, marker='o', ls='--', markersize=3,
                 capsize=3)
    plt.xlabel('Num showers')
    plt.ylabel('Fraction')
    plt.legend(['Found / Truth', 'Fakes / Predicted'])

    plt.title('Found/Missed' + (' - ticl' if ticl else ''))



def hist_2d_found_efficiency_vs_local_fraction_and_truth_shower_energy(plt, ax, truth_energies, local_fraction,
                                                                       found_or_not, found_or_not_ticl):
    bins_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    bins_e = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    H = np.zeros((len(truth_energies), 2), np.float)

    H[:, 0] = np.digitize(truth_energies, bins_e)
    H[:, 1] = np.digitize(local_fraction, bins_l)

    values_obc = np.zeros((10, 10), np.float)
    values_ticl = np.zeros((10, 10), np.float)
    values_count = np.zeros((10, 10), np.float)

    for i, l in enumerate(bins_l):
        if i == 10:
            continue
        for j, e in enumerate(bins_e):
            if j == 10:
                continue
            filter = np.logical_and(H[:, 0] == j, H[:, 1] == i)
            found_or_not_filtered = found_or_not[filter]
            found_or_not_filtered_ticl = found_or_not_ticl[filter]

            values_count[i, j] = len(found_or_not_filtered)
            values_obc[i, j] = np.mean(found_or_not_filtered) if values_count[i, j] != 0 else 0
            values_ticl[i, j] = np.mean(found_or_not_filtered_ticl) if values_count[i, j] != 0 else 0

    pos = plt.imshow(values_ticl, vmax=1, vmin=0)

    # # Major ticks
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))

    # Labels for major ticks
    ax.set_xticklabels([str(x) for x in np.arange(0., 1., 0.1)])
    ax.set_yticklabels(np.arange(0, 100, 10))
    plt.colorbar(pos)


def draw_text_page(plt, s):
    text_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                 'verticalalignment': 'bottom'}
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.text(0.5, 0.5, s, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             fontdict=text_font)


def draw_settings(plt, b, d, i, soft):
    text_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal',
                 'verticalalignment': 'bottom'}
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_visible(False)
    ax.axis('off')

    s = 'Beta threshold: %.2f\nDist threshold: %.2f\niou  threshold: %.2f\n Is soft: %r' % (b,d,i,soft)

    plt.text(0.5, 0.5, s, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             fontdict=text_font)


def plot_total_response_diff_fractions(plt, truth_sample_id, truth_sid, truth_energies, pred_sample_id, pred_sid,
                                       pred_energies, ticl_sample_id, ticl_sid, ticl_energies):
    num_samples = int(np.max(truth_sample_id))

    # print(len(truth_sample_id), len(truth_sid), len(truth_energies), len(pred_sample_id), len(pred_sid), len(pred_energies), len(ticl_sample_id), len(ticl_sid), len(ticl_energies))

    ax_array = [0, 0, 0, 0]

    fig, ((ax_array[0], ax_array[1]), (ax_array[2], ax_array[3])) = plt.subplots(2, 2, figsize=(16, 10))

    fractions = [0.2, 0.4, 0.6, 0.8]
    show_ticl = ticl_sid is not None

    for f in range(len(fractions)):
        _f = fractions[f]
        response_values_ticl = []
        response_values_pred = []
        for i in range(num_samples):
            _truth_sid = truth_sid[truth_sample_id == i]
            _truth_energies = truth_energies[truth_sample_id == i]
            _pred_sid = pred_sid[pred_sample_id == i]
            _pred_energies = pred_energies[pred_sample_id == i]
            if show_ticl:
                _ticl_sid = ticl_sid[ticl_sample_id == i]
                _ticl_energies = ticl_energies[ticl_sample_id == i]

            remove_sids = [_truth_sid[x] for x in
                           np.random.choice(_truth_sid.shape[0], int(len(_truth_sid) * _f), replace=False)]
            for j in remove_sids:
                _truth_energies[_truth_sid == j] = 0
                _pred_energies[_pred_sid == j] = 0
                if show_ticl:
                    _ticl_energies[_ticl_sid == j] = 0

            response_values_pred.append(np.sum(_pred_energies) / np.sum(_truth_energies))
            if show_ticl:
                response_values_ticl.append(np.sum(_ticl_energies) / np.sum(_truth_energies))

        ax = ax_array[f]
        ax.hist(response_values_pred, bins=20, histtype='step', log=True, label='Object condensation', density=True)
        if show_ticl:
            ax.hist(response_values_ticl, bins=20, histtype='step', log=True, label='ticl', density=True)
        ax.legend()
        ax.set_title('Total energy response - PU fraction %.2f' % _f)


def get_analysis_plotting_configuration(config):
    config_return = dict()

    if config == 'standard_hgcal_with_ticl':
        config_return['use_ticl'] = True
        config_return['use_classification'] = False
    elif config == 'toy_set_without_ticl':
        config_return['use_ticl'] = False
        config_return['use_classification'] = True
    elif config == 'standard_hgcal_without_ticl':
        config_return['use_ticl'] = False
        config_return['use_classification'] = False
    else:
        raise NotImplemented('Can\'t find')

    return config_return

from hplots.general_2d_plot_extensions import EfficiencyFoTruthEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoTruthEnergyPlot
from hplots.general_2d_plot_extensions import ResolutionFoTruthEnergyPlot
from hplots.general_2d_plot_extensions import FakeRateFoPredEnergyPlot
from hplots.general_2d_plot_extensions import ResponseFoIouPlot

def do_analysis_plots_2(dataset_analysis_dict, pdf=None, database_manager=None, tags=None, table_prefix=''):
    dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)

    if pdf is None and database_manager is None:
        raise RuntimeError("No output location selected")

    if tags is None:
        tags = dict()
        tags['beta_threshold'] = dataset_analysis_dict['beta_threshold']
        tags['distance_threshold'] = dataset_analysis_dict['distance_threshold']
        tags['iou_threshold'] = dataset_analysis_dict['iou_threshold']
        tags['soft'] = dataset_analysis_dict['soft']

    efficiency_plot = EfficiencyFoTruthEnergyPlot()
    x_values = dataset_analysis_dict['truth_shower_energy']
    y_values = dataset_analysis_dict['truth_shower_found_or_not']
    efficiency_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)

    if pdf is not None:
        efficiency_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        print("Putting first to database", table_prefix+'plot_eff_fo_truth_energy')
        efficiency_plot.write_to_database(database_manager, table_prefix+'plot_eff_fo_truth_energy')

    response_plot = ResponseFoTruthEnergyPlot()
    filter = dataset_analysis_dict['truth_shower_matched_energy_regressed'] != -1
    x_values = dataset_analysis_dict['truth_shower_energy'][filter]
    y_values = dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter] / \
               dataset_analysis_dict['truth_shower_energy'][filter]
    response_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)

    if pdf is not None:
        response_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        response_plot.write_to_database(database_manager, table_prefix+'plot_response_fo_truth_energy')

    resolution_plot = ResolutionFoTruthEnergyPlot()
    filter = dataset_analysis_dict['truth_shower_matched_energy_regressed'] != -1
    x_values = dataset_analysis_dict['truth_shower_energy'][filter]
    y_values = dataset_analysis_dict['truth_shower_matched_energy_regressed'][filter] / \
               dataset_analysis_dict['truth_shower_energy'][filter]
    resolution_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
    if pdf is not None:
        resolution_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        resolution_plot.write_to_database(database_manager, table_prefix+'plot_resolution_fo_truth_energy')

    response_plot = ResponseFoTruthEnergyPlot(y_label='Response computed from rechit sum')
    filter = dataset_analysis_dict['truth_shower_matched_energy_regressed'] != -1
    x_values = dataset_analysis_dict['truth_shower_energy'][filter]
    y_values = dataset_analysis_dict['truth_shower_matched_energy_sum'][filter] / \
               dataset_analysis_dict['truth_shower_energy'][filter]
    response_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
    if pdf is not None:
        response_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        response_plot.write_to_database(database_manager, table_prefix+'plot_response_from_sum_fo_truth_energy')

    resolution_plot = ResolutionFoTruthEnergyPlot(y_label='Resolution computed from rechit sum')
    filter = dataset_analysis_dict['truth_shower_matched_energy_regressed'] != -1
    x_values = dataset_analysis_dict['truth_shower_energy'][filter]
    y_values = dataset_analysis_dict['truth_shower_matched_energy_sum'][filter] / \
               dataset_analysis_dict['truth_shower_energy'][filter]
    resolution_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
    if pdf is not None:
        resolution_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        resolution_plot.write_to_database(database_manager, table_prefix+'plot_resolution_from_sum_fo_truth_energy')


    fake_rate_plot = FakeRateFoPredEnergyPlot()
    x_values = dataset_analysis_dict['pred_shower_regressed_energy']
    y_values = dataset_analysis_dict['pred_shower_matched_energy'] == -1
    fake_rate_plot.add_raw_values(x_values=x_values, y_values=y_values, tags=tags)
    if pdf is not None:
        fake_rate_plot.draw()
        pdf.savefig()
    if database_manager is not None:
        fake_rate_plot.write_to_database(database_manager, table_prefix+'plot_fake_fo_pred_energy')

    if pdf is not None:
        pdf.close()





def do_analysis_plots_simplified_to_pdf(pdfpath, dataset_analysis_dict):
    dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)

    pdf = PdfPages(pdfpath)

    efficiency_plot = EfficiencyFoTruthEnergyPlot()
    fake_rate_plot = FakeRateFoPredEnergyPlot()
    fake_rate_sum_plot = FakeRateFoPredEnergyPlot()
    response_plot = ResponseFoTruthEnergyPlot()
    response_sum_plot = ResponseFoTruthEnergyPlot()
    tags = dict()
    tags['beta_threshold'] = dataset_analysis_dict['beta_threshold']
    tags['distance_threshold'] = dataset_analysis_dict['distance_threshold']
    tags['iou_threshold'] = dataset_analysis_dict['iou_threshold']
    tags['soft'] = dataset_analysis_dict['soft']

    draw_settings(plt, dataset_analysis_dict['beta_threshold'], dataset_analysis_dict['distance_threshold'], dataset_analysis_dict['iou_threshold'], dataset_analysis_dict['soft'])
    pdf.savefig()

    efficiency_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'],
                                   dataset_analysis_dict['truth_shower_found_or_not'], tags)
    efficiency_plot.draw()
    pdf.savefig()
    # efficiency_plot.write_to_database(database_manager, 'eff_plot_alpha')

    fake_rate_plot.add_raw_values(dataset_analysis_dict['pred_shower_regressed_energy'],
                                  dataset_analysis_dict['pred_shower_matched_energy'], tags)
    # fake_rate_plot.write_to_database(database_manager, 'fake_rate_plot_alpha')
    fake_rate_plot.draw(name_tag_formatter=lambda x: 'predicted energy is regressed energy')
    pdf.savefig()


    fake_rate_sum_plot.add_raw_values(dataset_analysis_dict['pred_shower_energy_sum'],
                                      dataset_analysis_dict['pred_shower_matched_energy'], tags)
    # fake_rate_plot.write_to_database(database_manager, 'fake_rate_plot_alpha')
    fake_rate_sum_plot.draw(name_tag_formatter=lambda x: 'predicted energy is rechit energy sum')
    pdf.savefig()

    response_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'],
                                 dataset_analysis_dict['truth_shower_matched_energy_regressed'], tags)
    # response_plot.write_to_database(database_manager, 'response_plot_alpha')

    response_plot.draw(name_tag_formatter=lambda x: 'predicted energy is regressed energy')
    pdf.savefig()

    response_sum_plot.add_raw_values(dataset_analysis_dict['truth_shower_energy'],
                                     dataset_analysis_dict['truth_shower_matched_energy_sum'], tags)
    # response_plot.write_to_database(database_manager, 'response_plot_alpha')

    response_sum_plot.draw(name_tag_formatter=lambda x: 'predicted energy is rechit energy sum')
    pdf.savefig()

    pdf.close()



def do_analysis_plots_to_pdf(pdfpath, dataset_analysis_dict, plotting_config):
    dataset_analysis_dict = convert_dataset_dict_elements_to_numpy(dataset_analysis_dict)

    pdf = PdfPages(pdfpath)

    draw_settings(plt, dataset_analysis_dict['beta_threshold'], dataset_analysis_dict['distance_threshold'], dataset_analysis_dict['iou_threshold'], dataset_analysis_dict['soft'])
    pdf.savefig()


    use_ticl = plotting_config['use_ticl']
    use_classification = plotting_config['use_classification']


    #################################################################################
    draw_text_page(plt, 'Total energy response')
    pdf.savefig()

    fig = plt.figure()
    histogram_total_endcap_resolution(plt, fig.axes, dataset_analysis_dict['endcap_total_energy_truth'],
                                      dataset_analysis_dict['endcap_total_energy_pred'],
                                      dataset_analysis_dict['endcap_total_energy_ticl'] if use_ticl else None)
    pdf.savefig()

    fig = plt.figure()
    histogram_total_endcap_resolution(plt, fig.axes, dataset_analysis_dict['endcap_total_energy_truth'],
                                      dataset_analysis_dict['endcap_total_energy_pred'],
                                      dataset_analysis_dict['endcap_total_energy_ticl'] if use_ticl else None, energy_filter=2)
    pdf.savefig()
    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Total energy response fractional')
    pdf.savefig()

    # plot_total_response_diff_fractions(plt, dataset_analysis_dict['truth_shower_sample_id'],
    #                                    dataset_analysis_dict['truth_shower_sid'],
    #                                    dataset_analysis_dict['truth_shower_energy'],
    #                                    dataset_analysis_dict['pred_shower_sample_id'],
    #                                    dataset_analysis_dict['pred_shower_sid'],
    #                                    dataset_analysis_dict['pred_shower_regressed_energy'],
    #                                    dataset_analysis_dict['ticl_shower_sample_id'] if use_ticl else None,
    #                                    dataset_analysis_dict['ticl_shower_sid'] if use_ticl else None,
    #                                    dataset_analysis_dict['ticl_shower_regressed_energy'] if use_ticl else None)
    # pdf.savefig()

    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Fake rate comparison')
    pdf.savefig()

    fix, ax = plt.subplots()
    fake_rate_comparison_with_distribution_fo_energy(plt, ax, dataset_analysis_dict['pred_shower_regressed_energy'],
                                                     dataset_analysis_dict['pred_shower_matched_energy'],
                                                     dataset_analysis_dict['ticl_shower_regressed_energy'] if use_ticl else None,
                                                     dataset_analysis_dict['ticl_shower_matched_energy'] if use_ticl else None)
    pdf.savefig()

    # fix, ax = plt.subplots()
    # fake_rate_comparison_with_distribution_fo_energy_with_multiple_matches(plt, ax, dataset_analysis_dict['pred_shower_sample_id'],
    #                                                  dataset_analysis_dict['pred_shower_sid_merged'],
    #                                                  dataset_analysis_dict['pred_shower_sid'],
    #                                                  dataset_analysis_dict['pred_shower_regressed_energy'],
    #                                                  dataset_analysis_dict['truth_shower_sample_id'],
    #                                                  dataset_analysis_dict['truth_shower_sid'],
    #                                                  dataset_analysis_dict['truth_shower_energy'],
    #                                                  dataset_analysis_dict['ticl_shower_sample_id'] if use_ticl else None,
    #                                                  dataset_analysis_dict['ticl_shower_sid_merged'] if use_ticl else None,
    #                                                  dataset_analysis_dict['ticl_shower_sid'] if use_ticl else None,
    #                                                  dataset_analysis_dict['ticl_shower_regressed_energy'] if use_ticl else None)
    # pdf.savefig()


    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - local fraction')
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes,
                                                                   dataset_analysis_dict['truth_shower_local_density'],
                                                                   dataset_analysis_dict['truth_shower_found_or_not'],
                                                                   dataset_analysis_dict[
                                                                       'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                   dataset_analysis_dict['truth_shower_energy'])
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes,
                                                                   dataset_analysis_dict['truth_shower_local_density'],
                                                                   dataset_analysis_dict['truth_shower_found_or_not'],
                                                                   dataset_analysis_dict[
                                                                       'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                   dataset_analysis_dict['truth_shower_energy'],
                                                                   energy_filter=2)
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes,
                                                                   dataset_analysis_dict['truth_shower_local_density'],
                                                                   dataset_analysis_dict['truth_shower_found_or_not'],
                                                                   dataset_analysis_dict[
                                                                       'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                   dataset_analysis_dict['truth_shower_energy'],
                                                                   make_segments=True)
    pdf.savefig()
    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - closest particle distance')
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_found_or_not'],
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_energy'])
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_found_or_not'],
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_energy'],
                                                                              energy_filter=2)
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_closest_partical_distance(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_found_or_not'],
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_energy'],
                                                                              make_segments=True)
    pdf.savefig()
    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Efficiency comparison - truth energy')
    pdf.savefig()

    fig = plt.figure()
    efficiency_comparison_plot_with_distribution_fo_truth_energy(plt, fig.axes,
                                                                 dataset_analysis_dict['truth_shower_found_or_not'],
                                                                 dataset_analysis_dict[
                                                                     'truth_shower_found_or_not_ticl'] if use_ticl else None,
                                                                 dataset_analysis_dict['truth_shower_energy'])
    pdf.savefig()

    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Response comparison - local fraction')
    pdf.savefig()

    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes,
                                                                 dataset_analysis_dict['truth_shower_local_density'],
                                                                 dataset_analysis_dict[
                                                                     'truth_shower_matched_energy_regressed'],
                                                                 dataset_analysis_dict[
                                                                     'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                 dataset_analysis_dict['truth_shower_energy'])
    pdf.savefig()

    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes,
                                                                 dataset_analysis_dict['truth_shower_local_density'],
                                                                 dataset_analysis_dict[
                                                                     'truth_shower_matched_energy_regressed'],
                                                                 dataset_analysis_dict[
                                                                     'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                 dataset_analysis_dict['truth_shower_energy'],
                                                                 energy_filter=2)
    pdf.savefig()

    fig1, fig2 = response_comparison_plot_with_distribution_fo_local_fraction(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_local_density'], dataset_analysis_dict['truth_shower_matched_energy_regressed'],
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                              dataset_analysis_dict[
                                                                                  'truth_shower_energy'],
                                                                              make_segments=True)
    pdf.savefig(fig1)
    pdf.savefig(fig2)

    #################################################################################
    draw_text_page(plt, 'Response comparison - closest particle distance')
    pdf.savefig()

    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_matched_energy_regressed'],
                                                                            dataset_analysis_dict[
                                                                                'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                            dataset_analysis_dict[
                                                                                'truth_shower_energy'])
    pdf.savefig()

    fig = plt.figure()
    response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes, dataset_analysis_dict[
        'truth_shower_closest_particle_distance'], dataset_analysis_dict['truth_shower_matched_energy_regressed'],
                                                                            dataset_analysis_dict[
                                                                                'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                            dataset_analysis_dict[
                                                                                'truth_shower_energy'], energy_filter=2)
    pdf.savefig()

    fig1, fig2 = response_comparison_plot_with_distribution_fo_closest_particle_distance(plt, fig.axes,
                                                                                         dataset_analysis_dict[
                                                                                             'truth_shower_closest_particle_distance'],
                                                                                         dataset_analysis_dict[
                                                                                             'truth_shower_matched_energy_regressed'],
                                                                                         dataset_analysis_dict[
                                                                                             'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None,
                                                                                         dataset_analysis_dict[
                                                                                             'truth_shower_energy'],
                                                                                         make_segments=True)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    #################################################################################

    #################################################################################
    draw_text_page(plt, 'Response comparison - function of truth energy')
    pdf.savefig()

    fig = plt.figure()
    response_curve_comparison_with_distribution_fo_energy(plt, fig.axes, dataset_analysis_dict['truth_shower_energy'],
                                                          dataset_analysis_dict[
                                                              'truth_shower_matched_energy_regressed'],
                                                          dataset_analysis_dict[
                                                              'truth_shower_matched_energy_regressed_ticl'] if use_ticl else None)
    pdf.savefig()



    #################################################################################
    draw_text_page(plt, 'Dataset stats')
    fig = plt.figure()
    make_histogram_of_number_of_rechits_per_segment(plt, fig.axes, dataset_analysis_dict['endcap_num_rechits'])
    pdf.savefig()
    fig = plt.figure()
    make_histogram_of_number_of_rechits_per_shower(plt, fig.axes, dataset_analysis_dict['truth_shower_num_rechits'])
    pdf.savefig()
    fig = plt.figure()
    # make_histogram_of_number_of_showers_per_segment(plt, fig.axes, dataset_analysis_dict['truth_shower_sample_id'])
    # pdf.savefig()



    #################################################################################


    #################################################################################
    # draw_text_page(plt, 'Visualization qualtitative')
    # # pdf.savefig()
    #
    #
    # x = 0
    # for vis_dict in dataset_analysis_dict['visualized_segments']:
    #     print("Plotting full",x)
    #     x+=1
    #     visualize_the_segment(plt, vis_dict['truth_sid'], vis_dict['pred_and_truth_dict'], vis_dict['feature_dict'],
    #         vis_dict['ticl_sid'], vis_dict['pred_sid'], vis_dict['coords_representatives'], dataset_analysis_dict['distance_threshold'])
    #     print("Saving fig")
    #     plt.savefig('x%d.png'%x)

    # def visualize_the_segment(plt, truth_showers_this_segment, truth_and_pred_dict, feature_dict, ticl_showers, labels,
    #                           coords_representative_predicted_showers, distance_threshold):


    #################################################################################

    pdf.close()





def analyse_hgcal_endcap(features, truth, predictions, beta_threshold,
                         distance_threshold, iou_threshold, endcap_id, should_return_visualization_data=False,
                         soft=False, with_local_distance_scaling=False):
    for key in predictions.keys():
        if type(predictions[key]) is not np.ndarray:
           predictions[key] = predictions[key].numpy()

    results_dict = HGCalAnalyzer(features, truth, predictions, beta_threshold,
                                  distance_threshold, iou_threshold, endcap_id,
                                  should_return_visualization_data=should_return_visualization_data,
                                  soft=soft, with_local_distance_scaling=with_local_distance_scaling).analyse()

    return results_dict



def analyse_hgcal_endcap_parallel(features, truth, predictions, beta_threshold,
                         distance_threshold, iou_threshold, endcap_id, should_return_visualization_data,
                         soft, main_thread):

    for key in predictions.keys():
        if type(predictions[key]) is not np.ndarray:
           predictions[key] = predictions[key].numpy()


    results_dict = HGCalAnalyzer(features, truth, predictions, beta_threshold,
                                  distance_threshold, iou_threshold, endcap_id,
                                  should_return_visualization_data=should_return_visualization_data,
                                  soft=soft, main_thread=main_thread).analyse()
    #
    return results_dict



endcap_id = 0

import multiprocessing
def analyse_multiple_endcaps_multi_cpu(data_loaded, soft, beta_threshold, distance_threshold, iou_threshold, limit=-1):
    # from global_database_manager import database_manager, x_counter
    global endcap_id

    params_list=[]
    processes_list = []

    m = multiprocessing.Manager()
    queue = m.Queue()
    t = threading.Thread(target=calculate_iou_main_thread, args=(queue,))
    t.start()


    import time

    t1 = time.time()
    x = 0

    for endcap_features, endcap_truth, endcap_predictions in data_loaded:
        # print("Endcap ", endcap_id)

        # if x_counter < 10:
        #     print("Inserting")
        #     pp_dist = endcap_predictions['pred_dist'][:, 0].numpy()
        #     pp_dist = np.random.choice(pp_dist, 1000)
        #     database_manager.insert_experiment_data('pred_dist',
        #                                             {'pred_dist': pp_dist})
        #     x_counter = x_counter +1
        # else:
        #     print("Closing")
        #     database_manager.close()
        #     0 / 0

        params_list += [(endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,
                                                        endcap_id, False, soft, queue)]

        # t = Process(target=analyse_hgcal_endcap_parallel, args=params_list)
        # t.start()

        # processes_list.append(t)

        endcap_id += 1
        x += 1

        if limit == x and limit != -1:
            break

    pool = Pool(20)

    result = [pool.apply_async( analyse_hgcal_endcap_parallel, t) for t in params_list]

    # print("Done???")
    results = []
    for r in result:
        results.append(r.get())

    print("TOok ", time.time()-t1)

    pool.close()
    pool.join()

    queue.put(None)
    t.join()

    return results

import time

def analyse_multiple_endcaps_single_cpu(data_loaded, soft, beta_threshold, distance_threshold, iou_threshold):

    global endcap_id

    results = []
    t1 = time.time()
    x = 0
    for endcap_features, endcap_truth, endcap_predictions in data_loaded:
        # print("Endcap ", endcap_id)

        # params_list += [(endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,
        #                                                 endcap_id, False, soft, queue)]

        # t = Process(target=analyse_hgcal_endcap_parallel, args=params_list)
        # t.start()

        # processes_list.append(t)
        results.append(analyse_hgcal_endcap(endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,endcap_id, False, soft))

        endcap_id += 1
        x += 1
    print("It took", time.time()-t1)

    return results



"""

    # params_list=[]
    for endcap_features, endcap_truth, endcap_predictions in data_loaded:
        print("Endcap ", endcap_id)
        params_list = (endcap_features, endcap_truth, endcap_predictions, beta_threshold, distance_threshold, iou_threshold,
                                                        endcap_id, False, soft)

        t = threading.Thread(target=analyse_hgcal_endcap, args=params_list)
        t.start()

        endcap_id += 1
        # break
"""