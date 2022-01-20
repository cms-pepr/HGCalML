import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

def calculate_iou_tf(truth_sid,
                     pred_sid,
                     truth_shower_sid,
                     pred_shower_sid,
                     hit_weight, return_all=False):

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
        truth_idx_2 = tf.zeros_like(truth_sid) -1
        pred_idx_2 = tf.zeros_like(pred_sid) -1
        hit_weight_2 = tf.zeros_like(hit_weight)

        # print("3.1")

        for i in range(len(pred_shower_sid)):
            pred_idx_2 = tf.where(pred_sid == pred_shower_sid[i], i, pred_idx_2)

        for i in range(len(truth_shower_sid)):
            truth_idx_2 = tf.where(truth_sid == truth_shower_sid[i], i, truth_idx_2)

        # print("3.3")
        one_hot_pred = tf.one_hot(pred_idx_2, depth=len_pred_showers)
        one_hot_truth = tf.one_hot(truth_idx_2, depth=len_truth_showers)

        intersection_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)

        pred_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], tf.ones_like(one_hot_truth),
                                        transpose_a=True)

        truth_sum_matrix = tf.linalg.matmul(
            tf.ones_like(one_hot_pred) * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)

        union_sum_matrix = pred_sum_matrix + truth_sum_matrix - intersection_sum_matrix

        overlap_matrix = (intersection_sum_matrix / union_sum_matrix).numpy()

        if return_all:
            return overlap_matrix, pred_sum_matrix, truth_sum_matrix, intersection_sum_matrix
        else:
            return overlap_matrix


def angle(p, t):
    t = np.array([t['x'], t['y'], t['z']])
    p = np.array([p['dep_x'], p['dep_y'], p['dep_z']])

    angle = np.arccos(np.sum(t*p) / (np.sqrt(np.sum(t*t))*np.sqrt(np.sum(p*p))))

    return angle


class ShowersMatcher:
    _NODE_TYPE_TRUTH_SHOWER = 0
    _NODE_TYPE_PRED_SHOWER = 1
    _NODE_TYPE_RECHIT = 7

    def __init__(self, match_mode, iou_threshold, de_e_cut, angle_cut):
        super().__init__()
        self.match_mode=match_mode
        self.iou_threshold=iou_threshold
        self.de_e_cut=de_e_cut
        self.angle_cut=angle_cut


    def set_inputs(self, features_dict, truth_dict, predictions_dict, pred_alpha_idx):
        self.features_dict = features_dict.copy()
        self.truth_dict = truth_dict.copy()
        self.predictions_dict = predictions_dict.copy()
        self.pred_alpha_idx = pred_alpha_idx.copy()

    def _build_data_graph(self):
        graph = nx.Graph()

        truth_sid = self.truth_dict['truthHitAssignementIdx'][:, 0].astype(np.int32)
        truth_shower_sid, truth_shower_idx = np.unique(truth_sid, return_index=True)
        f = truth_shower_sid !=- 1
        truth_shower_sid, truth_shower_idx = truth_shower_sid[f], truth_shower_idx[f]

        truth_nodes = []
        keys = self.truth_dict.keys()

        for i in range(len(truth_shower_sid)):
            node_attributes = dict()

            for k in keys:
                node_attributes[k] = self.truth_dict[k][truth_shower_idx[i], 0]

            node_attributes['type'] = ShowersMatcher._NODE_TYPE_TRUTH_SHOWER

            node = (int(truth_shower_sid[i]), node_attributes)
            truth_nodes.append(node)

        graph.add_nodes_from(truth_nodes)

        pred_sid = self.predictions_dict['pred_sid'] + np.max(truth_shower_sid) + 1000   # Offset node ids for pred nodes, can also be +1.
        self.pred_sid = pred_sid
        pred_shower_sid = [pred_sid[x,0] for x in self.pred_alpha_idx if pred_sid[x] != -1]

        keys = self.predictions_dict.keys()
        pred_nodes = []
        skip = {'row_splits'}

        for i in range(len(pred_shower_sid)):
            node_attributes = dict()
            for k in keys:
                if k in skip:
                    continue
                node_attributes[k] = self.predictions_dict[k][self.pred_alpha_idx[i], 0]
            node = (pred_shower_sid[i], node_attributes)
            node_attributes['type'] = ShowersMatcher._NODE_TYPE_PRED_SHOWER

            pred_nodes.append(node)

        graph.add_nodes_from(pred_nodes)

        self.graph = graph

    def _cost_matrix_intersection_based(self, truth_shower_sid, pred_shower_sid):
        pred_shower_energy = [self.graph.nodes[x]['pred_energy'] for x in pred_shower_sid]
        truth_shower_energy = [self.graph.nodes[x]['truthHitAssignedEnergies'] for x in truth_shower_sid]
        weight = self.features_dict['recHitEnergy'][:, 0]
        iou_matrix = calculate_iou_tf(self.truth_dict['truthHitAssignementIdx'][:, 0],
                                      self.pred_sid[:, 0],
                                       truth_shower_sid,
                                       pred_shower_sid,
                                       weight)

        if self.de_e_cut == -1:
            allow = lambda i,j: True
        else:
            de_e_cut_on_matching = self.de_e_cut
            allow = lambda i,j : (np.abs(pred_shower_energy[i] - truth_shower_energy[j]) / truth_shower_energy[j]) < de_e_cut_on_matching

        n = max(len(truth_shower_sid), len(pred_shower_sid))
        C = np.zeros((n, n))
        if self.match_mode == 'iou_max':
            for i in range(len(pred_shower_sid)):
                for j in range(len(truth_shower_sid)):
                    overlap = iou_matrix[i, j]
                    if overlap >= self.iou_threshold and allow(i,j):
                        C[i, j] = overlap
        elif self.match_mode == 'emax_iou':
            for i in range(len(pred_shower_sid)):
                for j in range(len(truth_shower_sid)):
                    overlap = iou_matrix[i, j]
                    if overlap >= self.iou_threshold and allow(i,j):
                        C[i, j] = min(self.graph.nodes[truth_shower_sid[j]]['energy'], self.graph.nodes[pred_shower_sid[i]]['energy'])
        return C


    def _cost_matrix_angle_based(self, truth_shower_sid, pred_shower_sid):
        pred_shower_energy = [self.graph.nodes[x]['dep_energy'] for x in pred_shower_sid]
        truth_shower_energy = [self.graph.nodes[x]['energy'] for x in truth_shower_sid]
        if self.de_e_cut==-1:
            allow = lambda i,j: True
        else:
            de_e_cut_on_matching = self.de_e_cut
            allow = lambda i,j : (np.abs(pred_shower_energy[i] - truth_shower_energy[j]) / truth_shower_energy[j]) < de_e_cut_on_matching

        n = max(len(truth_shower_sid), len(pred_shower_sid))
        C = np.zeros((n, n))

        for a, j in enumerate(pred_shower_sid):
            x = self.graph.nodes(data=True)[j]
            for b, i in enumerate(truth_shower_sid):
                y = self.graph.nodes(data=True)[i]
                if angle(x,y) < self.angle_cut and allow(x,y):
                    C[a, b] = min(x['energy'],
                                  y['energy'])

        return C


    def _match_single_pass(self):
        truth_shower_sid = [x[0] for x in self.graph.nodes(data=True) if x[1]['type']==ShowersMatcher._NODE_TYPE_TRUTH_SHOWER]
        pred_shower_sid = [x[0] for x in self.graph.nodes(data=True) if x[1]['type']==ShowersMatcher._NODE_TYPE_PRED_SHOWER]

        if self.match_mode == 'iou_max' or self.match_mode == 'emax_iou':
            C = self._cost_matrix_intersection_based(truth_shower_sid, pred_shower_sid)
        elif self.match_mode == 'emax_angle':
            C = self._cost_matrix_angle_based(truth_shower_sid, pred_shower_sid)
        else:
            raise NotImplementedError('Error in match mode')

        row_id, col_id = linear_sum_assignment(C, maximize=True)

        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.graph.nodes(data=True))

        for p, t in zip(row_id, col_id):
            if C[p, t] > 0:
                matched_full_graph.add_edge(truth_shower_sid[t], pred_shower_sid[p], attached_in_pass=0)

        self.calculated_graph = matched_full_graph

    # def _reduce_graph(self, graph):
    #     pairs = []  # List of all the pairs to which to attach to
    #     free_nodes = []  # List of all the free nodes (truth or pred)
    #
    #     connected_components = nx.connected_components(graph)
    #     for c in connected_components:
    #         if len(c) > 1:
    #             pred_c = []
    #             truth_c = []
    #             for n in c:
    #                 data_n = graph.nodes(data=True)[n]
    #                 if graph.nodes(data=True)[n]['type'] == ShowersMatcher._NODE_TYPE_TRUTH_SHOWER:
    #                     truth_c.append((n, data_n))
    #                 else:
    #                     pred_c.append((n, data_n))
    #             pairs.append((pred_c, truth_c))
    #         else:
    #             free_nodes.append(c.pop())
    #
    #     reduced_graph = nx.Graph()
    #     for x in free_nodes:
    #         reduced_graph.add_nodes_from([(x, graph.nodes(data=True)[x])])
    #
    #     for x in pairs:
    #         # TODO: This needs to be fixed
    #         u = self.reduce_pred_nodes(x[0])
    #         v = self.reduce_truth_nodes(x[1])
    #         reduced_graph.add_nodes_from([u, v])
    #         reduced_graph.add_edge(u[0], v[0])
    #
    #     return reduced_graph
    #
    # def _match_multipass(self):
    #     truth_shower_sid = [x for x in self.graph.nodes()]
    #     pred_shower_sid = [x for x in self.graph.nodes()]
    #     for i, x in enumerate(self.graph.nodes(data=True)):
    #         x[1]['idx'] = i
    #     for i, x in enumerate(self.graph.nodes(data=True)):
    #         x[1]['idx'] = i
    #
    #     weight = self.features_dict['recHitEnergy'][:, 0]
    #
    #     iou_matrix, pred_sum_matrix, truth_sum_matrix, intersection_matrix = calculate_iou_tf(self.truth_dict['truthHitAssignementIdx'][:,0],
    #                                                                                           self.pred_sid,
    #                                                                                           truth_shower_sid,
    #                                                                                           pred_shower_sid,
    #                                                                                           weight,
    #                                                                                           return_all=True)
    #     intersection_matrix = intersection_matrix.numpy()
    #     pred_sum_matrix = pred_sum_matrix.numpy()
    #     truth_sum_matrix = truth_sum_matrix.numpy()
    #
    #     min_matrix = intersection_matrix / np.minimum(pred_sum_matrix, truth_sum_matrix)
    #
    #     n = max(len(truth_shower_sid), len(pred_shower_sid))
    #     # Cost matrix
    #     C = np.zeros((n, n))
    #     for i in range(len(pred_shower_sid)):
    #         for j in range(len(truth_shower_sid)):
    #             overlap = min_matrix[i, j]
    #             if overlap >= self.iou_threshold:
    #                 C[i, j] = overlap
    #
    #     row_id, col_id = linear_sum_assignment(C, maximize=True)
    #
    #     matched_full_graph = nx.Graph()
    #     matched_full_graph.add_nodes_from(self.graph.nodes(data=True))
    #
    #     for p, t in zip(row_id, col_id):
    #         if C[p, t] > 0:
    #             matched_full_graph.add_edge(truth_shower_sid[t], pred_shower_sid[p], attached_in_pass=0)
    #
    #     passes = [x for x in range(self.passes)]
    #     passes.pop(0)
    #
    #     graphs = []
    #     graphs.append(matched_full_graph.copy())
    #     for npass in passes:
    #         pairs = []  # List of all the pairs to which to attach to
    #         free_nodes = []  # List of all the free nodes (truth or pred)
    #
    #         connected_components = nx.connected_components(matched_full_graph)
    #         for c in connected_components:
    #             if len(c) > 1:
    #                 pred_c = set()
    #                 truth_c = set()
    #                 for n in c:
    #                     if matched_full_graph.nodes(data=True)[n]['type'] == NODE_TYPE_TRUTH_SHOWER:
    #                         truth_c.add(n)
    #                     else:
    #                         pred_c.add(n)
    #                 pairs.append((pred_c, truth_c))
    #             else:
    #                 free_nodes.append(c.pop())
    #
    #         # Construct another cost matrix
    #         C = np.zeros((len(pairs), len(free_nodes)))
    #         for i, p in enumerate(pairs):
    #             for j, f in enumerate(free_nodes):
    #                 score = 0
    #                 type_f = matched_full_graph.nodes(data=True)[f]['type']
    #                 idx_f = matched_full_graph.nodes(data=True)[f]['idx']
    #                 matching_mode = -1
    #                 # Length of either p[0] will be 1 or p[1] will be 1
    #                 if len(p[0]) == 1:
    #                     # A pred shower is matched to one or more truth showers
    #                     if type_f == 0:
    #                         matching_mode = 0
    #                     else:
    #                         pass
    #                 if len(p[1]) == 1:
    #                     # A truth shower is matched to one or more pred showers
    #                     if type_f == 0:
    #                         pass
    #                     else:
    #                         matching_mode = 1
    #
    #                 if matching_mode == 0:
    #                     sid_p = next(iter(p[0]))
    #                     idx_p = matched_full_graph.nodes(data=True)[sid_p]['idx']
    #                     numerator = intersection_matrix[idx_p, idx_f]
    #                     denominator1 = pred_sum_matrix[idx_p, 0]
    #                     denominator2 = truth_sum_matrix[0, idx_f]
    #                     score = numerator / min(denominator1, denominator2)
    #                 elif matching_mode == 1:
    #                     sid_t = next(iter(p[1]))
    #                     idx_t = matched_full_graph.nodes(data=True)[sid_t]['idx']
    #                     numerator = intersection_matrix[idx_f, idx_t]
    #                     denominator1 = truth_sum_matrix[0, idx_t]
    #                     denominator2 = pred_sum_matrix[idx_f, 0]
    #                     score = numerator / min(denominator1, denominator2)
    #                 if score > self.iou_threshold:
    #                     C[i, j] = score
    #
    #         # # Let's match these guys again
    #         C = np.array(C)
    #         row_id, col_id = linear_sum_assignment(C, maximize=True)
    #         for r, c in zip(row_id, col_id):
    #             if C[r, c] > 0:
    #                 p = pairs[r]
    #                 if matched_full_graph.nodes(data=True)[free_nodes[c]]['type'] == NODE_TYPE_TRUTH_SHOWER:
    #                     # Free node was a truth node
    #                     matched_full_graph.add_edge(next(iter(p[0])), free_nodes[c], attached_in_pass=npass)
    #                 else:
    #                     # Free node was a pred node
    #                     matched_full_graph.add_edge(next(iter(p[1])), free_nodes[c], attached_in_pass=npass)
    #         graphs.append(matched_full_graph.copy())
    #
    #
    #     self.calculated_graph = self._reduce_graph(matched_full_graph)


    def process(self):
        self._build_data_graph()

        if self.match_mode == 'iou_max':
            self._match_single_pass()
        elif self.match_mode == 'iom_max_multi':
            raise NotImplementedError('Multi passing needs to be re-written')
            # self._match_multipass()
        elif self.match_mode == 'emax_iou':
            raise NotImplementedError('Need to finish some work')
            # self._match_single_pass()
        elif self.match_mode == 'emax_angle':
            raise NotImplementedError('Need to finish some work')
            # self._match_single_pass()

    def get_hit_data(self):
        event_pred_dataframe = None
        event_truth_dataframe = None
        return event_truth_dataframe, event_pred_dataframe

    def get_result_as_dataframe(self):
        event_variables = None # Not sure if it should be a dataframe, can just be a dictionary
        skip_keys = {'type'}
        keys_ = [list(attr.keys()) for n, attr in self.calculated_graph.nodes(data=True)]
        keys_ = [item for sublist in keys_ for item in sublist]
        keys_ = [k for k in keys_ if k not in skip_keys]
        result_data = {k:[] for k in keys_}

        done = set()
        for n, attr in self.calculated_graph.nodes(data=True):
            if n in done:
                continue
            done.add(n)

            N = list(self.calculated_graph.neighbors(n))
            assert len(N) == 0 or len(N) == 1

            if len(N) == 1:
                attr.update(self.calculated_graph.nodes[N[0]])
                done.add(N[0])

            add = {key: np.NaN for key in keys_}

            for k,v in attr.items():
                if k in add:
                    add[k] = v

            for k,v in add.items():
                result_data[k].append(v)

        frame = pd.DataFrame(result_data)

        total_energy_truth = 0.0
        total_energy_pred = 0.0
        for n, attr in self.calculated_graph.nodes(data=True):
            pass

        return frame

    def get_result_as_graph(self):
        return self.calculated_graph

