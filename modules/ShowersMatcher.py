import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from numba import njit


def angle(p, t):
    t = np.array([t['x'], t['y'], t['z']])
    p = np.array([p['dep_x'], p['dep_y'], p['dep_z']])

    angle = np.arccos(np.sum(t*p) / (np.sqrt(np.sum(t*t))*np.sqrt(np.sum(p*p))))

    return angle


@njit
def _calculate_iou_serial_fast_comp(intersection_sum_matrix, pred_sum, truth_sum, truth_sid, pred_sid, hit_weight):
    if( not( len(truth_sid) == len(pred_sid) == len(hit_weight))):
        raise RuntimeError('Length of truth_sid, pred_sid and hig_weight must be the same')

    for i in range(len(truth_sid)):
        intersection_sum_matrix[pred_sid[i]+1,truth_sid[i]+1] += hit_weight[i]
        pred_sum[pred_sid[i]+1] += hit_weight[i]
        truth_sum[truth_sid[i]+1] += hit_weight[i]

@njit
def _find_idx(uniques, original):
    idx = np.zeros_like(original)
    for i in range(len(original)):
        for j in range(len(uniques)):
            if uniques[j] == original[i]:
                idx[i] = j + 1
                break
    return idx

def calculate_iou_serial_fast(truth_sid,
                     pred_sid,
                     truth_shower_sid,
                     pred_shower_sid,
                     hit_weight, return_all=False):

    truth_shower_sid = np.array(truth_shower_sid)
    pred_shower_sid = np.array(pred_shower_sid)

    t_unique, truth_sid_2 = np.unique(truth_sid, return_inverse=True)
    p_unique, pred_sid_2 = np.unique(pred_sid, return_inverse=True)

    t_idx = _find_idx(t_unique, truth_shower_sid)
    p_idx = _find_idx(p_unique, pred_shower_sid)
    # overlap_matrix = (intersection_sum_matrix / (union_sum_matrix + 1e-7)).numpy()

    intersection_matrix = np.zeros((len(p_unique)+1, len(t_unique)+1), np.float32)
    truth_sum = np.zeros((len(t_unique)+1), np.float32)
    pred_sum = np.zeros((len(p_unique)+1), np.float32)

    _calculate_iou_serial_fast_comp(intersection_matrix, pred_sum, truth_sum, truth_sid_2, pred_sid_2, hit_weight)


    intersection_matrix = intersection_matrix[p_idx]
    intersection_matrix = intersection_matrix[:, t_idx]

    truth_sum = truth_sum[t_idx]
    pred_sum = pred_sum[p_idx]


    union_matrix = pred_sum[:, np.newaxis] + truth_sum[np.newaxis, :] - intersection_matrix
    overlap_matrix = np.nan_to_num(intersection_matrix / union_matrix)

    if return_all:
        return overlap_matrix, pred_sum, truth_sum, intersection_matrix
    else:
        return overlap_matrix


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
        skip = ['row_splits', 'rechit_energy', 'no_noise_rs', 'noise_backscatter']

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

    def _cost_matrix_intersection_based(self, truth_shower_sid, pred_shower_sid, return_raw=False):
        pred_shower_energy = np.array([self.graph.nodes[x]['pred_energy'] for x in pred_shower_sid])
        truth_shower_energy = np.array([self.graph.nodes[x]['truthHitAssignedEnergies'] for x in truth_shower_sid])
        weight = self.features_dict['recHitEnergy'][:, 0]

        iou_matrix = calculate_iou_serial_fast(self.truth_dict['truthHitAssignementIdx'][:, 0],
                                               self.pred_sid,
                                               truth_shower_sid,
                                               pred_shower_sid,
                                               weight)

        # print("IOU Matrix", iou_matrix)
        # 0/0
        secondary_condition = np.ones((len(pred_shower_energy), len(truth_shower_energy)), np.bool)

        n = max(len(truth_shower_sid), len(pred_shower_sid))

        C = np.zeros((n, n))
        if self.de_e_cut != -1:
            c2 = np.abs(pred_shower_energy[:, np.newaxis] - truth_shower_energy[np.newaxis, :]) / - truth_shower_energy[
                                                                                                    np.newaxis,
                                                                                                    :] < self.de_e_cut
            secondary_condition = np.logical_and(c2, secondary_condition)

        if self.match_mode == 'iou_max':
            C_s = iou_matrix * 1.0
            C_s[iou_matrix < self.iou_threshold] = 0
            C_s[np.logical_not(secondary_condition)] = 0
            C[0:len(pred_shower_sid), 0:len(truth_shower_sid)] = C_s
        elif self.match_mode == 'emax_iou':
            C_s = np.min(pred_shower_energy[:, np.newaxis], truth_shower_energy[np.newaxis, :])
            C_s[iou_matrix < self.iou_threshold] = 0
            C_s[np.logical_not(secondary_condition)] = 0
            C[0:len(pred_shower_sid), 0:len(truth_shower_sid)] = C_s

        if return_raw:
            return C, iou_matrix
        return C

    def _match_single_pass(self):
        truth_shower_sid = [x[0] for x in self.graph.nodes(data=True) if x[1]['type']==ShowersMatcher._NODE_TYPE_TRUTH_SHOWER]
        pred_shower_sid = [x[0] for x in self.graph.nodes(data=True) if x[1]['type']==ShowersMatcher._NODE_TYPE_PRED_SHOWER]

        if len(truth_shower_sid) > 0 and len(pred_shower_sid) > 0:
            if self.match_mode == 'iou_max' or self.match_mode == 'emax_iou':
                C = self._cost_matrix_intersection_based(truth_shower_sid, pred_shower_sid)
            elif self.match_mode == 'emax_angle':
                C = self._cost_matrix_angle_based(truth_shower_sid, pred_shower_sid)
            else:
                raise NotImplementedError('Error in match mode')

            row_id, col_id = linear_sum_assignment(C, maximize=True)

        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.graph.nodes(data=True))

        if len(truth_shower_sid) > 0 and len(pred_shower_sid) > 0:
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
        keys_ = set([k for k in keys_ if k not in skip_keys])
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

