import pdb
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from numba import njit

def x_y_z_to_eta_phi_theta(x, y, z):
    # phi = np.arctan(y / x)
    phi = np.arctan2(y, x)
    s = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(s, z)
    eta = - np.log(np.tan(theta / 2))

    return eta, phi, theta

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

    def __init__(self, match_mode, iou_threshold, de_e_cut, angle_cut, shower0=False):
        super().__init__()
        self.match_mode=match_mode
        self.iou_threshold=iou_threshold
        self.de_e_cut=de_e_cut
        self.angle_cut=angle_cut
        self.iom_threshold=0.9
        self.shower0 = shower0


    def set_inputs(self, features_dict, truth_dict, predictions_dict, pred_alpha_idx):
        self.features_dict = features_dict.copy()
        self.truth_dict = truth_dict.copy()
        self.predictions_dict = predictions_dict.copy()
        self.pred_alpha_idx = pred_alpha_idx.copy()


    def _assign_additional_attr(self):
        x = self.features_dict['recHitX']
        y = self.features_dict['recHitY']
        z = self.features_dict['recHitZ']
        e = self.features_dict['recHitEnergy']
        truth_sid = self.truth_dict['truthHitAssignementIdx'][:, 0].astype(np.int32) * 1
        pred_sid = self.pred_sid * 1

        for n, attr in self.calculated_graph.nodes(data=True):
            if attr['type'] == ShowersMatcher._NODE_TYPE_TRUTH_SHOWER:
                filt = truth_sid == n

                e2 = e[filt]
                x2 = np.sum(x[filt] * e2) / np.sum(e2)
                y2 = np.sum(y[filt] * e2) / np.sum(e2)
                z2 = np.sum(z[filt] * e2) / np.sum(e2)
                eta, phi, _ = x_y_z_to_eta_phi_theta(x2, y2, z2)
                attr['truth_mean_x'] = x2
                attr['truth_mean_y'] = y2
                attr['truth_mean_z'] = z2

                # assigned_to_noise_frac = np.sum(e[np.logical_and(truth_sid==-1, pred_sid!=-1)]) / np.sum(e[truth_sid==-1])
                # attr['assigned_to_noise_frac'] = assigned_to_noise_frac

            else:
                filt = pred_sid == n

                e2 = e[filt]
                x2 = np.sum(x[filt] * e2) / np.sum(e2)
                y2 = np.sum(y[filt] * e2) / np.sum(e2)
                z2 = np.sum(z[filt] * e2) / np.sum(e2)
                eta, phi, _ = x_y_z_to_eta_phi_theta(x2, y2, z2)
                attr['pred_mean_x'] = x2
                attr['pred_mean_y'] = y2
                attr['pred_mean_z'] = z2

                # noise_frac = np.sum(e[np.logical_and(truth_sid==-1, filt)]) / np.sum(e2)
                # assigned_to_noise_frac = np.sum(e[np.logical_and(truth_sid==-1, filt)]) / np.sum(e[truth_sid==-1])
                # print("noise frac", noise_frac, np.sum(e2))
                # attr['noise_frac'] = noise_frac
                # attr['assigned_to_noise_frac'] = assigned_to_noise_frac
                # print("A2N", assigned_to_noise_frac)


    def _build_data_graph(self):
        """
        Builds a graph in which all truth showers and all predicted showers are nodes
        This does not yet build any edges.
        """

        graph = nx.Graph()

        truth_sid = self.truth_dict['truthHitAssignementIdx'][:, 0].astype(np.int32)
        truth_shower_sid, truth_shower_idx = np.unique(truth_sid, return_index=True)
        f = truth_shower_sid !=- 1
        truth_shower_sid, truth_shower_idx = truth_shower_sid[f], truth_shower_idx[f]

        truth_nodes = []
        keys = self.truth_dict.keys()

        # Building adding truth nodes to the graph
        if self.shower0:
            n_truth_showers = 1
        else:
            n_truth_showers = len(truth_shower_sid)
        for i in range(n_truth_showers):
            node_attributes = dict()

            for k in keys:
                node_attributes[k] = self.truth_dict[k][truth_shower_idx[i], 0]

            node_attributes['type'] = ShowersMatcher._NODE_TYPE_TRUTH_SHOWER

            node = (int(truth_shower_sid[i]), node_attributes)
            truth_nodes.append(node)

        graph.add_nodes_from(truth_nodes)

        offset = np.max(truth_shower_sid) + 1000
        pred_sid = self.predictions_dict['pred_sid']
        pred_sid = np.where(pred_sid==-1, -1, pred_sid + offset)   # Offset node ids for pred nodes, can also be +1.
        self.pred_sid = pred_sid
        pred_shower_sid = [pred_sid[x,0] for x in self.pred_alpha_idx if pred_sid[x] != -1]

        keys = self.predictions_dict.keys()
        pred_nodes = []
        skip = ['row_splits', 'rechit_energy', 'no_noise_rs', 'noise_backscatter']

        # Adding nodes for the predicted showers
        for i in range(len(pred_shower_sid)):
            node_attributes = dict()
            for k in keys:
                if k in skip:
                    continue
                elif k == 'pred_id':
                    # print("SHAPE pred_id: ", self.predictions_dict[k].shape)
                    # print("UNIQUE pred_id: ", np.unique(self.predictions_dict[k]))
                    # node_attributes[k] = np.argmax(
                            # self.predictions_dict[k][self.pred_alpha_idx[i]],
                            # axis=1)
                    # import pdb
                    # pdb.set_trace()
                    node_attributes[k] = self.predictions_dict[k][self.pred_alpha_idx[i]]
                    # Same as 'else' should work.
                else:
                    # print("X", k)
                    node_attributes[k] = self.predictions_dict[k][self.pred_alpha_idx[i], 0]
            node = (pred_shower_sid[i], node_attributes)
            node_attributes['type'] = ShowersMatcher._NODE_TYPE_PRED_SHOWER

            pred_nodes.append(node)

        graph.add_nodes_from(pred_nodes)

        self.graph = graph


    def _cost_matrix_angle_based(self, truth_shower_sid, pred_shower_sid):
        #SHAH RUKH: This might be the better solution
        # However: Our predicted showers currently don't contain
        # the necessary information to directly calculate an angle.
        #   This is fixable, but one has to decide if the tracks should play
        #   a part in this or not. They will have a huge impact due to their high energy 
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
        is_track = np.abs(self.features_dict['recHitZ'][:,0]) == 315 # TODO: This works for now, but is very hacky
        weight[is_track] = 0


        iou_matrix = calculate_iou_serial_fast(self.truth_dict['truthHitAssignementIdx'][:, 0],
                                               self.pred_sid,
                                               truth_shower_sid,
                                               pred_shower_sid,
                                               weight)

        # print("IOU Matrix", iou_matrix)
        # 0/0
        secondary_condition = np.ones((len(pred_shower_energy), len(truth_shower_energy)), bool)

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
        #SHAH RUKH:
        # 1. match on tracks (i.e. match the predicted shower containing the
        #       track to its truth shower
        # 2. match on hits as we already do
        # 3. My problem: I have to exclude showers that have already been matched
        #       without messing up any of the following code
        #       a.) I can't remove the nodes, otherwise the dataframe will not work anymore
        #       b.) The cost matrix's indices are used to draw the edges in the graph
        #           calculating a smaller cost matrix will lead to wrong edges
        truth_shower_sid = [
            x[0]
            for x in self.graph.nodes(data=True)
            if x[1]['type']==ShowersMatcher._NODE_TYPE_TRUTH_SHOWER
            ]
        pred_shower_sid = [
            x[0]
            for x in self.graph.nodes(data=True)
            if x[1]['type']==ShowersMatcher._NODE_TYPE_PRED_SHOWER
            ]

        if not len(truth_shower_sid) > 0:
            print("No truth showers")
            return
        if not len(pred_shower_sid) > 0:
            print("No predicted showers")
            return
        if not self.match_mode == 'iou_max':
            raise NotImplementedError("Only use iou_max for now")


        C = self._cost_matrix_intersection_based(truth_shower_sid, pred_shower_sid)
        pdb.set_trace()

        row_id, col_id = linear_sum_assignment(C, maximize=True)

        self.truth_sid = self.truth_dict['truthHitAssignementIdx'][:, 0]
        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.graph.nodes(data=True))

        for p, t in zip(row_id, col_id):
            if C[p, t] > 0:
                matched_full_graph.add_edge(
                    truth_shower_sid[t],
                    pred_shower_sid[p],
                    attached_in_pass=0)

        self.calculated_graph = matched_full_graph


    def process(self, extra=False):
        self._build_data_graph()
        if self.match_mode == 'iou_max':
            self._match_single_pass()
        else:
            raise NotImplementedError('Match mode not found')
        if extra:
            self._assign_additional_attr()


    def get_matched_hit_sids(self):
        pred_sid = self.pred_sid[:, 0]
        pred_sid_2 = np.zeros_like(pred_sid) -1
        truth_sid = self.truth_sid
        truth_sid_2 = np.zeros_like(truth_sid) -1

        done = set()
        index = 0


        done_p = set()
        for n, attr in self.calculated_graph.nodes(data=True):
            if n in done:
                continue

            # print("Node is", n, index, attr)

            if attr['type']==ShowersMatcher._NODE_TYPE_TRUTH_SHOWER:
                truth_sid_2[truth_sid==n] = index
                # print("Hello")
            else:
                done_p.add(n)
                pred_sid_2[pred_sid==n] = index

            done.add(n)

            N = list(self.calculated_graph.neighbors(n))
            assert len(N) == 0 or len(N) == 1

            if len(N) == 1:
                neighbor = N[0]

                neighbor_data = self.calculated_graph.nodes[neighbor]
                done.add(neighbor)

                if neighbor_data['type'] == ShowersMatcher._NODE_TYPE_TRUTH_SHOWER:
                    truth_sid_2[truth_sid == neighbor] = index
                    # print("Hello")
                else:
                    pred_sid_2[pred_sid == neighbor] = index

            index += 1

        # event_pred_dataframe = None
        # event_truth_dataframe = None
        # 0/0
        return truth_sid_2, pred_sid_2


    def get_result_as_dataframe(self):
        event_variables = None # Not sure if it should be a dataframe, can just be a dictionary
        skip_keys = {'type'}

        my_calculated_graph = self.calculated_graph.copy()
        keys_ = [list(attr.keys()) for n, attr in my_calculated_graph.nodes(data=True)]
        keys_ = [item for sublist in keys_ for item in sublist]
        keys_ = set([k for k in keys_ if k not in skip_keys])
        result_data = {k:[] for k in keys_}
        done = set()
        for n, attr in my_calculated_graph.nodes(data=True):
            if n in done:
                continue
            done.add(n)

            N = list(my_calculated_graph.neighbors(n))
            assert len(N) == 0 or len(N) == 1

            if attr['type']==1 and len(N)==0:
                # print("X", attr) # This caused everything to be very verbose
                pass

            if len(N) == 1:
                attr.update(my_calculated_graph.nodes[N[0]])
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


