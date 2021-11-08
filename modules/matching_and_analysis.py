
import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import linear_sum_assignment
import pickle
from graph_functions import *
import networkx as nx
import gzip
import scalar_metrics



# Matching types
MATCHING_TYPE_IOU_MAX = 0
MATCHING_TYPE_MAX_FOUND = 1
MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD = 2
MATCHING_TYPE_MAX_PRECISION_ANGLE_THRESHOLD = 3
MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD_PRECISION_THRESHOLD = 4
MATCHING_TYPE_IOM_MAX_MULTIPASS = 5


NODE_TYPE_TRUTH_SHOWER = 0
NODE_TYPE_PRED_SHOWER = 1
NODE_TYPE_RECHIT = 7

def angle(p, t):
    t = np.array([t['x'], t['y'], t['z']])
    p = np.array([p['dep_x'], p['dep_y'], p['dep_z']])

    angle = np.arccos(np.sum(t*p) / (np.sqrt(np.sum(t*t))*np.sqrt(np.sum(p*p))))

    return angle



def precision_function(x, y, angle_threshold):
    e1 = max(x['energy'], 0)
    e2 = y['energy']
    thisp = min(e1 / e2, e2 / e1) if e1 != 0. else 0
    thisp = thisp * (angle(x, y) <= angle_threshold)

    return thisp

def get_truth_matched_attribute(graphs_list, attribute_name_truth, attribute_name_pred, numpy=False, not_found_value=-1, sum_multi=False):
    truth_data = []
    pred_data = []
    for g in graphs_list:
        for n, att in g.nodes(data=True):
            if att['type'] == NODE_TYPE_TRUTH_SHOWER:
                matched = [x for x in g.neighbors(n)]
                if len(matched)==0:
                    pred_data.append(None)
                elif len(matched)==1:
                    pred_data.append( g.nodes(data=True)[matched[0]][attribute_name_pred])
                elif len(matched) == 2:
                    if not sum_multi:
                        raise RuntimeError()
                    pred_data.append( g.nodes(data=True)[matched[0]][attribute_name_pred] + g.nodes(data=True)[matched[1]][attribute_name_pred])

                else:
                    raise RuntimeError("Truth shower matched to multiple pred showers?")
                truth_data.append(att[attribute_name_truth])
    if numpy:
        truth_data = np.array(truth_data)
        pred_data = [x if x is not None else not_found_value for x in pred_data]
        pred_data =np.array(pred_data)

    return truth_data, pred_data



def get_pred_matched_attribute(graphs_list, attribute_name_truth, attribute_name_pred, numpy=False, not_found_value=-1, sum_multi=False):
    pred_data = []
    truth_data = []
    for g in graphs_list:
        for n, att in g.nodes(data=True):
            if att['type'] == NODE_TYPE_PRED_SHOWER:
                matched = [x for x in g.neighbors(n)]
                if len(matched)==0:
                    truth_data.append(None)
                elif len(matched)==1:
                    truth_data.append(g.nodes(data=True)[matched[0]][attribute_name_pred])
                elif len(matched) == 2:
                    if not sum_multi:
                        raise RuntimeError()
                    truth_data.append(g.nodes(data=True)[matched[0]][attribute_name_pred] + g.nodes(data=True)[matched[1]][attribute_name_pred])
                else:
                    raise RuntimeError("Pred shower matched to multiple truth showers?")

                pred_data.append(att[attribute_name_truth])
    if numpy:
        pred_data = np.array(pred_data)
        truth_data = [x if x is not None else not_found_value for x in truth_data]
        truth_data =np.array(truth_data)

    return pred_data, truth_data



def build_metadeta_dict(beta_threshold=0.5, distance_threshold=0.5, iou_threshold=0.0001, matching_type=MATCHING_TYPE_MAX_FOUND,
                        with_local_distance_scaling=False, beta_weighting_param=1, angle_threshold=0.08, precision_threshold=0.2,
                        passes=5):
    metadata = dict()
    metadata['beta_threshold'] = beta_threshold
    metadata['distance_threshold'] = distance_threshold
    metadata['iou_threshold'] = iou_threshold
    metadata['matching_type'] = matching_type
    metadata['with_local_distance_scaling'] = with_local_distance_scaling

    metadata['reco_score'] = -1
    metadata['pred_energy_matched'] = -1
    metadata['truth_energy_matched'] = -1
    metadata['angle_threshold'] = angle_threshold
    metadata['precision_threshold'] = precision_threshold
    metadata['passes'] = passes


    metadata['beta_weighting_param'] = beta_weighting_param # This is not beta threshold

    return metadata


def matching_type_to_str(matching_type):
    data = {MATCHING_TYPE_IOU_MAX:'IOU max',
            MATCHING_TYPE_MAX_FOUND:'Found energy max, iou threshold',
            MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD:'Found energy max, angle threshold',
            MATCHING_TYPE_MAX_PRECISION_ANGLE_THRESHOLD:'Precision max, angle threshold',
            MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD_PRECISION_THRESHOLD:'Found energy max, angle threshold, precision threshold',
            MATCHING_TYPE_IOM_MAX_MULTIPASS: 'IOM multi pass'}
    return data[matching_type]




class OCRecoGraphAnalyzer:
    def __init__(self, metadata):
        self.change_metadata(metadata)

    def change_metadata(self, metadata):
        self.metadata = metadata
        self.with_local_distance_scaling = self.metadata['with_local_distance_scaling']
        self.beta_threshold = self.metadata['beta_threshold']
        self.distance_threshold = self.metadata['distance_threshold']
        self.iou_threshold = self.metadata['iou_threshold']
        self.matching_type = self.metadata['matching_type']
        self.angle_threshold = self.metadata['angle_threshold']
        self.precision_threshold = self.metadata['precision_threshold']
        self.passes = self.metadata['passes']

    def build_truth_graph(self, truth_dict):
        # A disconnected graph with all the nodes with truth information
        truth_graph = nx.Graph()

        truth_sid = truth_dict['truthHitAssignementIdx'][:, 0].astype(np.int32)
        truth_shower_sid, truth_shower_idx = np.unique(truth_sid, return_index=True)

        truth_nodes = []
        for i in range(len(truth_shower_sid)):
            if truth_shower_sid[i] == -1:
                continue

            node_attributes = dict()

            node_attributes['id'] = int(truth_dict['truthHitAssignementIdx'][truth_shower_idx[i], 0])


            node_attributes['x'] = truth_dict['truthHitAssignedX'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedX' in truth_dict  else 0.0

            node_attributes['y'] = truth_dict['truthHitAssignedY'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedY' in truth_dict  else 0.0
            node_attributes['z'] = truth_dict['truthHitAssignedZ'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedZ' in truth_dict  else 0.0
            node_attributes['eta'] = truth_dict['truthHitAssignedEta'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedEta' in truth_dict  else 0.0
            node_attributes['phi'] = truth_dict['truthHitAssignedPhi'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedPhi' in truth_dict  else 0.0
            node_attributes['t'] = truth_dict['truthHitAssignedT'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedT' in truth_dict  else 0.0

            node_attributes['dep_energy'] = np.sum(self.feat_dict['recHitEnergy'][truth_sid==truth_shower_sid[i]])
            node_attributes['energy'] = truth_dict['truthHitAssignedEnergies'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedEnergies' in truth_dict else node_attributes['dep_energy']

            #node_attributes['dep_energy'] = truth_dict['truthHitAssignedDepEnergies'][
            #    truth_shower_idx[i], 0].item()
            node_attributes['pid'] = truth_dict['truthHitAssignedPIDs'][truth_shower_idx[i], 0].item()\
                if 'truthHitAssignedPIDs' in truth_dict  else 0

            node_attributes['type'] = NODE_TYPE_TRUTH_SHOWER

            node = (int(truth_shower_sid[i]), node_attributes)
            truth_nodes.append(node)

        truth_graph.add_nodes_from(truth_nodes)


        return truth_graph, truth_sid

    def build_pred_graph(self, pred_dict, feat_dict):
        # A disconnected graph with all the nodes with the pred showers information
        pred_graph = nx.Graph()

        start_indicing_from = np.max(self.truth_sid) + 1000

        if 'pred_isnoise' in self.pred_dict:
            # Set to something very large so it doesn't come into receptive field of any predicted showers
            self.pred_dict['pred_ccoords'] = np.where(pred_dict['pred_isnoise']!=0, self.pred_dict['pred_ccoords'], self.pred_dict['pred_ccoords']*10000)

        if self.with_local_distance_scaling:
            # print("Doing with pred dist")
            pred_sid, pred_shower_alpha_idx = reconstruct_showers(pred_dict['pred_ccoords'], pred_dict['pred_beta'][:,0], self.beta_threshold, self.distance_threshold, return_alpha_indices=True, limit=1000, pred_dist=pred_dict['pred_dist'][:, 0])
        else:
            pred_sid, pred_shower_alpha_idx = reconstruct_showers(pred_dict['pred_ccoords'], pred_dict['pred_beta'][:,0], self.beta_threshold, self.distance_threshold, return_alpha_indices=True, limit=1000)

        pred_sid += start_indicing_from

        pred_shower_sid = []
        for i in pred_shower_alpha_idx:
            pred_shower_sid.append(pred_sid[i])

        pred_nodes = []
        for i in range(len(pred_shower_sid)):
            if pred_shower_sid[i] == -1:
                raise RuntimeError("Check this")
            sid = int(pred_shower_sid[i])

            node_attributes = dict()

            node_attributes['id']  = sid
            node_attributes['x']  = pred_dict['pred_pos'][pred_shower_alpha_idx[i]][0].item()\
                if 'pred_pos' in pred_dict else 0
            node_attributes['y']  = pred_dict['pred_pos'][pred_shower_alpha_idx[i]][1].item()\
                if 'pred_pos' in pred_dict else 0
            node_attributes['time']  = pred_dict['pred_time'][pred_shower_alpha_idx[i]][0].item()\
                if 'pred_time' in pred_dict else 0
            node_attributes['pid']  = np.argmax(pred_dict['pred_id'][pred_shower_alpha_idx[i]]).item()\
                if 'pred_id' in pred_dict else 0

            node_attributes['dep_energy'] = np.sum(feat_dict['recHitEnergy'][pred_sid==sid]).item()
            node_attributes['energy']  = max(pred_dict['pred_energy'][pred_shower_alpha_idx[i]][0].item(), 0)\
                if 'pred_energy' in pred_dict else node_attributes['dep_energy']

            rechit_energy = feat_dict['recHitEnergy'][pred_sid==sid]
            rechit_x = feat_dict['recHitX'][pred_sid==sid]
            rechit_y = feat_dict['recHitY'][pred_sid==sid]
            rechit_z = feat_dict['recHitZ'][pred_sid==sid]

            node_attributes['dep_energy'] = np.sum(rechit_energy).item()
            node_attributes['dep_x'] = (np.sum(rechit_energy * rechit_x) / np.sum(rechit_energy)).item()
            node_attributes['dep_y'] = (np.sum(rechit_energy * rechit_y) / np.sum(rechit_energy)).item()
            node_attributes['dep_z'] = (np.sum(rechit_energy * rechit_z) / np.sum(rechit_energy)).item()
            node_attributes['type'] = NODE_TYPE_PRED_SHOWER

            node = (sid, node_attributes)
            pred_nodes.append(node)

        pred_graph.add_nodes_from(pred_nodes)

        return pred_graph, pred_sid

    def cost_matrix_intersection_based(self, truth_shower_sid, pred_shower_sid):

        iou_matrix = calculate_iou_tf(self.truth_sid,
                                      self.pred_sid,
                                       truth_shower_sid,
                                       pred_shower_sid,
                                       self.feat_dict['recHitEnergy'][:, 0])

        n = max(len(truth_shower_sid), len(pred_shower_sid))
        # Cost matrix
        C = np.zeros((n, n))
        if self.matching_type == MATCHING_TYPE_IOU_MAX:
            for i in range(len(pred_shower_sid)):
                for j in range(len(truth_shower_sid)):
                    overlap = iou_matrix[i, j]
                    if overlap >= self.iou_threshold:
                        C[i, j] = overlap
        elif self.matching_type == MATCHING_TYPE_MAX_FOUND:
            for i in range(len(pred_shower_sid)):
                for j in range(len(truth_shower_sid)):
                    overlap = iou_matrix[i, j]
                    if overlap >= self.iou_threshold:
                        C[i, j] = min(self.truth_graph.nodes[truth_shower_sid[j]]['energy'], self.pred_graph.nodes[pred_shower_sid[i]]['energy'])
        return C




    def cost_matrix_angle_based(self, truth_shower_sid, pred_shower_sid):
        n = max(len(truth_shower_sid), len(pred_shower_sid))
        C = np.zeros((n, n))

        for a, j in enumerate(pred_shower_sid):
            x = self.pred_graph.nodes(data=True)[j]
            for b, i in enumerate(truth_shower_sid):
                y = self.truth_graph.nodes(data=True)[i]
                if angle(x,y) < self.angle_threshold:
                    if self.matching_type == MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD:
                        C[a, b] = min(x['energy'],
                                      y['energy'])
                    elif self.matching_type == MATCHING_TYPE_MAX_PRECISION_ANGLE_THRESHOLD:
                        C[a, b] = precision_function(x, y, self.angle_threshold) * x['energy']
                    elif self.matching_type == MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD_PRECISION_THRESHOLD:
                        C[a, b] = min(x['energy'],
                                      y['energy']) * (precision_function(x, y, self.angle_threshold) > self.precision_threshold)
        return C

    def _match(self, return_rechit_data=False):
        truth_shower_sid = [x for x in self.truth_graph.nodes()]
        pred_shower_sid = [x for x in self.pred_graph.nodes()]
        if self.matching_type == MATCHING_TYPE_MAX_FOUND or self.matching_type == MATCHING_TYPE_IOU_MAX:
            C = self.cost_matrix_intersection_based(truth_shower_sid, pred_shower_sid)
        elif self.matching_type == MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD or self.matching_type == MATCHING_TYPE_MAX_PRECISION_ANGLE_THRESHOLD or self.matching_type==MATCHING_TYPE_MAX_FOUND_ANGLE_THRESHOLD_PRECISION_THRESHOLD:
            C = self.cost_matrix_angle_based(truth_shower_sid, pred_shower_sid)

        row_id, col_id = linear_sum_assignment(C, maximize=True)

        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.truth_graph.nodes(data=True))
        matched_full_graph.add_nodes_from(self.pred_graph.nodes(data=True))

        for p, t in zip(row_id, col_id):
            if C[p, t] > 0:
                matched_full_graph.add_edge(truth_shower_sid[t], pred_shower_sid[p], attached_in_pass=0)

        if return_rechit_data:
            matched_full_graph = self.attach_rechit_data(matched_full_graph)
        self.non_reduced_graph = matched_full_graph

        return matched_full_graph

    def reduce_truth_nodes(self, truth_nodes):
        if len(truth_nodes) == 0:
            return None
        if len(truth_nodes) == 1:
            return truth_nodes[0]

        node_attributes = dict()

        node_attributes['id'] = truth_nodes[0][0]

        node_attributes['energy'] = np.sum([x[1]['energy'] for x in truth_nodes]).item()
        node_attributes['dep_energy'] = np.sum([x[1]['dep_energy'] for x in truth_nodes]).item()
        node_attributes['x'] = (np.sum([x[1]['x'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['y'] = (np.sum([x[1]['y'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['z'] = (np.sum([x[1]['z'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['t'] = (np.sum([x[1]['t'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['pid'] = truth_nodes[0][1]['pid']
        node_attributes['eta'] = (np.sum([x[1]['eta'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['phi'] = (np.sum([x[1]['phi'] * x[1]['dep_energy'] for x in truth_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['type'] = NODE_TYPE_TRUTH_SHOWER

        return node_attributes['id'], node_attributes

    def reduce_pred_nodes(self, pred_nodes):
        if len(pred_nodes) == 0:
            return None
        if len(pred_nodes) == 1:
            return pred_nodes[0]

        node_attributes = dict()

        node_attributes['id'] = pred_nodes[0][0]

        node_attributes['energy'] = np.sum([x[1]['energy'] for x in pred_nodes]).item()
        node_attributes['dep_energy'] = np.sum([x[1]['dep_energy'] for x in pred_nodes]).item()
        node_attributes['x'] = (
                    np.sum([x[1]['x'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes['dep_energy']).item()
        node_attributes['y'] = (
                    np.sum([x[1]['y'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes['dep_energy']).item()
        node_attributes['time'] = (np.sum([x[1]['time'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['pid'] = pred_nodes[0][1]['pid']

        node_attributes['dep_x'] = (np.sum([x[1]['dep_x'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['dep_y'] = (np.sum([x[1]['dep_y'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['dep_z'] = (np.sum([x[1]['dep_z'] * x[1]['dep_energy'] for x in pred_nodes]) / node_attributes[
            'dep_energy']).item()
        node_attributes['type'] = NODE_TYPE_PRED_SHOWER

        return node_attributes['id'], node_attributes

    def reduce_graph(self, graph):
        pairs = []  # List of all the pairs to which to attach to
        free_nodes = []  # List of all the free nodes (truth or pred)

        connected_components = nx.connected_components(graph)
        for c in connected_components:
            if len(c) > 1:
                pred_c = []
                truth_c = []
                for n in c:
                    data_n = graph.nodes(data=True)[n]
                    if graph.nodes(data=True)[n]['type'] == NODE_TYPE_TRUTH_SHOWER:
                        truth_c.append((n, data_n))
                    else:
                        pred_c.append((n, data_n))
                pairs.append((pred_c, truth_c))
            else:
                free_nodes.append(c.pop())
        reduced_graph = nx.Graph()
        for x in free_nodes:
            reduced_graph.add_nodes_from([(x, graph.nodes(data=True)[x])])

        for x in pairs:
            u = self.reduce_pred_nodes(x[0])
            v = self.reduce_truth_nodes(x[1])
            reduced_graph.add_nodes_from([u, v])
            reduced_graph.add_edge(u[0], v[0])

        return reduced_graph

    def attach_rechit_data(self, g):
        id_max = np.max(g.nodes()) + 1000

        graph_2 = g.copy()

        # Add all the rechit nodes in the graph
        rechit_nodes = []

        rechit_node_id = []

        idd = []
        for i in range(len(self.feat_dict['recHitEnergy'])):
            node = dict()
            node['rechit_energy'] = self.feat_dict['recHitEnergy'][i,0].item()
            node['rechit_x'] = self.feat_dict['recHitX'][i,0].item()
            node['rechit_y'] = self.feat_dict['recHitY'][i,0].item()
            node['rechit_z'] = self.feat_dict['recHitZ'][i,0].item()
            node['type'] = NODE_TYPE_RECHIT
            node['id'] = id_max


            idd.append(id_max)
            rechit_nodes.append((id_max, node))
            rechit_node_id.append(id_max)
            id_max += 1
        rechit_node_id = np.array(rechit_node_id)

        graph_2.add_nodes_from(rechit_nodes)

        for n, att in g.nodes(data=True):
            sid = att['id']
            if att['type'] == NODE_TYPE_TRUTH_SHOWER:
                # Is a truth node
                search_in = self.truth_sid
            elif att['type'] == NODE_TYPE_PRED_SHOWER:
                # Is a pred node
                search_in = self.pred_sid
            else:
                continue

            rechit_nodes = rechit_node_id[search_in == sid]

            for i in rechit_nodes:
                graph_2.add_edge(n, i)

            graph_2.add_nodes_from(rechit_nodes)
        return graph_2

    def _match_multipass(self, return_rechit_data=False):
        truth_shower_sid = [x for x in self.truth_graph.nodes()]
        pred_shower_sid = [x for x in self.pred_graph.nodes()]
        for i, x in enumerate(self.truth_graph.nodes(data=True)):
            x[1]['idx'] = i
        for i, x in enumerate(self.pred_graph.nodes(data=True)):
            x[1]['idx'] = i

        iou_matrix, pred_sum_matrix, truth_sum_matrix, intersection_matrix = calculate_iou_tf(self.truth_sid,
                                                                                              self.pred_sid,
                                                                                              truth_shower_sid,
                                                                                              pred_shower_sid,
                                                                                              self.feat_dict[
                                                                                                  'recHitEnergy'][:, 0],
                                                                                              return_all=True)
        intersection_matrix = intersection_matrix.numpy()
        pred_sum_matrix = pred_sum_matrix.numpy()
        truth_sum_matrix = truth_sum_matrix.numpy()

        min_matrix = intersection_matrix / np.minimum(pred_sum_matrix, truth_sum_matrix)

        n = max(len(truth_shower_sid), len(pred_shower_sid))
        # Cost matrix
        C = np.zeros((n, n))
        if self.matching_type == MATCHING_TYPE_IOM_MAX_MULTIPASS:
            for i in range(len(pred_shower_sid)):
                for j in range(len(truth_shower_sid)):
                    overlap = min_matrix[i, j]
                    if overlap >= self.iou_threshold:
                        C[i, j] = overlap

        row_id, col_id = linear_sum_assignment(C, maximize=True)

        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.truth_graph.nodes(data=True))
        matched_full_graph.add_nodes_from(self.pred_graph.nodes(data=True))

        for p, t in zip(row_id, col_id):
            if C[p, t] > 0:
                matched_full_graph.add_edge(truth_shower_sid[t], pred_shower_sid[p], attached_in_pass=0)

        passes = [x for x in range(self.passes)]
        passes.pop(0)

        graphs = []
        graphs.append(matched_full_graph.copy())
        for npass in passes:
            pairs = []  # List of all the pairs to which to attach to
            free_nodes = []  # List of all the free nodes (truth or pred)

            connected_components = nx.connected_components(matched_full_graph)
            for c in connected_components:
                if len(c) > 1:
                    pred_c = set()
                    truth_c = set()
                    for n in c:
                        if matched_full_graph.nodes(data=True)[n]['type'] == NODE_TYPE_TRUTH_SHOWER:
                            truth_c.add(n)
                        else:
                            pred_c.add(n)
                    pairs.append((pred_c, truth_c))
                else:
                    free_nodes.append(c.pop())

            # print("Pass", npass)
            # print("Free nodes", free_nodes)
            # print("Matches", pairs)

            # Construct another cost matrix
            C = np.zeros((len(pairs), len(free_nodes)))
            for i, p in enumerate(pairs):
                for j, f in enumerate(free_nodes):
                    score = 0
                    type_f = matched_full_graph.nodes(data=True)[f]['type']
                    idx_f = matched_full_graph.nodes(data=True)[f]['idx']
                    matching_mode = -1
                    # Length of either p[0] will be 1 or p[1] will be 1
                    if len(p[0]) == 1:
                        # A pred shower is matched to one or more truth showers
                        if type_f == 0:
                            matching_mode = 0
                        else:
                            pass
                        # sid_p = next(iter(p[0]))
                        # idx_p = matched_full_graph.nodes(data=True)[sid_p]['idx']
                        #
                        # numerator = intersection_matrix[idx_p, 0]
                    if len(p[1]) == 1:
                        # A truth shower is matched to one or more pred showers
                        if type_f == 0:
                            pass
                        else:
                            matching_mode = 1

                        # sid_t = next(iter(p[1]))
                        # idx_t = matched_full_graph.nodes(data=True)[sid_t]['idx']

                    if matching_mode == 0:
                        sid_p = next(iter(p[0]))
                        idx_p = matched_full_graph.nodes(data=True)[sid_p]['idx']
                        numerator = intersection_matrix[idx_p, idx_f]
                        # denominator1 = 0
                        # for x in p[1]:
                        #     denominator1 += intersection_matrix[idx_p, matched_full_graph.nodes(data=True)[x]['idx']]
                        denominator1 = pred_sum_matrix[idx_p, 0]
                        denominator2 = truth_sum_matrix[0, idx_f]
                        score = numerator / min(denominator1, denominator2)
                    elif matching_mode == 1:
                        sid_t = next(iter(p[1]))
                        idx_t = matched_full_graph.nodes(data=True)[sid_t]['idx']
                        numerator = intersection_matrix[idx_f, idx_t]
                        # denominator1 = 0
                        # for x in p[0]:
                        #     denominator1 += intersection_matrix[matched_full_graph.nodes(data=True)[x]['idx'], idx_t]
                        denominator1 = truth_sum_matrix[0, idx_t]
                        denominator2 = pred_sum_matrix[idx_f, 0]
                        score = numerator / min(denominator1, denominator2)
                    if score > self.iou_threshold:
                        C[i, j] = score

            # # Let's match these guys again
            C = np.array(C)
            row_id, col_id = linear_sum_assignment(C, maximize=True)
            for r, c in zip(row_id, col_id):
                if C[r, c] > 0:
                    p = pairs[r]
                    if matched_full_graph.nodes(data=True)[free_nodes[c]]['type'] == NODE_TYPE_TRUTH_SHOWER:
                        # Free node was a truth node
                        matched_full_graph.add_edge(next(iter(p[0])), free_nodes[c], attached_in_pass=npass)
                    else:
                        # Free node was a pred node
                        matched_full_graph.add_edge(next(iter(p[1])), free_nodes[c], attached_in_pass=npass)
            graphs.append(matched_full_graph.copy())

        self.non_reduced_graph = matched_full_graph
        self.non_reduced_graph = self.attach_rechit_data(self.non_reduced_graph)

        graphs_reduced = [self.reduce_graph(g) for g in graphs]
        main_graph = self.reduce_graph(matched_full_graph)

        graphs_reduced_2 = []
        if return_rechit_data:
            main_graph = self.attach_rechit_data(main_graph)
            for g in graphs_reduced:
                graphs_reduced_2 += [self.attach_rechit_data(g)]


        return main_graph

    def match(self, return_rechit_data=False):
        if self.matching_type != MATCHING_TYPE_IOM_MAX_MULTIPASS:
            return self._match(return_rechit_data)
        else:
            return self._match_multipass(return_rechit_data)

    def analyse(self, feat_dict, pred_dict, truth_dict, return_rechit_data=False):
        self.feat_dict = feat_dict
        self.pred_dict = pred_dict
        self.truth_dict = truth_dict

        truth_graph, truth_sid = self.build_truth_graph(truth_dict)
        self.truth_graph = truth_graph
        self.truth_sid = truth_sid
        pred_graph, pred_sid = self.build_pred_graph(pred_dict, feat_dict)
        self.pred_graph = pred_graph
        self.pred_sid = pred_sid

        return self.match(return_rechit_data)


class OCMatchingVisualizer():
    def draw(self, fig, ax, graph, title):
        fig.clear()
        # ax.clear()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(title)

        showers = [x for x, y in graph.nodes(data=True) if y['type'] == 0 or y['type'] == 1]

        X = []
        Y = []
        Z = []
        E = []
        C = []
        for s in showers:
            if graph.nodes[s]['visible'] == False:
                continue

            recs = list(graph.neighbors(s))
            x = [graph.nodes[x]['rechit_x'] for x in recs]
            y = [graph.nodes[x]['rechit_y'] for x in recs]
            z = [graph.nodes[x]['rechit_z'] for x in recs]
            e = [graph.nodes[x]['rechit_energy'] for x in recs]
            c = [self.cmap(graph.nodes[s]['secondary_color_id'])] * len(e)



            X += x
            Y += y
            Z += z
            C += c
            E += e

        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')

        E = np.array(E)

        S = 10 * np.power(E, (3./2.))

        ax.scatter(Z, X, Y, s=S, c=C)
        # plt.show()

    def click_event(self, event):
        ix, iy = event.xdata, event.ydata

        if ix != None:
            pos_clicked = np.array([ix,iy])[np.newaxis, :]

            delta = np.sum((self.node_positions_positions - pos_clicked)**2, axis=-1)
            closest = self.node_positions_ids[np.argmin(delta)]
            # Flip visibility
            self.graph_showers_only.nodes[closest]['visible'] = not self.graph_showers_only.nodes[closest]['visible']
        self.update_network(clear=True)

        # node_ids = self.node_positions[]

    def find_center_node_of_star_graph(self, full_graph, star_sub_graph):
        star_sub_graph = list(star_sub_graph)
        # Could do a test if its a star graph here
        neighbors = [len([x for x in full_graph.neighbors(n)]) for n in star_sub_graph]
        return star_sub_graph[np.argmax(neighbors)]

    def update_network(self, clear=False):
        # self.ax.clear()
        self.figure_graph.clear()
        self.ax = self.figure_graph.add_axes([0, 0, 1, 1])

        edges_removed_graph = self.graph_showers_only.copy()
        for node1, node2, data in self.graph_showers_only.edges(data=True):
            attached_in_pass = data['attached_in_pass']
            if attached_in_pass >= self.number_of_passes:
                edges_removed_graph.remove_edge(node1, node2)
        connected_components = list(nx.connected_components(edges_removed_graph))
        for c in connected_components:
            center_node = self.find_center_node_of_star_graph(self.graph_showers_only, c)
            for node in c:
                self.graph_showers_only.nodes[node]['secondary_color_id'] = self.graph_showers_only.nodes[center_node]['color_id']


        self.figure_graph.canvas.draw()
        for node, position in self.node_positions.items():
            node_data = self.graph_showers_only.nodes[node]
            # print(node, position, node_data)
            if node_data['visible']:
                alpha = 1
            else:
                alpha = 0.5

            if node_data['type'] == NODE_TYPE_TRUTH_SHOWER:
                shape = 'o'
                color=self.cmap(node_data['secondary_color_id'])
            else:
                shape = 's'
                color=self.cmap(node_data['secondary_color_id'])
            # print(alpha)
            nx.draw_networkx_nodes(self.graph_showers_only, {node:position}, node_shape=shape, alpha=alpha, nodelist=[node], node_color=color, ax=self.ax)

        # nx.draw_networkx_edges(edges_removed_graph, self.node_positions, ax=self.ax)
        self.figure_graph.canvas.draw()

        graph_truth_and_hits_induced = self.graph_data.copy()
        graph_pred_and_hits_induced = self.graph_data.copy()

        for node,att in self.graph_showers_only.nodes(data=True):
            if att['type'] == NODE_TYPE_TRUTH_SHOWER:
                graph_pred_and_hits_induced.remove_node(node)
            if att['type'] == NODE_TYPE_PRED_SHOWER:
                graph_truth_and_hits_induced.remove_node(node)

        graph_pred_and_hits_induced.remove_nodes_from(list(nx.isolates(graph_pred_and_hits_induced)))
        graph_truth_and_hits_induced.remove_nodes_from(list(nx.isolates(graph_truth_and_hits_induced)))

        self.draw(self.fig_3d_truth, self.ax_3d_truth, graph_truth_and_hits_induced, title='Truth')
        self.draw(self.fig_3d_pred, self.ax_3d_pred, graph_pred_and_hits_induced, title='Pred')

        self.fig_3d_truth.canvas.draw()
        self.fig_3d_pred.canvas.draw()

    def update_num_passes_slider(self, val):
        self.number_of_passes = self.slider_num_passes.val
        self.update_network()

    def draw_network(self):
        network = self.graph_showers_only
        self.node_positions = nx.spring_layout(network)
        self.node_positions_ids = []
        self.node_positions_positions = []
        for k, v in self.node_positions.items():
            self.node_positions_ids.append(k)
            self.node_positions_positions.append(v)
        self.node_positions_positions = np.array(self.node_positions_positions)

        self.figure_graph, _ = plt.subplots()
        cid = self.figure_graph.canvas.mpl_connect('button_press_event', self.click_event)

        self.figure_slider_num_passes = plt.figure(figsize=(5,1))


        self.slider_num_passes = Slider(plt.axes([0.2, 0.2, 0.7, 0.3]), 'Num passes', valmin=0, valmax=10, valstep=1, valinit=self.number_of_passes)
        self.slider_num_passes.on_changed(self.update_num_passes_slider)
        self.fig_3d_truth = plt.figure()
        self.ax_3d_truth = self.fig_3d_truth.add_subplot(projection='3d')
        self.ax_3d_truth.set_title('Truth 3d')

        self.fig_3d_pred = plt.figure()
        self.ax_3d_pred = self.fig_3d_pred.add_subplot(projection='3d')
        self.ax_3d_pred.set_title('Pred 3d')

        self.update_network()

        plt.show()



    def collect_showers(self):
        graph = self.graph_data
        # graph_showers_only = graph.copy()
        self.max_color_id = 0

        nodes_showers_only = []
        for n, att in graph.nodes(data=True):
            if att['type'] == NODE_TYPE_TRUTH_SHOWER or att['type'] == NODE_TYPE_PRED_SHOWER:
                nodes_showers_only.append(n)

        graph_showers_only = graph.subgraph(nodes_showers_only)
        for n, att in graph_showers_only.nodes(data=True):
            graph_showers_only.nodes[n]['visible']=True
            graph_showers_only.nodes[n]['color_id'] = self.max_color_id
            self.max_color_id += 1

        self.graph_showers_only = graph_showers_only
        self.cmap = plt.cm.get_cmap("prism", self.max_color_id)
        self.number_of_passes = 10

    def __init__(self, graph_data):
        self.graph_data = graph_data
        self.collect_showers()

    def visualize(self):
        # self.draw_calo(self.graph_data)
        self.draw_network()


class OCAnlayzerWrapper():
    def __init__(self, metadata):
        self.metadata=metadata
        self.graph_analyzer = OCRecoGraphAnalyzer(metadata)

    def _add_metadata(self, analysed_graphs):
        metadata = self.metadata.copy()
        _, _, _, percentage_pred_matched, percentage_truth_matched, _ = scalar_metrics.compute_scalar_metrics_graph(analysed_graphs, beta=metadata['beta_weighting_param'])
        precision_value, absroption_value = scalar_metrics.compute_precision_and_absorption_graph(analysed_graphs, metadata)

        if precision_value ==0 or absroption_value ==0:
            reco_score = 0
        else:
            reco_score = 2 * precision_value * absroption_value / (precision_value + absroption_value)

        metadata['reco_score'] = reco_score
        metadata['pred_energy_percentage_matched'] = percentage_pred_matched
        metadata['truth_energy_percentage_matched'] = percentage_truth_matched
        metadata['precision_value'] = precision_value
        metadata['absorption_value'] = absroption_value
        metadata['matching_type_str'] = matching_type_to_str(metadata['matching_type'])
        metadata['angle_threshold'] = metadata['angle_threshold']

        efficiency, fake_rate, response_mean, response_sum_mean = scalar_metrics.compute_scalar_metrics_graph_eff_fake_rate_response(analysed_graphs)

        metadata['efficiency'] = efficiency
        metadata['fake_rate'] = fake_rate
        metadata['response_mean'] = response_mean
        metadata['response_sum_mean'] = response_sum_mean

        num_truth_showers, num_pred_showers = scalar_metrics.compute_num_showers(analysed_graphs)

        metadata['num_truth_showers'] = num_truth_showers
        metadata['num_pred_showers'] = num_pred_showers

        return metadata

    def analyse_from_files(self, files_to_be_tested):
        # all_data = []
        analysed_graphs = []
        for i, file in enumerate(files_to_be_tested):
            print("Analysing file", i)
            with gzip.open(file, 'rb') as f:
                file_data = pickle.load(f)
                # all_data.append(data_loaded)
                for j, endcap_data in enumerate(file_data):
                    print("\tAnalysing Endcap", j)
                    x = self.graph_analyzer.analyse(endcap_data[0], endcap_data[2], endcap_data[1])
                    analysed_graphs.append(x)

        metadata = self._add_metadata(analysed_graphs)
        return analysed_graphs, metadata

    def analyse_single_endcap(self, feat_dict, truth_dict, pred_dict):
        analysed_graphs = []
        x = self.graph_analyzer.analyse(feat_dict, pred_dict, truth_dict)
        analysed_graphs.append(x)
        metadata = self._add_metadata(analysed_graphs)
        return analysed_graphs, metadata


    def analyse_from_data(self, data, beta_threshold=-1, distance_threshold=-1, limit_endcaps=-1):
        """
        This function is used in hyper param optimizer potentially so it gives an option to override beta threshold and distance threshold.
        Leave -1 for normal functioning otherwise change them both together.

        :return:
        """
        metadata = self.metadata
        if beta_threshold !=-1:
            metadata['beta_threshold'] = beta_threshold
            metadata['distance_threshold'] = distance_threshold
        self.graph_analyzer.change_metadata(metadata)
        analysed_graphs = []
        done=False
        nendcaps_done = 0
        for i, file_data in enumerate(data):
            # print("Analysing file", i)
            for j, endcap_data in enumerate(file_data):
                # print("\tAnalysing Endcap", j)
                x = self.graph_analyzer.analyse(endcap_data[0], endcap_data[2], endcap_data[1])
                analysed_graphs.append(x)
                nendcaps_done += 1
                if nendcaps_done == limit_endcaps and limit_endcaps != -1:
                    done = True
                    break

            if done:
                break

        metadata = self._add_metadata(analysed_graphs)
        return analysed_graphs, metadata

