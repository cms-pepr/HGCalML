import numpy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pickle
from graph_functions import *
import networkx as nx
import gzip
import scalar_metrics


MATCHING_TYPE_IOU_MAX = 0
MATCHING_TYPE_MAX_FOUND = 1


def get_truth_matched_attribute(graphs_list, attribute_name_truth, attribute_name_pred, numpy=False, not_found_value=-1):
    truth_data = []
    pred_data = []
    for g in graphs_list:
        for n, att in g.nodes(data=True):
            if att['type'] == 0:
                matched = [x for x in g.neighbors(n)]
                if len(matched)==0:
                    pred_data.append(None)
                elif len(matched)==1:
                    pred_data.append( g.nodes(data=True)[matched[0]][attribute_name_pred])
                else:
                    raise RuntimeError("Truth shower matched to multiple pred showers?")
                truth_data.append(att[attribute_name_truth])
    if numpy:
        truth_data = np.array(truth_data)
        pred_data = [x if x is not None else not_found_value for x in pred_data]
        pred_data =np.array(pred_data)

    return truth_data, pred_data



def get_pred_matched_attribute(graphs_list, attribute_name_truth, attribute_name_pred, numpy=False, not_found_value=-1):
    pred_data = []
    truth_data = []
    for g in graphs_list:
        for n, att in g.nodes(data=True):
            if att['type'] == 1:
                matched = [x for x in g.neighbors(n)]
                if len(matched)==0:
                    truth_data.append(None)
                elif len(matched)==1:
                    truth_data.append(g.nodes(data=True)[matched[0]][attribute_name_pred])
                else:
                    raise RuntimeError("Pred shower matched to multiple truth showers?")

                pred_data.append(att[attribute_name_truth])
    if numpy:
        pred_data = np.array(pred_data)
        truth_data = [x if x is not None else not_found_value for x in truth_data]
        truth_data =np.array(truth_data)

    return pred_data, truth_data



def build_metadeta_dict(beta_threshold=0.5, distance_threshold=0.5, iou_threshold=0.0001, matching_type=MATCHING_TYPE_MAX_FOUND,
                        with_local_distance_scaling=False, beta_weighting_param=1):
    metadata = dict()
    metadata['beta_threshold'] = beta_threshold
    metadata['distance_threshold'] = distance_threshold
    metadata['iou_threshold'] = iou_threshold
    metadata['matching_type'] = matching_type
    metadata['with_local_distance_scaling'] = with_local_distance_scaling

    metadata['reco_score'] = -1
    metadata['pred_energy_matched'] = -1
    metadata['truth_energy_matched'] = -1


    metadata['beta_weighting_param'] = beta_weighting_param # This is not beta threshold

    return metadata


def matching_type_to_str(matching_type):
    data = {MATCHING_TYPE_IOU_MAX:'IOU max',
            MATCHING_TYPE_MAX_FOUND:'Found energy max'}
    return data[matching_type]




class OCRecoGraphAnalyzer:
    def __init__(self, metadata):

        self.metadata = metadata
        self.with_local_distance_scaling = self.metadata['with_local_distance_scaling']
        self.beta_threshold = self.metadata['beta_threshold']
        self.distance_threshold = self.metadata['distance_threshold']
        self.iou_threshold = self.metadata['iou_threshold']
        self.matching_type = self.metadata['matching_type']

    def change_metadata(self, metadata):
        self.metadata = metadata
        self.with_local_distance_scaling = self.metadata['with_local_distance_scaling']
        self.beta_threshold = self.metadata['beta_threshold']
        self.distance_threshold = self.metadata['distance_threshold']
        self.iou_threshold = self.metadata['iou_threshold']
        self.matching_type = self.metadata['matching_type']

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
            node_attributes['energy'] = truth_dict['truthHitAssignedEnergies'][truth_shower_idx[i], 0].item()
            node_attributes['x'] = truth_dict['truthHitAssignedX'][truth_shower_idx[i], 0].item()
            node_attributes['y'] = truth_dict['truthHitAssignedY'][truth_shower_idx[i], 0].item()
            node_attributes['z'] = truth_dict['truthHitAssignedZ'][truth_shower_idx[i], 0].item()
            node_attributes['eta'] = truth_dict['truthHitAssignedEta'][truth_shower_idx[i], 0].item()
            node_attributes['phi'] = truth_dict['truthHitAssignedPhi'][truth_shower_idx[i], 0].item()
            node_attributes['t'] = truth_dict['truthHitAssignedT'][truth_shower_idx[i], 0].item()
            node_attributes['dep_energy'] = truth_dict['truthHitAssignedDepEnergies'][
                truth_shower_idx[i], 0].item()
            node_attributes['pid'] = truth_dict['truthHitAssignedPIDs'][truth_shower_idx[i], 0].item()

            node_attributes['type'] = 0

            node = (int(truth_shower_sid[i]), node_attributes)
            truth_nodes.append(node)

        truth_graph.add_nodes_from(truth_nodes)


        return truth_graph, truth_sid

    def build_pred_graph(self, pred_dict, feat_dict):
        # A disconnected graph with all the nodes with the pred showers information
        pred_graph = nx.Graph()

        start_indicing_from = np.max(self.truth_sid) + 1000

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
            node_attributes['energy']  = pred_dict['pred_energy'][pred_shower_alpha_idx[i]][0].item()
            node_attributes['x']  = pred_dict['pred_pos'][pred_shower_alpha_idx[i]][0].item()
            node_attributes['y']  = pred_dict['pred_pos'][pred_shower_alpha_idx[i]][1].item()
            node_attributes['time']  = pred_dict['pred_time'][pred_shower_alpha_idx[i]][0].item()
            node_attributes['pid']  = np.argmax(pred_dict['pred_id'][pred_shower_alpha_idx[i]]).item()

            node_attributes['dep_energy'] = np.sum(feat_dict['recHitEnergy'][pred_sid==sid]).item()

            rechit_energy = feat_dict['recHitEnergy'][pred_sid==sid]
            rechit_x = feat_dict['recHitX'][pred_sid==sid]
            rechit_y = feat_dict['recHitY'][pred_sid==sid]
            rechit_z = feat_dict['recHitZ'][pred_sid==sid]

            node_attributes['dep_energy'] = np.sum(rechit_energy).item()
            node_attributes['dep_x'] = (np.sum(rechit_energy * rechit_x) / np.sum(rechit_energy)).item()
            node_attributes['dep_y'] = (np.sum(rechit_energy * rechit_y) / np.sum(rechit_energy)).item()
            node_attributes['dep_z'] = (np.sum(rechit_energy * rechit_z) / np.sum(rechit_energy)).item()
            node_attributes['type'] = 1

            node = (sid, node_attributes)
            pred_nodes.append(node)

        pred_graph.add_nodes_from(pred_nodes)

        return pred_graph, pred_sid

    def match(self):
        truth_shower_sid = [x for x in self.truth_graph.nodes()]
        pred_shower_sid = [x for x in self.pred_graph.nodes()]

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
        row_id, col_id = linear_sum_assignment(C, maximize=True)


        matched_full_graph = nx.Graph()
        matched_full_graph.add_nodes_from(self.truth_graph.nodes(data=True))
        matched_full_graph.add_nodes_from(self.pred_graph.nodes(data=True))

        for p, t in zip(row_id, col_id):
            if C[p, t] > 0:
                matched_full_graph.add_edge(truth_shower_sid[t], pred_shower_sid[p])

        return matched_full_graph

    def analyse(self, feat_dict, pred_dict, truth_dict):
        self.feat_dict = feat_dict
        self.pred_dict = pred_dict
        self.truth_dict = truth_dict

        truth_graph, truth_sid = self.build_truth_graph(truth_dict)
        self.truth_graph = truth_graph
        self.truth_sid = truth_sid
        pred_graph, pred_sid = self.build_pred_graph(pred_dict, feat_dict)
        self.pred_graph = pred_graph
        self.pred_sid = pred_sid

        return self.match()



class OCAnlayzerWrapper():
    def __init__(self, metadata):
        self.metadata=metadata
        self.graph_analyzer = OCRecoGraphAnalyzer(metadata)

    def _add_metadata(self, analysed_graphs):
        metadata = self.metadata.copy()
        precision, recall, f_score, precision_energy, recall_energy, f_score_energy = scalar_metrics.compute_scalar_metrics_graph(analysed_graphs, beta=metadata['beta_weighting_param'])
        metadata['reco_score'] = f_score_energy
        metadata['pred_energy_percentage_matched'] = precision_energy
        metadata['truth_energy_percentage_matched'] = recall_energy
        metadata['matching_type_str'] = matching_type_to_str(metadata['matching_type'])

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
        all_data = []
        for file in files_to_be_tested:
            with gzip.open(file, 'rb') as f:
                data_loaded = pickle.load(f)
                all_data.append(data_loaded)

        analysed_graphs = []
        for i, file_data in enumerate(all_data):
            print("Analysing file", i)
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


