import numpy as np
import networkx as nx
import tensorflow as tf


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


def calculate_eiou(truth_sid,
                   pred_sid,
                   truth_shower_sid,
                   pred_shower_sid,
                   hit_weight,
                   iou_threshold):
    truth_sid = tf.cast(tf.convert_to_tensor(truth_sid), tf.int32)
    pred_sid = tf.cast(tf.convert_to_tensor(pred_sid), tf.int32)
    hit_weight = tf.cast(tf.convert_to_tensor(hit_weight), tf.float32)

    truth_shower_sid = tf.convert_to_tensor(truth_shower_sid)
    pred_shower_sid = tf.convert_to_tensor(pred_shower_sid)
    len_pred_showers = len(pred_shower_sid)
    len_truth_showers = len(truth_shower_sid)

    truth_idx_2 = tf.zeros_like(truth_sid)
    pred_idx_2 = tf.zeros_like(pred_sid)

    for i in range(len(pred_shower_sid)):
        pred_idx_2 = tf.where(pred_sid == pred_shower_sid[i], i, pred_idx_2)

    for i in range(len(truth_shower_sid)):
        truth_idx_2 = tf.where(truth_sid == truth_shower_sid[i], i, truth_idx_2)

    one_hot_pred = tf.one_hot(pred_idx_2, depth=len_pred_showers)
    one_hot_truth = tf.one_hot(truth_idx_2, depth=len_truth_showers)

    intersection_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)

    pred_sum_matrix = tf.linalg.matmul(one_hot_pred * hit_weight[..., tf.newaxis], tf.ones_like(one_hot_truth),
                                    transpose_a=True)

    truth_sum_matrix = tf.linalg.matmul(
        tf.ones_like(one_hot_pred) * hit_weight[..., tf.newaxis], one_hot_truth, transpose_a=True)

    union_sum_matrix = pred_sum_matrix + truth_sum_matrix - intersection_sum_matrix


    overlap_matrix = (intersection_sum_matrix / union_sum_matrix).numpy()
    pred_shower_sid = pred_shower_sid.numpy()
    truth_shower_sid = truth_shower_sid.numpy()


    all_iou = []
    for i in range(len_pred_showers):
        for j in range(len_truth_showers):
            overlap = overlap_matrix[i, j]

            if overlap > iou_threshold:
                if pred_shower_sid[i] == -1 or truth_shower_sid[j] == -1:
                    continue
                all_iou.append((pred_shower_sid[i], truth_shower_sid[j], overlap))
    return all_iou, overlap_matrix, pred_sum_matrix.numpy(), truth_sum_matrix.numpy(), intersection_sum_matrix.numpy()


def reconstruct_showers(cc, beta, beta_threshold=0.5, dist_threshold=0.5, limit=500, return_alpha_indices=False, pred_dist=None, max_hits_per_shower=-1):
    # print(beta.shape, cc.shape, type(beta), type(cc))
    
    if pred_dist is None:
        pred_dist = np.ones_like(beta)
        
    #pred_dist = pred_dist[:,0]
    #beta = beta[:,0]
    
    beta_filtered_indices = np.argwhere(beta>beta_threshold)
    beta_filtered = np.array(beta[beta_filtered_indices])
    beta_filtered_remaining = beta_filtered.copy()
    cc_beta_filtered = np.array(cc[beta_filtered_indices])
    pred_sid = beta*0 - 1
    pred_sid = pred_sid.astype(np.int32)
    
        

    max_index = 0
    alpha_indices = []

    while np.sum(beta_filtered_remaining) > 0:
        alpha_index = beta_filtered_indices[np.argmax(beta_filtered_remaining)]
        cc_alpha = cc[alpha_index]
        # print(cc[alpha_index].shape, cc.shape)
        dists = np.sum((cc - cc_alpha)**2, axis=-1)

        if max_hits_per_shower != -1:
            raise NotImplementedError("Error")
        pred_sid[np.logical_and(dists < (pred_dist * dist_threshold), pred_sid == -1)] = max_index

        max_index += 1

        dists_filtered = np.sum((cc_alpha - cc_beta_filtered)**2, axis=-1)
        beta_filtered_remaining[dists_filtered < dist_threshold] = 0

        alpha_indices.append(alpha_index[0])

    if return_alpha_indices:
        return pred_sid, np.array(alpha_indices)
    else:
        return pred_sid
    
def _tbi_reconstruct_showers(cc, beta, beta_threshold=0.5, dist_threshold=0.5, 
                        limit=500, return_alpha_indices=False, pred_dist=None, 
                        max_hits_per_shower=-1):
    
    from assign_condensate_op import BuildAndAssignCondensates

    asso, iscond, _ = BuildAndAssignCondensates(
        tf.convert_to_tensor(tf.convert_to_tensor(cc)),
        tf.convert_to_tensor(tf.convert_to_tensor(beta[..., np.newaxis])),
        row_splits=tf.convert_to_tensor(np.array([0, len(cc)], np.int32)),
        dist=tf.convert_to_tensor(pred_dist[..., np.newaxis]) if pred_dist is not None else None,
        min_beta=beta_threshold,
        radius=dist_threshold)

    asso = asso.numpy().tolist()
    iscond = iscond.numpy()

    pred_shower_alpha_idx = np.argwhere(iscond==1)[:, 0].tolist()

    map_fn = {x:i for i,x in enumerate(pred_shower_alpha_idx)}
    map_fn[-1] = -1
    pred_sid = [map_fn[x] for x in asso]

    if return_alpha_indices:
        return pred_sid, pred_shower_alpha_idx

    return pred_sid


def match(truth_sid, pred_sid, energy, iou_threshold=0.1):
    truth_sid = truth_sid.astype(np.int32)
    pred_sid = pred_sid.astype(np.int32)

    truth_shower_sid, truth_shower_sid_idx = np.unique(truth_sid, return_index=True)
    pred_shower_sid, pred_shower_sid_idx = np.unique(pred_sid, return_index=True)
    truth_shower_sid = truth_shower_sid[truth_shower_sid>-1]
    pred_shower_sid = pred_shower_sid[pred_shower_sid>-1]


    all_iou, iou_matrix, pred_sum_matrix, truth_sum_matrix, intersection_matrix = calculate_eiou(truth_sid,
                                                                                                 pred_sid,
                                                                                                 truth_shower_sid,
                                                                                                 pred_shower_sid,
                                                                                                 energy,
                                                                                                 iou_threshold)

    G = nx.Graph()

    for iou in all_iou:
        # print(iou)
        G.add_edge('p%d' % iou[0], 't%d' % iou[1], weight=iou[2])

    X = nx.algorithms.max_weight_matching(G)

    pred_shower_sid_to_pred_shower_idx = {}

    truth_shower_sid = np.unique(truth_sid)
    pred_shower_sid_to_truth_shower_sid = {}
    for i, x in enumerate(pred_shower_sid):
        pred_shower_sid_to_truth_shower_sid[x] = -1
        pred_shower_sid_to_pred_shower_idx[x] = i

    for x, y in X:
        if x[0] == 'p':
            prediction_index = int(x[1:])
            truth_index = int(y[1:])
        else:
            truth_index = int(x[1:])
            prediction_index = int(y[1:])

        pred_shower_sid_to_truth_shower_sid[prediction_index] = truth_index

    new_indicing = np.max(truth_shower_sid) + 1
    pred_sid_2 = np.zeros_like(pred_sid, np.int32) - 1

    pred_shower_sid_2 = []

    num_total_predicted = 0
    num_fakes_predicted = 0
    for k in pred_shower_sid:
        num_total_predicted += 1
        v = pred_shower_sid_to_truth_shower_sid[k]
        # num_total_showers += 1 if obc else 0
        if v != -1:
            pred_sid_2[pred_sid == k] = v
            pred_shower_sid_2.append(v)
        else:
            num_fakes_predicted += 1
            pred_sid_2[pred_sid == k] = new_indicing
            pred_shower_sid_2.append(new_indicing)
            new_indicing += 1

    # 0/0
    return pred_sid_2



def compute_efficiency_and_fake_rate(pred_sid, truth_sid):
    truth_shower_sid = np.unique(truth_sid[truth_sid>=0])
    pred_shower_sid = np.unique(pred_sid[pred_sid>=0])

    missed = np.setdiff1d(truth_shower_sid, pred_shower_sid)
    eff = 1 - float(len(missed))/len(truth_shower_sid)

    fakes = len(np.setdiff1d(pred_shower_sid, truth_shower_sid))
    fake_rate = float(fakes)/len(pred_shower_sid) if len(pred_shower_sid>0) else -1

    return eff, fake_rate



def compute_response_mean(pred_sid, truth_sid, rechit_energy, truth_energy, pred_energy, beta):
    T, idx = np.unique(truth_sid, return_index=True)
    idx = idx[T>=0]
    T = T[T>=0]

    truth_shower_energy = truth_energy[idx]


    same_shower = (truth_sid[..., np.newaxis] == T[np.newaxis, ...])
    truth_shower_sum = np.sum(same_shower * rechit_energy[..., np.newaxis], axis=0)

    same_shower = (pred_sid[..., np.newaxis] == T[np.newaxis, ...])
    found = np.sum(same_shower, axis=0) > 0
    pred_energy_sum = np.sum(same_shower * rechit_energy[..., np.newaxis], axis=0)
    alpha_indices = found * np.argmax((beta[..., tf.newaxis]+0.1) * same_shower, axis=0) + (1-found) * (-1)

    response_mean = np.mean(pred_energy[alpha_indices[found]]/truth_shower_energy[found]).item()
    response_sum_mean = np.mean(pred_energy_sum[found] / truth_shower_sum[found]).item()

    if np.sum(found)==0 or len(truth_shower_energy)==0:
        response_sum_mean, response_mean = -1, -1


    if np.any(truth_shower_energy==0.):
        raise ZeroDivisionError('Truth energy for some shower is zero. Check!')

    return response_mean, response_sum_mean


