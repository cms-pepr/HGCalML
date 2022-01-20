raise NotImplementedError('Needs to be revamped with the new code. To be done soon')
import numpy as np
import tensorflow as tf
from importlib import reload
import matplotlib.pyplot as plt
import matching_and_analysis


def w_func(x, alpha):
    return np.power(x, 1-alpha) if alpha >= 0 else 1-np.power(1-x, 1-abs(alpha))

"""
Precision = found / found + fake
Reall = found / found + missed

"""

def compute_precision_and_recall_analytic(energy_truth, energy_matched_to_truth, energy_predicted, energy_matched_to_predicted, alpha=0., beta=1., return_all_dict=False, prevent_norm=False):
    upper_limit = 200

    energy_matched_to_truth = np.array(energy_matched_to_truth)
    # print(energy_matched_to_truth.shape)
    energy_truth = np.array(energy_truth)
    # print(energy_truth.shape)

    if not prevent_norm:
        energy_matched_to_truth = energy_matched_to_truth[energy_truth<upper_limit]
        energy_truth = energy_truth[energy_truth< upper_limit]
    energy_matched_to_truth_fixed = np.maximum(energy_matched_to_truth, 0)


    energy_matched_to_predicted = np.array(energy_matched_to_predicted)
    energy_predicted = np.array(energy_predicted)
    if len(energy_predicted.shape)==2:
        energy_predicted = energy_predicted[:, 0]

    if not prevent_norm:
        energy_matched_to_predicted = energy_matched_to_predicted[energy_predicted < upper_limit]
        energy_predicted = energy_predicted[energy_predicted < upper_limit]
    energy_predicted = np.maximum(energy_predicted, 0)

    found_mask = energy_matched_to_truth!=-1
    missed_mask = energy_matched_to_truth==-1
    fake_mask = energy_matched_to_predicted==-1

    wt = w_func(energy_truth/upper_limit, alpha) * float(not prevent_norm) + 1 * float(prevent_norm)
    wp = w_func(energy_predicted/upper_limit, alpha) * float(not prevent_norm) + 1 * float(prevent_norm)

    found_reduced = np.sum(found_mask * wt) / np.sum(wt)
    fake_reduced = np.sum(fake_mask * wp) / np.sum(wp)
    missed_reduced = np.sum(missed_mask * wt) / np.sum(wt)

    precision = found_reduced / (found_reduced + fake_reduced)
    recall = found_reduced / (found_reduced + missed_reduced)
    f_score = (1+beta**2)  * precision * recall / ((beta**2 * precision)+recall)

    found_energy_reduced = np.sum(found_mask * (energy_truth - np.maximum(energy_truth - energy_matched_to_truth_fixed, 0)) * wt)

    # Over predicted
    fake_1 = np.sum(wt * found_mask * np.maximum(energy_matched_to_truth_fixed - energy_truth, 0))
    # Missed pred
    fake_2 = np.sum(fake_mask * energy_predicted * wp)
    # fake_energy_reduced = (np.sum(fake_1) + np.sum(fake_2)) / (np.sum(np.concatenate((wt*found_mask, wp*fake_mask), axis=0)))
    fake_energy_reduced = fake_1 + fake_2

    # missed is underpredicted and unmatched truth
    missed_energy_reduced = np.sum(found_mask * wt * np.maximum(energy_truth - energy_matched_to_truth_fixed, 0) + missed_mask * wt * energy_truth)

    precision_energy = found_energy_reduced / (found_energy_reduced + fake_energy_reduced)
    recall_energy = found_energy_reduced / (found_energy_reduced + missed_energy_reduced)
    f_score_energy = (1+beta**2)  * precision_energy * recall_energy / ((beta**2 * precision_energy)+recall_energy)

    if return_all_dict:
        return precision, recall, f_score, precision_energy, recall_energy, f_score_energy

    return f_score, f_score_energy

def compute_precision_and_recall(energy_truth, energy_matched_to_truth, energy_predicted, energy_matched_to_predicted):

    # print(np.min(energy_truth), np.max(energy_truth))
    # print(np.min(energy_matched_to_truth), np.max(energy_matched_to_truth))
    # print(np.min(energy_predicted), np.max(energy_predicted))
    # print(np.min(energy_matched_to_predicted), np.max(energy_matched_to_predicted))

    e_bins = [0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120,140,160,180,200]

    energy_truth = np.array(energy_truth)
    energy_matched_to_truth = np.array(energy_matched_to_truth)

    energy_predicted = np.array(energy_predicted)
    energy_matched_to_predicted = np.array(energy_matched_to_predicted)


    probability_values = np.histogram(energy_truth, bins=e_bins)[0]
    # probability_values = probability_values / np.sum(probability_values)
    precision_values = []
    recall_values = []
    precision_energy_values = []
    recall_energy_values = []

    mean_energy = []


    # print("Starting")
    for i in range(len(e_bins) - 1):
        l = e_bins[i]
        h = e_bins[i + 1]
        # print(h, l)

        filter = np.argwhere(np.logical_and(energy_truth > l, energy_truth < h))
        filtered_truth_energy = energy_truth[filter].astype(np.float)
        filtered_matched_to_truth_energy = energy_matched_to_truth[filter]

        # second_filter = filtered_matched_to_truth_energy >= 0
        # filtered_truth_energy_2 = filtered_truth_energy[second_filter][:, np.newaxis]
        # filtered_matched_to_truth_energy_2 = filtered_matched_to_truth_energy[second_filter][:, np.newaxis]


        filter = np.argwhere(np.logical_and(energy_predicted > l, energy_predicted < h))
        filtered_predicted_energy = energy_predicted[filter].astype(np.float)
        filtered_matched_to_predicted_energy = energy_matched_to_predicted[filter].astype(np.float)


        found = float(len(filtered_matched_to_truth_energy[filtered_matched_to_truth_energy>=0]))
        fake = float(len(filtered_matched_to_predicted_energy[filtered_matched_to_predicted_energy<0]))
        missed = float(len(filtered_matched_to_truth_energy[filtered_matched_to_truth_energy<0]))

        filtered_matched_to_truth_energy_2 = np.abs(filtered_matched_to_truth_energy)

        fake_energy =  np.sum(filtered_matched_to_predicted_energy[filtered_matched_to_predicted_energy<0])
        fake_energy += np.sum(np.maximum(filtered_matched_to_truth_energy_2 - filtered_truth_energy, 0))

        missed_energy =  np.sum(filtered_truth_energy[filtered_matched_to_truth_energy<0])
        missed_energy += np.sum(np.maximum(filtered_truth_energy - filtered_matched_to_truth_energy_2, 0))

        # print(filtered_truth_energy_2, filtered_matched_to_truth_energy.shape)
        found_energy = np.sum(filtered_truth_energy - np.maximum(filtered_truth_energy - filtered_matched_to_truth_energy_2, 0))

        precision = found / (found + fake)
        recall = found / (found + missed)

        precision_energy = found_energy / (found_energy + fake_energy)
        recall_energy = found_energy / (found_energy + missed_energy)

        # print(h, l, precision_energy, recall_energy, found_energy, missed_energy, fake_energy)
        # print(h, l, precision_energy, recall_energy, precision, recall)
        # print(h, l, precision, recall)

        # probability_values.append(len(filtered_truth_energy))
        precision_values.append(precision)
        recall_values.append(recall)
        precision_energy_values.append(precision_energy)
        recall_energy_values.append(recall_energy)

        mean_energy.append(np.mean(filtered_truth_energy))

        #
        #
        # print(l, h, precision, recall)

    # weight = probability_values / np.sum(probability_values)
    weight = mean_energy * probability_values / np.sum(mean_energy * probability_values)

    precision = np.sum(precision_values * weight)
    recall = np.sum(recall_values * weight)
    f1 = 2 * precision * recall / (precision + recall)
    # print("PRF found", precision, recall, f1)

    precision_energy = np.sum(precision_energy_values * weight)
    recall_energy = np.sum(recall_energy_values * weight)
    f1_energy = 2 * precision_energy * recall_energy / (precision_energy + recall_energy)
    # print("PRF energy", precision_energy, recall_energy, f1_energy)

    return f1, f1_energy


def check(result, use_energy_f_score=True, alpha=0, beta=1):

    print("XYZW")
    # print((result['truth_shower_energy'].shape))
    # print((result['truth_shower_matched_energy_regressed'].shape))
    # print((result['pred_shower_regressed_energy'].shape))
    # print((result['pred_shower_matched_energy'].shape))

    precision, recall, f_score, precision_energy, recall_energy, f_score_energy =  compute_precision_and_recall_analytic(result['truth_shower_energy'], result['truth_shower_matched_energy_regressed'],
                                 result['pred_shower_regressed_energy'],
                                 result['pred_shower_matched_energy'], alpha=alpha, beta=beta, return_all_dict=True)

    f_score_energy = float(f_score_energy)
    f_score = float(f_score)

    import math
    if not math.isfinite(f_score):
        f_score = 0
    if not math.isfinite(f_score_energy):
        f_score_energy = 0

    return f_score_energy if use_energy_f_score else f_score


def compute_scalar_metrics(result, alpha=0, beta=1, prevent_norm=False):

    precision, recall, f_score, precision_energy, recall_energy, f_score_energy =  compute_precision_and_recall_analytic(result['truth_shower_energy'], result['truth_shower_matched_energy_regressed'],
                                 result['pred_shower_regressed_energy'],
                                 result['pred_shower_matched_energy'], alpha=alpha, beta=beta, return_all_dict=True, prevent_norm=prevent_norm)

    f_score_energy = float(f_score_energy)
    f_score = float(f_score)

    precision = float(precision)
    recall = float(recall)

    precision_energy = float(precision_energy)
    recall_energy = float(recall_energy)

    import math
    if not math.isfinite(f_score):
        f_score = 0.0
    if not math.isfinite(f_score_energy):
        f_score_energy = 0.0

    if not math.isfinite(precision):
        precision = 0.0

    if not math.isfinite(recall):
        recall = 0.0

    if not math.isfinite(precision_energy):
        precision_energy = 0.0

    if not math.isfinite(recall_energy):
        recall_energy = 0.0

    return precision, recall, f_score, precision_energy, recall_energy, f_score_energy





def compute_scalar_metrics_graph(result, beta=1):

    truth_shower_energy, truth_shower_matched = matching_and_analysis.get_truth_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
    pred_shower_energy, pred_shower_matched = matching_and_analysis.get_pred_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)

    precision, recall, f_score, precision_energy, recall_energy, f_score_energy =  \
        compute_precision_and_recall_analytic(truth_shower_energy, truth_shower_matched, pred_shower_energy,
                                              pred_shower_matched, alpha=0, beta=beta, return_all_dict=True, prevent_norm=True)

    f_score_energy = float(f_score_energy)
    f_score = float(f_score)

    precision = float(precision)
    recall = float(recall)

    precision_energy = float(precision_energy)
    recall_energy = float(recall_energy)

    import math
    if not math.isfinite(f_score):
        f_score = 0.0
    if not math.isfinite(f_score_energy):
        f_score_energy = 0.0

    if not math.isfinite(precision):
        precision = 0.0

    if not math.isfinite(recall):
        recall = 0.0

    if not math.isfinite(precision_energy):
        precision_energy = 0.0

    if not math.isfinite(recall_energy):
        recall_energy = 0.0

    return precision, recall, f_score, precision_energy, recall_energy, f_score_energy


def calculate_overall_precision(M, c):
    num = 0.
    den = 0.
    for x, y in M:
        if x is None and y is not None:
            pass # Truth shower unmatched
        elif x is not None and y is None:
            # pred shower unmatched, zero precision
            num += 0
            den += max(x['energy'], 0)
        elif x is not None and y is not None:
            # pred shower unmatched, zero precision
            e1 = max(x['energy'], 0)
            e2 = y['energy']
            thisp = min (e1/e2, e2/e1) * e1 if e1 != 0. else 0
            thisp = thisp * (matching_and_analysis.angle(x,y) <= c)
            num += thisp
            den += e1
    precision_value = num / den if den !=0 else 0
    return precision_value


def calculate_overall_absorption(M, c):
    num = 0.
    den = 0.
    for x, y in M:
        if x is None and y is not None:
            pass # Truth shower unmatched
            den += y['energy']
        elif x is not None and y is None:
            # pred shower unmatched, zero precision
            pass
        elif x is not None and y is not None:
            # pred shower unmatched, zero precision
            e1 = max(x['energy'], 0)
            e2 = y['energy']
            den += e2
            num += min(e1, e2) * (matching_and_analysis.angle(x,y) <= c)


    ab_value = num / den
    return ab_value

def compute_precision_and_absorption_graph(graphs, metadata, beta=1):
    M = list()

    truth_count = 0
    for g in graphs:
        truth_count += len(g.nodes)

    for g in graphs:
        # Iterate through truth showers
        for n, att in g.nodes(data=True):
            if att['type'] == 0:
                matched = [x for x in g.neighbors(n)]
                if len(matched) == 0:
                    # Unmatched truth shower
                    M.append((None, att))
                elif len(matched) == 1:
                    M.append((g.nodes(data=True)[matched[0]], att))
                elif len(matched) == 2:
                    return -1, -1
                else:
                    raise RuntimeError("Truth shower matched to multiple pred showers?")

        # Iterate through pred showers
        for n, att in g.nodes(data=True):
            if att['type'] == 1:
                matched = [x for x in g.neighbors(n)]
                if len(matched) == 0:
                    # Unmatched pred shower
                    M.append((att, None))
                elif len(matched) == 1:
                    # Matched pred shower-- can skip?
                    pass
                elif len(matched) == 2:
                    return -1, -1
                else:
                    raise RuntimeError("Truth shower matched to multiple pred showers?")

    nm = 0
    for x, y in M:
        nm += 1 if x is None and y is not None else 0
        nm += 1 if x is not None and y is None else 0
        nm += 2 if x is not None and y is not None else 0
    # assert nm == truth_count

    precision_value = calculate_overall_precision(M, metadata['angle_threshold'])
    ab_value = calculate_overall_absorption(M, metadata['angle_threshold'])

    precision_value = float(precision_value)
    ab_value = float(ab_value)

    return precision_value, ab_value




def compute_scalar_metrics_graph_eff_fake_rate_response(result):
    truth_shower_energy, truth_shower_matched = matching_and_analysis.get_truth_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
    efficiency = float(np.mean(np.not_equal(truth_shower_matched, -1)).item())
    filter = np.not_equal(truth_shower_matched, -1)

    filtered_truth_energy = np.sum(truth_shower_energy[filter])
    response_mean = float(np.sum(truth_shower_energy[filter] * (truth_shower_matched[filter] / truth_shower_energy[filter])).item() / filtered_truth_energy)

    pred_shower_energy, pred_shower_matched = matching_and_analysis.get_pred_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
    fake_rate = float(np.mean(np.equal(pred_shower_matched, -1)).item())

    truth_shower_energy, truth_shower_matched = matching_and_analysis.get_truth_matched_attribute(result, 'dep_energy', 'dep_energy', numpy=True, not_found_value=-1, sum_multi=True)
    filter = np.not_equal(truth_shower_matched, -1)

    response_sum_mean = float(np.sum(truth_shower_energy[filter] * (truth_shower_matched[filter] / truth_shower_energy[filter])).item() / filtered_truth_energy )


    efficiency = efficiency if np.isfinite(efficiency) else 0.
    fake_rate = fake_rate if np.isfinite(fake_rate) else 0.
    response_mean = response_mean if np.isfinite(response_mean) else 0.
    response_sum_mean = response_sum_mean if np.isfinite(response_sum_mean) else 0.

    return efficiency, fake_rate, response_mean, response_sum_mean




def compute_num_showers(result):
    truth_shower_energy, truth_shower_matched = matching_and_analysis.get_truth_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)
    pred_shower_energy, pred_shower_matched = matching_and_analysis.get_pred_matched_attribute(result, 'energy', 'energy', numpy=True, not_found_value=-1, sum_multi=True)


    return len(truth_shower_energy), len(pred_shower_energy)



