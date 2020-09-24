import numpy as np


def build_dataset_analysis_dict():
    data_dict = dict()

    data_dict['beta_threshold'] = -1
    data_dict['distance_threshold'] = -1
    data_dict['iou_threshold'] = -1

    data_dict['truth_shower_energy'] = []
    data_dict['truth_shower_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum'] = []
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

    data_dict['window_num_rechits'] = []

    data_dict['window_total_energy_pred'] = []
    data_dict['window_total_energy_ticl'] = []
    data_dict['window_total_energy_truth'] = []

    data_dict['pred_shower_regressed_energy'] = []
    data_dict['pred_shower_matched_energy'] = []
    data_dict['pred_shower_energy_sum'] = []
    data_dict['pred_shower_matched_energy_sum'] = []
    data_dict['pred_shower_regressed_phi'] = []
    data_dict['pred_shower_matched_phi'] = []
    data_dict['pred_shower_regressed_eta'] = []
    data_dict['pred_shower_matched_eta'] = []
    data_dict['pred_shower_sid'] = []
    data_dict['pred_shower_sample_id'] = []

    data_dict['ticl_shower_regressed_energy'] = []
    data_dict['ticl_shower_matched_energy'] = []
    data_dict['ticl_shower_energy_sum'] = []
    data_dict['ticl_shower_matched_energy_sum'] = []
    data_dict['ticl_shower_regressed_phi'] = []
    data_dict['ticl_shower_matched_phi'] = []
    data_dict['ticl_shower_regressed_eta'] = []
    data_dict['ticl_shower_matched_eta'] = []
    data_dict['ticl_shower_sid'] = []
    data_dict['ticl_shower_sample_id'] = []

    data_dict['window_num_truth_showers'] = []

    data_dict['window_num_pred_showers'] = []
    data_dict['window_num_found_showers'] = []
    data_dict['window_num_missed_showers'] = []
    data_dict['window_num_fake_showers'] = []

    data_dict['window_num_ticl_showers'] = []
    data_dict['window_num_found_showers_ticl'] = []
    data_dict['window_num_missed_showers_ticl'] = []
    data_dict['window_num_fake_showers_ticl'] = []

    data_dict['visualized_segments'] = []

    return data_dict


def build_window_analysis_dict():
    data_dict = dict()

    data_dict['truth_shower_energy'] = []
    data_dict['truth_shower_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum'] = []
    data_dict['truth_shower_matched_energy_sum_ticl'] = []
    data_dict['truth_shower_matched_energy_regressed'] = []
    data_dict['truth_shower_matched_energy_regressed_ticl'] = []

    data_dict['truth_shower_eta'] = []
    data_dict['truth_shower_local_density'] = []
    data_dict['truth_shower_closest_particle_distance'] = []
    data_dict['truth_shower_found_or_not'] = []
    data_dict['truth_shower_found_or_not_ticl'] = []
    data_dict['truth_shower_sid'] = []
    data_dict['truth_shower_sample_id'] = []

    data_dict['window_total_energy_pred'] = -1
    data_dict['window_total_energy_ticl'] = -1
    data_dict['window_total_energy_truth'] = -1

    data_dict['truth_shower_num_rechits'] = []
    data_dict['window_num_rechits'] = -1
    data_dict['num_showers_per_window'] = -1

    data_dict['pred_shower_regressed_energy'] = []
    data_dict['pred_shower_matched_energy'] = []
    data_dict['pred_shower_energy_sum'] = []
    data_dict['pred_shower_matched_energy_sum'] = []
    data_dict['pred_shower_regressed_phi'] = []
    data_dict['pred_shower_matched_phi'] = []
    data_dict['pred_shower_regressed_eta'] = []
    data_dict['pred_shower_matched_eta'] = []
    data_dict['pred_shower_sid'] = []
    data_dict['pred_shower_sample_id'] = []

    data_dict['ticl_shower_regressed_energy'] = []
    data_dict['ticl_shower_matched_energy'] = []
    data_dict['ticl_shower_energy_sum'] = []
    data_dict['ticl_shower_matched_energy_sum'] = []
    data_dict['ticl_shower_regressed_phi'] = []
    data_dict['ticl_shower_matched_phi'] = []
    data_dict['ticl_shower_regressed_eta'] = []
    data_dict['ticl_shower_matched_eta'] = []
    data_dict['ticl_shower_sid'] = []
    data_dict['ticl_shower_sample_id'] = []

    data_dict['found_showers_predicted_truth_rotational_difference'] = []
    data_dict['window_num_truth_showers'] = -1
    data_dict['window_num_pred_showers'] = -1
    data_dict['window_num_found_showers'] = -1
    data_dict['window_num_missed_showers'] = -1
    data_dict['window_num_fake_showers'] = -1

    data_dict['window_num_ticl_showers'] = -1
    data_dict['window_num_found_showers_ticl'] = -1
    data_dict['window_num_missed_showers_ticl'] = -1
    data_dict['window_num_fake_showers_ticl'] = -1

    data_dict['visualization_data'] = -1

    return data_dict


def convert_dataset_dict_elements_to_numpy(dataset_dict):
    dataset_dict['beta_threshold'] = np.array(dataset_dict['beta_threshold'])
    dataset_dict['distance_threshold'] = np.array(dataset_dict['distance_threshold'])
    dataset_dict['iou_threshold'] = np.array(dataset_dict['iou_threshold'])

    dataset_dict['truth_shower_energy'] = np.array(dataset_dict['truth_shower_energy'])
    dataset_dict['truth_shower_eta'] = np.array(dataset_dict['truth_shower_eta'])
    dataset_dict['truth_shower_found_or_not'] = np.array(dataset_dict['truth_shower_found_or_not'])
    dataset_dict['truth_shower_found_or_not_ticl'] = np.array(dataset_dict['truth_shower_found_or_not_ticl'])
    dataset_dict['truth_shower_sid'] = np.array(dataset_dict['truth_shower_sid'])
    dataset_dict['truth_shower_sample_id'] = np.array(dataset_dict['truth_shower_sample_id'])

    dataset_dict['truth_shower_local_density'] = np.array(dataset_dict['truth_shower_local_density'])
    dataset_dict['truth_shower_closest_particle_distance'] = np.array(
        dataset_dict['truth_shower_closest_particle_distance'])

    dataset_dict['truth_shower_num_rechits'] = np.array(dataset_dict['truth_shower_num_rechits'])
    dataset_dict['window_num_rechits'] = np.array(dataset_dict['window_num_rechits'])

    dataset_dict['pred_shower_regressed_energy'] = np.array(dataset_dict['pred_shower_regressed_energy'])
    dataset_dict['pred_shower_matched_energy'] = np.array(dataset_dict['pred_shower_matched_energy'])
    dataset_dict['pred_shower_energy_sum'] = np.array(dataset_dict['pred_shower_energy_sum'])
    dataset_dict['pred_shower_matched_energy_sum'] = np.array(dataset_dict['pred_shower_matched_energy_sum'])

    dataset_dict['truth_shower_matched_energy_regressed'] = np.array(
        dataset_dict['truth_shower_matched_energy_regressed'])
    dataset_dict['truth_shower_matched_energy_regressed_ticl'] = np.array(
        dataset_dict['truth_shower_matched_energy_regressed_ticl'])

    dataset_dict['pred_shower_regressed_phi'] = np.array(dataset_dict['pred_shower_regressed_phi'])
    dataset_dict['pred_shower_matched_phi'] = np.array(dataset_dict['pred_shower_matched_phi'])
    dataset_dict['pred_shower_regressed_eta'] = np.array(dataset_dict['pred_shower_regressed_eta'])
    dataset_dict['pred_shower_matched_eta'] = np.array(dataset_dict['pred_shower_matched_eta'])
    dataset_dict['pred_shower_sid'] = np.array(dataset_dict['pred_shower_sid'])
    dataset_dict['pred_shower_sample_id'] = np.array(dataset_dict['pred_shower_sample_id'])

    dataset_dict['ticl_shower_regressed_energy'] = np.array(dataset_dict['ticl_shower_regressed_energy'])
    dataset_dict['ticl_shower_matched_energy'] = np.array(dataset_dict['ticl_shower_matched_energy'])
    dataset_dict['ticl_shower_energy_sum'] = np.array(dataset_dict['ticl_shower_energy_sum'])
    dataset_dict['ticl_shower_matched_energy_sum'] = np.array(dataset_dict['ticl_shower_matched_energy_sum'])
    dataset_dict['ticl_shower_regressed_phi'] = np.array(dataset_dict['ticl_shower_regressed_phi'])
    dataset_dict['ticl_shower_matched_phi'] = np.array(dataset_dict['ticl_shower_matched_phi'])
    dataset_dict['ticl_shower_regressed_eta'] = np.array(dataset_dict['ticl_shower_regressed_eta'])
    dataset_dict['ticl_shower_matched_eta'] = np.array(dataset_dict['ticl_shower_matched_eta'])
    dataset_dict['ticl_shower_sid'] = np.array(dataset_dict['ticl_shower_sid'])
    dataset_dict['ticl_shower_sample_id'] = np.array(dataset_dict['ticl_shower_sample_id'])

    dataset_dict['window_num_truth_showers'] = np.array(dataset_dict['window_num_truth_showers'])
    dataset_dict['window_num_pred_showers'] = np.array(dataset_dict['window_num_pred_showers'])
    dataset_dict['window_num_found_showers'] = np.array(dataset_dict['window_num_found_showers'])
    dataset_dict['window_num_missed_showers'] = np.array(dataset_dict['window_num_missed_showers'])
    dataset_dict['window_num_fake_showers'] = np.array(dataset_dict['window_num_fake_showers'])

    dataset_dict['window_num_ticl_showers'] = np.array(dataset_dict['window_num_ticl_showers'])
    dataset_dict['window_num_found_showers_ticl'] = np.array(dataset_dict['window_num_found_showers_ticl'])
    dataset_dict['window_num_missed_showers_ticl'] = np.array(dataset_dict['window_num_missed_showers_ticl'])
    dataset_dict['window_num_fake_showers_ticl'] = np.array(dataset_dict['window_num_fake_showers_ticl'])

    dataset_dict['window_total_energy_pred'] = np.array(dataset_dict['window_total_energy_pred'])
    dataset_dict['window_total_energy_ticl'] = np.array(dataset_dict['window_total_energy_ticl'])
    dataset_dict['window_total_energy_truth'] = np.array(dataset_dict['window_total_energy_truth'])

    dataset_dict['visualized_segments'] = np.array(dataset_dict['visualized_segments'])

    return dataset_dict


def append_window_dict_to_dataset_dict(dataset_dict, window_dict):
    dataset_dict['truth_shower_energy'] += window_dict['truth_shower_energy']

    dataset_dict['truth_shower_energy_sum'] += window_dict['truth_shower_energy_sum']
    dataset_dict['truth_shower_matched_energy_sum'] += window_dict['truth_shower_matched_energy_sum']
    dataset_dict['truth_shower_matched_energy_sum_ticl'] += window_dict['truth_shower_matched_energy_sum_ticl']

    dataset_dict['truth_shower_matched_energy_regressed'] += window_dict['truth_shower_matched_energy_regressed']
    dataset_dict['truth_shower_matched_energy_regressed_ticl'] += window_dict[
        'truth_shower_matched_energy_regressed_ticl']

    dataset_dict['truth_shower_eta'] += window_dict['truth_shower_eta']

    dataset_dict['truth_shower_local_density'] += window_dict['truth_shower_local_density']
    dataset_dict['truth_shower_closest_particle_distance'] += window_dict['truth_shower_closest_particle_distance']

    dataset_dict['truth_shower_found_or_not'] += window_dict['truth_shower_found_or_not']
    dataset_dict['truth_shower_found_or_not_ticl'] += window_dict['truth_shower_found_or_not_ticl']
    dataset_dict['truth_shower_sid'] += window_dict['truth_shower_sid']
    dataset_dict['truth_shower_sample_id'] += window_dict['truth_shower_sample_id']

    dataset_dict['truth_shower_num_rechits'] += window_dict['truth_shower_num_rechits']
    dataset_dict['window_num_rechits'].append(window_dict['window_num_rechits'])

    dataset_dict['pred_shower_regressed_energy'] += window_dict['pred_shower_regressed_energy']
    dataset_dict['pred_shower_matched_energy'] += window_dict['pred_shower_matched_energy']
    dataset_dict['pred_shower_energy_sum'] += window_dict['pred_shower_energy_sum']
    dataset_dict['pred_shower_matched_energy_sum'] += window_dict['pred_shower_matched_energy_sum']
    dataset_dict['pred_shower_regressed_phi'] += window_dict['pred_shower_regressed_phi']
    dataset_dict['pred_shower_matched_phi'] += window_dict['pred_shower_matched_phi']
    dataset_dict['pred_shower_regressed_eta'] += window_dict['pred_shower_regressed_eta']
    dataset_dict['pred_shower_matched_eta'] += window_dict['pred_shower_matched_eta']
    dataset_dict['pred_shower_sid'] += window_dict['pred_shower_sid']
    dataset_dict['pred_shower_sample_id'] += window_dict['pred_shower_sample_id']

    dataset_dict['ticl_shower_regressed_energy'] += window_dict['ticl_shower_regressed_energy']
    dataset_dict['ticl_shower_matched_energy'] += window_dict['ticl_shower_matched_energy']
    dataset_dict['ticl_shower_energy_sum'] += window_dict['ticl_shower_energy_sum']
    dataset_dict['ticl_shower_matched_energy_sum'] += window_dict['ticl_shower_matched_energy_sum']
    dataset_dict['ticl_shower_regressed_phi'] += window_dict['ticl_shower_regressed_phi']
    dataset_dict['ticl_shower_matched_phi'] += window_dict['ticl_shower_matched_phi']
    dataset_dict['ticl_shower_regressed_eta'] += window_dict['ticl_shower_regressed_eta']
    dataset_dict['ticl_shower_matched_eta'] += window_dict['ticl_shower_matched_eta']
    dataset_dict['ticl_shower_sid'] += window_dict['ticl_shower_sid']
    dataset_dict['ticl_shower_sample_id'] += window_dict['ticl_shower_sample_id']

    dataset_dict['window_num_truth_showers'].append(window_dict['window_num_truth_showers'])
    dataset_dict['window_num_pred_showers'].append(window_dict['window_num_pred_showers'])
    dataset_dict['window_num_found_showers'].append(window_dict['window_num_found_showers'])
    dataset_dict['window_num_missed_showers'].append(window_dict['window_num_missed_showers'])
    dataset_dict['window_num_fake_showers'].append(window_dict['window_num_fake_showers'])

    dataset_dict['window_num_ticl_showers'].append(window_dict['window_num_ticl_showers'])
    dataset_dict['window_num_found_showers_ticl'].append(window_dict['window_num_found_showers_ticl'])
    dataset_dict['window_num_missed_showers_ticl'].append(window_dict['window_num_missed_showers_ticl'])
    dataset_dict['window_num_fake_showers_ticl'].append(window_dict['window_num_fake_showers_ticl'])

    dataset_dict['window_total_energy_pred'].append(window_dict['window_total_energy_pred'])
    dataset_dict['window_total_energy_ticl'].append(window_dict['window_total_energy_ticl'])
    dataset_dict['window_total_energy_truth'].append(window_dict['window_total_energy_truth'])

    if window_dict['visualization_data'] != -1:
        dataset_dict['visualized_segments'].append(window_dict['visualization_data'])

    return window_dict


def build_window_visualization_dict():
    vis_dict = dict()
    vis_dict['truth_showers'] = -1

    vis_dict['pred_and_truth_dict'] = -1
    vis_dict['feature_dict'] = -1

    vis_dict['predicted_showers'] = -1
    vis_dict['ticl_showers'] = -1
    vis_dict['coords_representatives'] = -1
    vis_dict['identified_vertices'] = -1

    return vis_dict

