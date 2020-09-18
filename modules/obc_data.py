import numpy as np




def build_dataset_analysis_dict():
    data_dict = dict()

    data_dict['beta_threshold'] = -1
    data_dict['distance_threshold'] = -1
    data_dict['iou_threshold'] = -1

    data_dict['truth_shower_energies'] = []
    data_dict['truth_shower_energies_sum'] = []
    data_dict['truth_shower_matched_energies_sum'] = []
    data_dict['truth_shower_matched_energies_sum_ticl'] = []
    data_dict['truth_shower_etas'] = []
    data_dict['truth_shower_local_density'] = []
    data_dict['truth_shower_closest_particle_distance'] = []
    data_dict['truth_showers_found_or_not'] = []
    data_dict['truth_showers_found_or_not_ticl'] = []
    data_dict['truth_showers_sample_id'] = []

    # data_dict['found_showers_predicted_sum'] = []
    # data_dict['found_showers_truth_sum'] = []

    data_dict['num_rechits_per_truth_shower'] = []
    data_dict['num_rechits_per_window'] = []

    # data_dict['found_showers_predicted_energies'] = [] # TODO: These are redundant now - remove them later
    # data_dict['found_showers_target_energies'] = []
    # data_dict['found_showers_predicted_phi'] = []
    # data_dict['found_showers_target_phi'] = []
    # data_dict['found_showers_predicted_eta'] = []
    # data_dict['found_showers_target_eta'] = []

    data_dict['predicted_showers_regressed_energy'] = []
    data_dict['predicted_showers_matched_energy'] = []
    data_dict['predicted_showers_predicted_energy_sum'] = []
    data_dict['predicted_showers_matched_energy_sum'] = []

    data_dict['predicted_total_obc'] = []
    data_dict['predicted_total_ticl'] = []
    data_dict['predicted_total_truth'] = []

    data_dict['truth_shower_matched_energies_regressed'] = []
    data_dict['truth_shower_matched_energies_regressed_ticl'] = []

    data_dict['predicted_showers_regressed_phi'] = []
    data_dict['predicted_showers_matched_phi'] = []
    data_dict['predicted_showers_regressed_eta'] = []
    data_dict['predicted_showers_matched_eta'] = []
    data_dict['predicted_showers_sample_id'] = []

    data_dict['ticl_showers_regressed_energy'] = []
    data_dict['ticl_showers_matched_energy'] = []
    data_dict['ticl_showers_predicted_energy_sum'] = []
    data_dict['ticl_showers_matched_energy_sum'] = []
    data_dict['ticl_showers_regressed_phi'] = []
    data_dict['ticl_showers_matched_phi'] = []
    data_dict['ticl_showers_regressed_eta'] = []
    data_dict['ticl_showers_matched_eta'] = []
    data_dict['ticl_showers_sample_id'] = []

    data_dict['found_showers_predicted_truth_rotational_difference'] = []
    data_dict['num_real_showers'] = []
    data_dict['num_predicted_showers'] = []
    data_dict['num_found_showers'] = []
    data_dict['num_missed_showers'] = []
    data_dict['num_fake_showers'] = []

    data_dict['num_predicted_showers_ticl'] = []
    data_dict['num_found_showers_ticl'] = []
    data_dict['num_missed_showers_ticl'] = []
    data_dict['num_fake_showers_ticl'] = []

    data_dict['visualized_segments'] = []

    return data_dict

def build_window_analysis_dict():
    data_dict = dict()

    data_dict['truth_shower_energies'] = []
    data_dict['truth_shower_energies_sum'] = []
    data_dict['truth_shower_matched_energies_sum'] = []
    data_dict['truth_shower_matched_energies_sum_ticl'] = []
    data_dict['truth_shower_matched_energies_regressed'] = []
    data_dict['truth_shower_matched_energies_regressed_ticl'] = []

    data_dict['truth_shower_etas'] = []
    data_dict['truth_shower_local_density'] = []
    data_dict['truth_shower_closest_particle_distance'] = []
    data_dict['truth_showers_found_or_not'] = []
    data_dict['truth_showers_found_or_not_ticl'] = []
    data_dict['truth_showers_sample_id'] = []


    data_dict['predicted_total_obc'] = -1
    data_dict['predicted_total_ticl'] = -1
    data_dict['predicted_total_truth'] = -1

    # data_dict['found_showers_predicted_sum'] = []
    # data_dict['found_showers_truth_sum'] = []

    data_dict['num_rechits_per_truth_shower'] = []
    data_dict['num_rechits_per_window'] = -1
    data_dict['num_showers_per_window'] = -1

    # data_dict['found_showers_predicted_energies'] = []
    # data_dict['found_showers_target_energies'] = []
    # data_dict['found_showers_predicted_phi'] = []
    # data_dict['found_showers_target_phi'] = []
    # data_dict['found_showers_predicted_eta'] = []
    # data_dict['found_showers_target_eta'] = []


    data_dict['predicted_showers_regressed_energy'] = []
    data_dict['predicted_showers_matched_energy'] = []
    data_dict['predicted_showers_predicted_energy_sum'] = []
    data_dict['predicted_showers_matched_energy_sum'] = []
    data_dict['predicted_showers_regressed_phi'] = []
    data_dict['predicted_showers_matched_phi'] = []
    data_dict['predicted_showers_regressed_eta'] = []
    data_dict['predicted_showers_matched_eta'] = []
    data_dict['predicted_showers_sample_id'] = []


    data_dict['ticl_showers_regressed_energy'] = []
    data_dict['ticl_showers_matched_energy'] = []
    data_dict['ticl_showers_predicted_energy_sum'] = []
    data_dict['ticl_showers_matched_energy_sum'] = []
    data_dict['ticl_showers_regressed_phi'] = []
    data_dict['ticl_showers_matched_phi'] = []
    data_dict['ticl_showers_regressed_eta'] = []
    data_dict['ticl_showers_matched_eta'] = []
    data_dict['ticl_showers_sample_id'] = []

    data_dict['found_showers_predicted_truth_rotational_difference'] = []
    data_dict['num_real_showers'] = -1
    data_dict['num_predicted_showers'] = -1
    data_dict['num_found_showers'] = -1
    data_dict['num_missed_showers'] = -1
    data_dict['num_fake_showers'] = -1



    data_dict['num_predicted_showers_ticl'] = -1
    data_dict['num_found_showers_ticl'] = -1
    data_dict['num_missed_showers_ticl'] = -1
    data_dict['num_fake_showers_ticl'] = -1

    data_dict['visualization_data'] = -1


    return data_dict

def convert_dataset_dict_elements_to_numpy(dataset_dict):
    dataset_dict['beta_threshold'] = np.array(dataset_dict['beta_threshold'])
    dataset_dict['distance_threshold'] = np.array(dataset_dict['distance_threshold'])
    dataset_dict['iou_threshold'] = np.array(dataset_dict['iou_threshold'])

    dataset_dict['truth_shower_energies'] = np.array(dataset_dict['truth_shower_energies'])
    dataset_dict['truth_shower_etas'] = np.array(dataset_dict['truth_shower_etas'])
    dataset_dict['truth_showers_found_or_not'] = np.array(dataset_dict['truth_showers_found_or_not'])
    dataset_dict['truth_showers_found_or_not_ticl'] = np.array(dataset_dict['truth_showers_found_or_not_ticl'])
    dataset_dict['truth_showers_sample_id'] = np.array(dataset_dict['truth_showers_sample_id'])

    dataset_dict['truth_shower_local_density'] = np.array(dataset_dict['truth_shower_local_density'])
    dataset_dict['truth_shower_closest_particle_distance'] = np.array(dataset_dict['truth_shower_closest_particle_distance'])

    dataset_dict['num_rechits_per_truth_shower'] = np.array(dataset_dict['num_rechits_per_truth_shower'])
    dataset_dict['num_rechits_per_window'] = np.array(dataset_dict['num_rechits_per_window'])

    dataset_dict['predicted_showers_regressed_energy'] = np.array(dataset_dict['predicted_showers_regressed_energy'])
    dataset_dict['predicted_showers_matched_energy'] = np.array(dataset_dict['predicted_showers_matched_energy'])
    dataset_dict['predicted_showers_predicted_energy_sum'] = np.array(dataset_dict['predicted_showers_predicted_energy_sum'])
    dataset_dict['predicted_showers_matched_energy_sum'] = np.array(dataset_dict['predicted_showers_matched_energy_sum'])

    dataset_dict['truth_shower_matched_energies_regressed'] = np.array(dataset_dict['truth_shower_matched_energies_regressed'])
    dataset_dict['truth_shower_matched_energies_regressed_ticl'] = np.array(dataset_dict['truth_shower_matched_energies_regressed_ticl'])

    dataset_dict['predicted_showers_regressed_phi'] = np.array(dataset_dict['predicted_showers_regressed_phi'])
    dataset_dict['predicted_showers_matched_phi'] = np.array(dataset_dict['predicted_showers_matched_phi'])
    dataset_dict['predicted_showers_regressed_eta'] = np.array(dataset_dict['predicted_showers_regressed_eta'])
    dataset_dict['predicted_showers_matched_eta'] = np.array(dataset_dict['predicted_showers_matched_eta'])
    dataset_dict['predicted_showers_sample_id'] = np.array(dataset_dict['predicted_showers_sample_id'])

    dataset_dict['ticl_showers_regressed_energy'] = np.array(dataset_dict['ticl_showers_regressed_energy'])
    dataset_dict['ticl_showers_matched_energy'] = np.array(dataset_dict['ticl_showers_matched_energy'])
    dataset_dict['ticl_showers_predicted_energy_sum'] = np.array(dataset_dict['ticl_showers_predicted_energy_sum'])
    dataset_dict['ticl_showers_matched_energy_sum'] = np.array(dataset_dict['ticl_showers_matched_energy_sum'])
    dataset_dict['ticl_showers_regressed_phi'] = np.array(dataset_dict['ticl_showers_regressed_phi'])
    dataset_dict['ticl_showers_matched_phi'] = np.array(dataset_dict['ticl_showers_matched_phi'])
    dataset_dict['ticl_showers_regressed_eta'] = np.array(dataset_dict['ticl_showers_regressed_eta'])
    dataset_dict['ticl_showers_matched_eta'] = np.array(dataset_dict['ticl_showers_matched_eta'])
    dataset_dict['ticl_showers_sample_id'] = np.array(dataset_dict['ticl_showers_sample_id'])

    dataset_dict['found_showers_predicted_truth_rotational_difference'] = np.array(dataset_dict['found_showers_predicted_truth_rotational_difference'])
    dataset_dict['num_real_showers'] = np.array(dataset_dict['num_real_showers'])
    dataset_dict['num_predicted_showers'] = np.array(dataset_dict['num_predicted_showers'])
    dataset_dict['num_found_showers'] = np.array(dataset_dict['num_found_showers'])
    dataset_dict['num_missed_showers'] = np.array(dataset_dict['num_missed_showers'])
    dataset_dict['num_fake_showers'] = np.array(dataset_dict['num_fake_showers'])

    dataset_dict['num_predicted_showers_ticl'] = np.array(dataset_dict['num_predicted_showers_ticl'])
    dataset_dict['num_found_showers_ticl'] = np.array(dataset_dict['num_found_showers_ticl'])
    dataset_dict['num_missed_showers_ticl'] = np.array(dataset_dict['num_missed_showers_ticl'])
    dataset_dict['num_fake_showers_ticl'] = np.array(dataset_dict['num_fake_showers_ticl'])

    dataset_dict['predicted_total_obc'] = np.array(dataset_dict['predicted_total_obc'])
    dataset_dict['predicted_total_ticl'] = np.array(dataset_dict['predicted_total_ticl'])
    dataset_dict['predicted_total_truth'] = np.array(dataset_dict['predicted_total_truth'])

    dataset_dict['visualized_segments'] = np.array(dataset_dict['visualized_segments'])

    return dataset_dict


def append_window_dict_to_dataset_dict(dataset_dict, window_dict):

    dataset_dict['truth_shower_energies'] += window_dict['truth_shower_energies']

    dataset_dict['truth_shower_energies_sum'] += window_dict['truth_shower_energies_sum']
    dataset_dict['truth_shower_matched_energies_sum'] += window_dict['truth_shower_matched_energies_sum']
    dataset_dict['truth_shower_matched_energies_sum_ticl'] += window_dict['truth_shower_matched_energies_sum_ticl']

    dataset_dict['truth_shower_matched_energies_regressed'] += window_dict['truth_shower_matched_energies_regressed']
    dataset_dict['truth_shower_matched_energies_regressed_ticl'] += window_dict['truth_shower_matched_energies_regressed_ticl']

    dataset_dict['truth_shower_etas'] += window_dict['truth_shower_etas']

    dataset_dict['truth_shower_local_density'] += window_dict['truth_shower_local_density']
    dataset_dict['truth_shower_closest_particle_distance'] += window_dict['truth_shower_closest_particle_distance']

    dataset_dict['truth_showers_found_or_not'] += window_dict['truth_showers_found_or_not']
    dataset_dict['truth_showers_found_or_not_ticl'] += window_dict['truth_showers_found_or_not_ticl']
    dataset_dict['truth_showers_sample_id'] += window_dict['truth_showers_sample_id']

    # dataset_dict['found_showers_predicted_sum'] += window_dict['found_showers_predicted_sum']
    # dataset_dict['found_showers_truth_sum'] += window_dict['found_showers_truth_sum']

    dataset_dict['num_rechits_per_truth_shower'] += window_dict['num_rechits_per_truth_shower']
    dataset_dict['num_rechits_per_window'].append(window_dict['num_rechits_per_window'])

    # dataset_dict['found_showers_predicted_energies'] += window_dict['found_showers_predicted_energies']
    # dataset_dict['found_showers_target_energies'] += window_dict['found_showers_target_energies']
    # dataset_dict['found_showers_predicted_phi'] += window_dict['found_showers_predicted_phi']
    # dataset_dict['found_showers_target_phi'] += window_dict['found_showers_target_phi']
    # dataset_dict['found_showers_predicted_eta'] += window_dict['found_showers_predicted_eta']
    # dataset_dict['found_showers_target_eta'] += window_dict['found_showers_target_eta']

    dataset_dict['predicted_showers_regressed_energy'] += window_dict['predicted_showers_regressed_energy']
    dataset_dict['predicted_showers_matched_energy'] += window_dict['predicted_showers_matched_energy']
    dataset_dict['predicted_showers_predicted_energy_sum'] += window_dict['predicted_showers_predicted_energy_sum']
    dataset_dict['predicted_showers_matched_energy_sum'] += window_dict['predicted_showers_matched_energy_sum']
    dataset_dict['predicted_showers_regressed_phi'] += window_dict['predicted_showers_regressed_phi']
    dataset_dict['predicted_showers_matched_phi'] += window_dict['predicted_showers_matched_phi']
    dataset_dict['predicted_showers_regressed_eta'] += window_dict['predicted_showers_regressed_eta']
    dataset_dict['predicted_showers_matched_eta'] += window_dict['predicted_showers_matched_eta']
    dataset_dict['predicted_showers_sample_id'] += window_dict['predicted_showers_sample_id']


    dataset_dict['ticl_showers_regressed_energy'] += window_dict['ticl_showers_regressed_energy']
    dataset_dict['ticl_showers_matched_energy'] += window_dict['ticl_showers_matched_energy']
    dataset_dict['ticl_showers_predicted_energy_sum'] += window_dict['ticl_showers_predicted_energy_sum']
    dataset_dict['ticl_showers_matched_energy_sum'] += window_dict['ticl_showers_matched_energy_sum']
    dataset_dict['ticl_showers_regressed_phi'] += window_dict['ticl_showers_regressed_phi']
    dataset_dict['ticl_showers_matched_phi'] += window_dict['ticl_showers_matched_phi']
    dataset_dict['ticl_showers_regressed_eta'] += window_dict['ticl_showers_regressed_eta']
    dataset_dict['ticl_showers_matched_eta'] += window_dict['ticl_showers_matched_eta']
    dataset_dict['ticl_showers_sample_id'] += window_dict['ticl_showers_sample_id']



    dataset_dict['found_showers_predicted_truth_rotational_difference'] += window_dict['found_showers_predicted_truth_rotational_difference']

    dataset_dict['num_real_showers'].append(window_dict['num_real_showers'])
    dataset_dict['num_predicted_showers'].append(window_dict['num_predicted_showers'])
    dataset_dict['num_found_showers'].append(window_dict['num_found_showers'])
    dataset_dict['num_missed_showers'].append(window_dict['num_missed_showers'])
    dataset_dict['num_fake_showers'].append(window_dict['num_fake_showers'])

    dataset_dict['num_predicted_showers_ticl'].append(window_dict['num_predicted_showers_ticl'])
    dataset_dict['num_found_showers_ticl'].append(window_dict['num_found_showers_ticl'])
    dataset_dict['num_missed_showers_ticl'].append(window_dict['num_missed_showers_ticl'])
    dataset_dict['num_fake_showers_ticl'].append(window_dict['num_fake_showers_ticl'])

    dataset_dict['predicted_total_obc'].append(window_dict['predicted_total_obc'])
    dataset_dict['predicted_total_ticl'].append(window_dict['predicted_total_ticl'])
    dataset_dict['predicted_total_truth'].append(window_dict['predicted_total_truth'])

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




