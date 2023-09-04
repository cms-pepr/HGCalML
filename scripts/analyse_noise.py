import pdb
import os
import sys
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

input_file = sys.argv[1]
output_file = sys.argv[2]

with gzip.open(input_file, 'rb') as f:
    full_data = pickle.load(f)


total_energy_remaining = []
total_energy_removed = []
total_hits_remaining = []
total_hits_removed = []

noise_energy_remaining = []
noise_energy_removed = []
noise_hits_remaining = []
noise_hits_removed = []

signal_energy_remaining = []
signal_energy_removed = []
signal_hits_remaining = []
signal_hits_removed = []

lost_objects = [] # (e, n)


for data in full_data:
    feat, truth, pred = data
    selected_indices = pred['no_noise_sel'].flatten() # (N,1) -> (N,)
    is_noise = truth['truthHitAssignementIdx'].flatten() == -1
    is_noise_selected = is_noise[selected_indices]
    not_noise = truth['truthHitAssignementIdx'].flatten() != -1
    not_noise_selected = not_noise[selected_indices]
    energy = feat['recHitEnergy'].flatten()
    energy_selected = energy[selected_indices]
    object_ids = np.unique(truth['truthHitAssignementIdx'])
    object_ids_selected = np.unique(truth['truthHitAssignementIdx'][selected_indices])

    e_total = np.sum(energy)
    n_total = energy.shape[0]
    e_noise = np.sum(energy[is_noise])
    n_noise = energy[is_noise].shape[0]
    e_signal = np.sum(energy[not_noise])
    n_signal = energy[not_noise].shape[0]

    total_energy_remaining.append(np.sum(energy_selected))
    total_energy_removed.append(e_total - np.sum(energy_selected))
    total_hits_remaining.append(energy_selected.shape[0])
    total_hits_removed.append(n_total - energy_selected.shape[0])

    noise_energy_remaining.append(energy_selected[is_noise_selected].sum())
    noise_energy_removed.append(e_noise - energy_selected[is_noise_selected].sum())
    noise_hits_remaining.append(energy_selected[is_noise_selected].shape[0])
    noise_hits_removed.append(n_noise - energy_selected[is_noise_selected].shape[0])

    signal_energy_remaining.append(energy_selected[not_noise_selected].sum())
    signal_energy_removed.append(e_signal - energy_selected[not_noise_selected].sum())
    signal_hits_remaining.append(energy_selected[not_noise_selected].shape[0])
    signal_hits_removed.append(n_signal - energy_selected[not_noise_selected].shape[0])

    if len(object_ids) != len(object_ids_selected):
        for ID in object_ids:
            if (ID != -1) and (ID not in object_ids_selected):
                # this means a shower has been completely removed by the noise filter
                mask = truth['truthHitAssignementIdx'] == ID
                n_removed = truth['truthHitAssignementIdx'][mask].shape[0]
                e_removed = truth['truthHitAssignementIdx'][mask].sum()
                lost_objects.append((e_removed, n_removed))
        print(f"Lost {len(lost_objects)} full showers")


output = {
    'total_energy_remaining': np.array(total_energy_remaining),
    'total_energy_removed': np.array(total_energy_removed),
    'total_hits_remaining': np.array(total_hits_remaining),
    'total_hits_removed': np.array(total_hits_removed),
    'noise_energy_remaining': np.array(noise_energy_remaining),
    'noise_energy_removed': np.array(noise_energy_removed),
    'noise_hits_remaining': np.array(noise_hits_remaining),
    'noise_hits_removed': np.array(noise_hits_removed),
    'signal_energy_remaining': np.array(signal_energy_remaining),
    'signal_energy_removed': np.array(signal_energy_removed),
    'signal_hits_remaining': np.array(signal_hits_remaining),
    'signal_hits_removed': np.array(signal_hits_removed),
    'lost_objects': lost_objects,
    }


with open(output_file, 'wb') as f:
    pickle.dump(output, f)


print("DONE")


