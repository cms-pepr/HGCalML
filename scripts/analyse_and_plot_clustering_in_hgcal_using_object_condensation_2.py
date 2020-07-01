from __future__ import print_function

import tensorflow as tf
# from K import Layer
import numpy as np
from LayersRagged import RaggedConstructTensor
import os
import argparse
import matplotlib.pyplot as plt
import gzip
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from ragged_plotting_tools import make_cluster_coordinates_plot, make_original_truth_shower_plot, createRandomizedColors
from DeepJetCore.training.gpuTools import DJCSetGPUs


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


# tf.compat.v1.disable_eager_execution()


os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



ragged_constructor = RaggedConstructTensor()


num_real_showers_g = []
num_predicted_showers_g = []

num_found_g = []
num_missed_g = []
num_fakes_g = []

iii = 0


def find_uniques_from_betas(betas, coords, dist_threshold):

    n2_distances = np.sqrt(np.sum(np.square(np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)), axis=-1))
    betas_checked = np.zeros_like(betas) - 1

    index = 0

    arange_vector = np.arange(len(betas))

    representative_indices = []

    while True:
        betas_remaining = betas[betas_checked==-1]
        arange_remaining = arange_vector[betas_checked==-1]

        if len(betas_remaining)==0:
            break

        max_beta = arange_remaining[np.argmax(betas_remaining)]

        representative_indices.append(max_beta)


        n2 = n2_distances[max_beta]

        distances_less = np.logical_and(n2<dist_threshold, betas_checked==-1)
        betas_checked[distances_less] = index

        index += 1


    return betas_checked, representative_indices

def replace(arr, rep_dict):
    """Assumes all elements of "arr" are keys of rep_dict"""

    # Removing the explicit "list" breaks python3
    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))

    idces = np.digitize(arr, rep_keys, right=True)
    # Notice rep_keys[digitize(arr, rep_keys, right=True)] == arr

    return rep_vals[idces]


def assign_classes_to_full_unfiltered_vertices(beta_all, clustering_coords_all, labels, clustering_coords_all_filtered, betas_filtered, distance_threshold):
    unique_labels = np.unique(labels)

    labels_all = np.zeros_like(beta_all) - 1

    centers = []
    labelsc = []

    replacedict = {}
    iii = 0

    replacedict[-1] = -1

    for x in unique_labels:
        if x == -1:
            continue
        center = clustering_coords_all_filtered[labels==x][np.argmax(betas_filtered[labels==x])]
        centers.append(center[np.newaxis, ...])
        labelsc.append(x)

        replacedict[iii] = x


        distances = np.sqrt(np.sum(np.square(clustering_coords_all - center[np.newaxis, ...]), axis=-1))
        labels_all[distances < distance_threshold] = x

        iii += 1

    # centers = np.concatenate(centers, axis=0)

    # labels_all = np.argmin(np.sqrt(np.sum(np.square(np.expand_dims(clustering_coords_all, axis=1) - np.expand_dims(centers, axis=0)), axis=-1)), axis=-1)
    labels_all = replace(labels_all, replacedict)

    return labels_all




found_truth_energies_energies_truth = []
found_truth_energies_is_it_found = []

found_2_truth_energies_predicted_sum = []
found_2_truth_energies_truth_sum = []

found_2_truth_target_energies = []
found_2_truth_predicted_energies = []

found_2_truth_rotational_distance = []


beta_threshold = 0.1
distance_threshold = 0.5

def analyse_one_window_cut(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment):
    global  iii, num_predicted_showers_g, num_real_showers_g, num_fakes_g, num_found_g, num_missed_g, found_truth_energies_energies_truth, found_truth_energies_is_it_found
    global found_2_truth_energies_predicted_sum, found_2_truth_energies_truth_sum
    global beta_threshold, distance_threshold
    global found_2_truth_rotational_distance
    global found_2_truth_predicted_energies
    global found_2_truth_target_energies
    global num_rechits_per_shower
    global num_segments_to_visualize, num_visualized_segments

    # print(np.mean(y_this_segment[:, 8]), np.mean(y_this_segment[:, 9]))

    iii+=1
    unique_showers_this_segment,unique_showers_indices = np.unique(classes_this_segment, return_index=True)
    truth_energies_this_segment = y_this_segment[:, 1]
    unique_showers_energies = truth_energies_this_segment[unique_showers_indices]

    rechit_energies_this_segment = x_this_segment[:, 0]


    beta_all = pred_this_segment[:, -6]
    # is_spectator = np.logical_not(y_this_segment[:, 14])
    # is_spectator = np.logical_and(is_spectator, beta_all>beta_threshold)
    is_spectator = beta_all>beta_threshold
    # is_spectator = np.logical_and(is_spectator, classes_this_segment>=0)

    # print("XYZ", np.sum(classes_this_segment<=0), distance_threshold)


    beta_all_filtered = beta_all[is_spectator==1]
    y_all_filtered = y_this_segment[is_spectator==1]
    x_filtered = x_this_segment[is_spectator==1]

    prediction_filtered = pred_this_segment[is_spectator==1]


    clustering_coords_all = pred_this_segment[:, -2:]
    clustering_coords_all_filtered = clustering_coords_all[is_spectator==1, :]

    labels, representative_indices = find_uniques_from_betas(beta_all_filtered, clustering_coords_all_filtered, dist_threshold=distance_threshold)

    labels_for_all = assign_classes_to_full_unfiltered_vertices(beta_all, clustering_coords_all, labels, clustering_coords_all_filtered, beta_all_filtered, distance_threshold=distance_threshold)

    unique_labels = np.unique(labels)

    truth_showers_found = {}
    truth_showers_found_e = {}
    iii = 0
    for x in unique_showers_this_segment:
        truth_showers_found[x] = -1
        truth_showers_found_e[x] = unique_showers_energies[iii]
        iii += 1


    predicted_showers_found = {}
    for x in unique_labels:
        predicted_showers_found[x] = -1

    representative_coords = []
    ii_p = 0
    for representative_index in representative_indices:
        # rechit_energies_this_segment[labels_for_all==ii_p]
        representative_coords.append(clustering_coords_all_filtered[representative_index])

        top_match_index = -1
        top_match_shower = -1
        top_match_value = 0

        top_match_shower = classes_this_segment[representative_index]
        if truth_showers_found[top_match_shower] != -1:
            top_match_shower = -1

        for i in range(len(unique_showers_this_segment)):

            if truth_showers_found[unique_showers_this_segment[i]] != -1:
                continue

            overlap = np.sum(rechit_energies_this_segment * (classes_this_segment==unique_showers_this_segment[i]) * (labels_for_all==ii_p)) / np.sum(rechit_energies_this_segment * np.logical_or((classes_this_segment==unique_showers_this_segment[i]), (labels_for_all==ii_p)))

            if overlap > top_match_value:
                top_match_index == i
                top_match_shower = unique_showers_this_segment[i]
                top_match_value = overlap

        if top_match_shower != -1:
            truth_showers_found[top_match_shower] = ii_p
            predicted_showers_found[ii_p] = top_match_shower

        ii_p += 1

    num_found = 0.
    num_missed = 0.
    num_gt_showers = 0.

    num_fakes = 0.
    num_predicted_showers = 0.


    for k,v in truth_showers_found.items():
        found_truth_energies_energies_truth.append(truth_showers_found_e[k])

        if v > -1:
            found_truth_energies_is_it_found.append(True)
            num_found += 1
        else:
            found_truth_energies_is_it_found.append(False)
            num_missed += 1

        num_gt_showers += 1

    for k, v in predicted_showers_found.items():
        if v > -1:
            pass
        else:
            num_fakes += 1

        num_predicted_showers += 1

    print(num_found / num_gt_showers, num_missed / num_gt_showers, num_fakes / num_predicted_showers)

    num_found_g.append(num_found)
    num_missed_g.append(num_missed)
    num_fakes_g.append(num_fakes)
    num_real_showers_g.append(num_gt_showers)
    num_predicted_showers_g.append(num_predicted_showers)

    representative_coords = np.array(representative_coords)

    print("Visualizing")
    if num_visualized_segments < num_showers_to_visualize:
        visualize_the_segment(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment, labels_for_all, representative_coords)
        num_visualized_segments += 1




def visualize_the_segment(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment, labels, coords_representative_predicted_showers, identified_vertices=None):
    global pdf, distance_threshold
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(3,2)

    ax = [fig.add_subplot(gs[0, 0], projection='3d'),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0], projection='3d'),
          fig.add_subplot(gs[1, 1], projection='3d'),]

    # wrt ground truth colors

    ax[0].set_xlabel('z (cm)')
    ax[0].set_ylabel('y (cm)')
    ax[0].set_zlabel('x (cm)')
    ax[0].set_title('Input data')


    ax[1].set_xlabel('Clustering dimension 1')
    ax[1].set_ylabel('Clustering dimension 2')
    ax[1].set_title('Colors = truth showers')

    ax[2].set_xlabel('z (cm)')
    ax[2].set_ylabel('y (cm)')
    ax[2].set_zlabel('x (cm)')
    ax[2].set_title('Colors = truth showers')

    ax[3].set_xlabel('z (cm)')
    ax[3].set_ylabel('y (cm)')
    ax[3].set_zlabel('x (cm)')
    ax[3].set_title('Colors = predicted showers')

    cmap = createRandomizedColors('jet')

    make_original_truth_shower_plot(plt, ax[0], classes_this_segment*0, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Wistia'))
    make_cluster_coordinates_plot(plt, ax[1], classes_this_segment, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap, distance_threshold=distance_threshold)
    #
    # make_original_truth_shower_plot(plt, ax[4], 1-identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))
    # make_original_truth_shower_plot(plt, ax[5], identified_vertices, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=plt.get_cmap('Reds'))

    # wrt predicted colors
    make_original_truth_shower_plot(plt, ax[2], classes_this_segment, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap)
    make_original_truth_shower_plot(plt, ax[3], labels, x_this_segment[:, 0], x_this_segment[:, 5], x_this_segment[:, 6], x_this_segment[:, 7], cmap=cmap)

    # make_cluster_coordinates_plot(plt, ax[3], labels, pred_this_segment[:, -6], pred_this_segment[:, -2:], identified_coords=coords_representative_predicted_showers, cmap=cmap)

    print("\n\n\n\n\nSAVING OUR FIGURE TO PDF\n\n\n\n\n")
    pdf.savefig()





num_visualized_segments = 0
truth_energies = []

num_rechits_per_segment = []
num_rechits_per_shower = []
def analyse_one_file(features, predictions, truth):
    global num_rechits_per_segment
    predictions = tf.constant(predictions[0])

    row_splits = features[1][:, 0]

    x_data, _ = ragged_constructor((features[0], row_splits))
    y_data, _ = ragged_constructor((truth[0], row_splits))
    classes, row_splits = ragged_constructor((truth[0][:, 0][..., tf.newaxis], row_splits))

    classes = tf.cast(classes[:, 0], tf.int32)

    num_unique = []
    shower_sizes = []

    for i in range(len(row_splits) - 1):
        classes_this_segment = classes[row_splits[i]:row_splits[i + 1]].numpy()
        x_this_segment = x_data[row_splits[i]:row_splits[i + 1]].numpy()
        y_this_segment = y_data[row_splits[i]:row_splits[i + 1]].numpy()
        pred_this_segment = predictions[row_splits[i]:row_splits[i + 1]].numpy()

        truth_energies.append(np.unique(y_this_segment[:, 1]))

        analyse_one_window_cut(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment)

        num_rechits_per_segment.append(len(x_this_segment))



    i += 1





def make_plots(pdfpath, dumppath):
    global truth_energies, found_truth_energies_is_it_found, found_truth_energies_energies_truth
    global num_real_showers_g, num_predicted_showers_g
    global num_found_g, num_missed_g, num_fakes_g, pdf
    global found_2_truth_energies_predicted_sum, found_2_truth_energies_truth_sum
    global found_2_truth_rotational_distance
    global found_2_truth_predicted_energies
    global found_2_truth_target_energies
    global num_rechits_per_shower, num_rechits_per_segment

    truth_energies = np.concatenate(truth_energies, axis=0)
    truth_energies[truth_energies>250] = 300

    found_truth_energies_energies_truth = np.array(found_truth_energies_energies_truth)
    found_truth_energies_is_it_found = np.array(found_truth_energies_is_it_found)

    e_bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

    centers = []
    mean = []

    print(found_truth_energies_is_it_found)
    print(found_truth_energies_energies_truth)
    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies = np.argwhere(np.logical_and(found_truth_energies_energies_truth > l, found_truth_energies_energies_truth < h))

        filtered_found = found_truth_energies_is_it_found[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        centers.append(l+5)

    plt.figure()
    plt.hist(truth_energies, bins=50, histtype='step')
    plt.xlabel("Truth shower energy")
    plt.ylabel("Frequency")
    plt.title('Truth energies')
    pdf.savefig()

    found_2_truth_energies_predicted_sum = np.array(found_2_truth_energies_predicted_sum)
    found_2_truth_energies_truth_sum = np.array(found_2_truth_energies_truth_sum)
    found_2_truth_target_energies = np.array(found_2_truth_target_energies)
    found_2_truth_predicted_energies = np.array(found_2_truth_predicted_energies)

    response_rechit_sum_energy = found_2_truth_energies_predicted_sum/found_2_truth_energies_truth_sum
    response_rechit_sum_energy[response_rechit_sum_energy > 3] = 3

    response_energy_predicted = found_2_truth_predicted_energies/found_2_truth_target_energies
    response_energy_predicted[response_energy_predicted > 3] = 3
    response_energy_predicted[response_energy_predicted < 0.1] = 0.1


    data_dict = {}
    plt.figure()
    plt.hist(response_rechit_sum_energy, bins=20, histtype='step')
    plt.hist(response_energy_predicted, bins=20, histtype='step')
    plt.legend(['predicted shower sum / truth shower sum', 'predicted energy / target energy'])
    plt.xlabel("Predicted/truth")
    plt.ylabel("Frequency")
    plt.title('Response curves')
    pdf.savefig()

    data_dict['response_rechit_sum_energy'] = response_rechit_sum_energy

    found_2_truth_rotational_distance = np.array(found_2_truth_rotational_distance)
    found_2_truth_rotational_distance[found_2_truth_rotational_distance>0.2] = 0.2


    plt.figure()
    plt.hist(found_2_truth_rotational_distance, bins=20, histtype='step')
    plt.xlabel("Rotational distance between true and predicted eta/phi coordinates")
    plt.ylabel("Frequency")
    plt.title('Positional performance')
    pdf.savefig()


    data_dict['positional_performance'] = found_2_truth_rotational_distance


    print(this_energies, centers)
    plt.figure()
    plt.plot(centers, mean, linewidth=0.7, marker='o')
    plt.xticks(centers)



    data_dict['foenergy'] = (centers, mean)

    plt.xlabel('Shower energy')
    plt.ylabel('% found')
    plt.title('Function of energy')

    pdf.savefig()


    #
    # 0/0
    # print(num_real_showers, num_predicted_showers)



    plt.figure()
    plt.hist(num_real_showers_g, bins=np.arange(0,50), histtype='step')
    plt.hist(num_predicted_showers_g, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers')
    plt.ylabel('Frequency')
    plt.legend(['Real showers','Predicted showers'])
    plt.title('Histogram of predicted/real number of showers')
    pdf.savefig()

    data_dict['num_real_showers'] = num_real_showers_g
    data_dict['num_predicted_showers'] = num_predicted_showers_g


    plt.figure()
    plt.hist(num_rechits_per_shower, histtype='step')
    plt.xlabel('Num rechits per shower')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per window','Num rechits per shower'])
    plt.title('Distribution of number of rechits')
    pdf.savefig()

    plt.figure()
    plt.hist(num_rechits_per_segment, histtype='step')
    plt.xlabel('Num rechits per segment')
    plt.ylabel('Frequency')
    # plt.legend(['Num rechits per window','Num rechits per shower'])
    plt.title('Distribution of number of rechits')
    pdf.savefig()


    plt.figure()
    plt.hist(num_real_showers_g, bins=np.arange(0,50), histtype='step')
    # plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
    plt.xlabel('Num showers per window cut')
    plt.ylabel('Frequency')
    # plt.legend(['Real showers','Predicted showers'])
    plt.title('Distribution of number of showers')
    pdf.savefig()



    print(num_found_g)
    print(num_missed_g)
    print(num_fakes_g)


    num_found_g = np.array(num_found_g)
    num_missed_g = np.array(num_missed_g)
    num_fakes_g = np.array(num_fakes_g)


    plt.figure()
    plt.hist(num_found_g, bins=30, histtype='step')
    plt.hist(num_missed_g, bins=30, histtype='step')
    plt.hist(num_fakes_g, bins=30, histtype='step')
    plt.hist(num_real_showers_g, bins=30, histtype='step')
    plt.xlabel('Num showers')
    plt.ylabel('Frequency')
    plt.legend(['Found','Missed', 'Fakes', 'Real number of showers'])
    plt.title('Histogram of found/missed/fakes')
    pdf.savefig()



    bins = np.linspace(0,1,11)
    num_real_showers_g = np.array(num_real_showers_g, np.float)
    num_predicted_showers_g = np.array(num_predicted_showers_g, np.float)
    plt.figure()

    fake_fraction = num_fakes_g / num_predicted_showers_g
    fake_fraction[fake_fraction>1.5] = 1.5
    plt.hist(num_found_g/num_real_showers_g, bins=30, range=(0,1.1),histtype='step')
    plt.hist(num_missed_g/num_real_showers_g, bins=30, range=(0,1.1),histtype='step')
    plt.hist(fake_fraction, bins=30, range=(0,1.1),histtype='step')
    plt.xlabel('(Num found/missed/fakes) / Total number of showers')
    plt.ylabel('Frequency')
    plt.legend(['Found','Missed', 'Fakes'])
    pdf.savefig()



    x_num_real = []
    x_fraction_found = []
    x_fraction_missed = []
    x_fraction_fakes = []


    y_fraction_found = []
    y_fraction_missed = []
    y_fraction_fakes = []


    f_found_g = num_found_g/num_real_showers_g
    f_missed_g = num_missed_g/num_real_showers_g
    f_fakes_g = num_fakes_g/num_predicted_showers_g

    for i in np.unique(num_real_showers_g):
        if i<=0:
            continue

        print("XYZ", i, np.mean(f_found_g[num_real_showers_g==i]))
        print("ABC", i, np.mean(f_missed_g[num_real_showers_g==i]))
        print("DEF", i, np.mean(f_fakes_g[num_real_showers_g==i]))

        x_num_real.append(i)
        x_fraction_found.append(np.mean(f_found_g[num_real_showers_g==i]))
        x_fraction_missed.append(np.mean(f_missed_g[num_real_showers_g==i]))
        x_fraction_fakes.append(np.mean(f_fakes_g[num_real_showers_g==i]))


        y_fraction_found.append(np.var(f_found_g[num_real_showers_g==i]))
        y_fraction_missed.append(np.var(f_missed_g[num_real_showers_g==i]))
        y_fraction_fakes.append(np.var(f_fakes_g[num_real_showers_g==i]))

    x_num_real = np.array(x_num_real)
    x_fraction_found = np.array(x_fraction_found)
    x_fraction_missed = np.array(x_fraction_missed)
    x_fraction_fakes = np.array(x_fraction_fakes)

    y_fraction_found = np.array(y_fraction_found)
    y_fraction_missed = np.array(y_fraction_missed)
    y_fraction_fakes = np.array(y_fraction_fakes)


    order = np.argsort(x_num_real)
    x_num_real = x_num_real[order]

    x_fraction_found = x_fraction_found[order]
    x_fraction_missed = x_fraction_missed[order]
    x_fraction_fakes = x_fraction_fakes[order]


    y_fraction_found = y_fraction_found[order]
    y_fraction_missed = y_fraction_missed[order]
    y_fraction_fakes = y_fraction_fakes[order]


    print(x_fraction_found, x_fraction_missed, x_fraction_fakes)
    print(y_fraction_found, y_fraction_missed, y_fraction_fakes)



    plt.figure()

    x_fraction_fakes[x_fraction_fakes>1.4] = 1.4

    plt.plot(x_num_real, x_fraction_found, linewidth=0.7, marker='o')
    plt.plot(x_num_real, x_fraction_fakes, linewidth=0.7, marker='o')
    plt.xlabel('Num showers')
    plt.ylabel('Fraction (mean)')
    plt.legend(['Found','Fakes'])
    pdf.savefig()


    data_dict['x_num_real_showers']  = x_num_real
    data_dict['fraction_found']  = x_fraction_found
    data_dict['fraction_missed']  = x_fraction_missed
    data_dict['fraction_fakes']  = x_fraction_fakes



    plt.figure()
    plt.plot(x_num_real, y_fraction_found, linewidth=0.7, marker='o')
    plt.plot(x_num_real, y_fraction_fakes, linewidth=0.7, marker='o')
    plt.xlabel('Num showers')
    plt.ylabel('Fraction (variance)')
    plt.legend(['Found','Missed', 'Fakes'])
    pdf.savefig()


    if len(dumppath) > 0:
        print("Dumping")
        with open(dumppath, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        print("DIDNT DUMP")

def main(files, pdfpath, dumppath):


    for file in files:
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'])

    make_plots(pdfpath, dumppath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('output', help='Output directory with .bin.gz files (all will be analysed) or a text file containing lest of those which are to be analysed')
    parser.add_argument('-p', help='Path of the pdf file. Otherwise will be produced in the output directory.', default='')
    parser.add_argument('-b', help='Beta threshold', default='0.1')
    parser.add_argument('-d', help='Distance threshold', default='0.5')
    parser.add_argument('-v', help='Visualize number of showers', default='10')
    parser.add_argument('--datadumppath', help='Data dump in form of numpy arrays', default='')

    parser.add_argument('--gpu', help='GPU', default='')
    args = parser.parse_args()

    # DJCSetGPUs(args.gpu)
    num_segments_to_visualize = int(args.v)

    beta_threshold = beta_threshold*0 + float(args.b)
    distance_threshold = distance_threshold*0 + float(args.d)

    files_to_be_tested = []
    pdfpath = ''
    if os.path.isdir(args.output):
        for x in os.listdir(args.output):
            if x.endswith('.bin.gz'):
                files_to_be_tested.append(os.path.join(args.output, x))
        pdfpath = args.output
    elif os.path.isfile(args.output):
        with open(args.output) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        files_to_be_tested = [x.strip() for x in content]
        pdfpath = os.path.split(pdfpath)[0]
    else:
        raise Exception('Error: couldn\'t locate output folder/file')

    print(files_to_be_tested)
    pdfpath = os.path.join(pdfpath, 'plots.pdf')
    if len(args.p) != 0:
        pdfpath = args.p


    pdf = PdfPages(pdfpath)
    main(files_to_be_tested, pdfpath, args.datadumppath)
    pdf.close()



