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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# from tensorflow.keras.optimizer_v2 import Adam

from ragged_callbacks import plotEventDuringTraining
from DeepJetCore.DJCLayers import ScalarMultiply


# tf.compat.v1.disable_eager_execution()





ragged_constructor = RaggedConstructTensor()


num_real_showers = []
num_predicted_showers = []

num_found_g = []
num_missed_g = []
num_fakes_g = []

iii = 0


def find_uniques_from_betas(betas, coords, dist_threshold):

    n2_distances = np.sqrt(np.sum(np.abs(np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)), axis=-1))
    betas_checked = np.zeros_like(betas) - 1

    index = 0

    arange_vector = np.arange(len(betas))

    while True:
        betas_remaining = betas[betas_checked==-1]
        arange_remaining = arange_vector[betas_checked==-1]

        if len(betas_remaining)==0:
            break

        max_beta = arange_remaining[np.argmax(betas_remaining)]


        n2 = n2_distances[max_beta]

        distances_less = np.logical_and(n2<dist_threshold, betas_checked==-1)
        betas_checked[distances_less] = index

        index += 1


    return betas_checked


found_truth_energies_energies = []
found_truth_energies_found = []
def analyse_one_window_cut(classes_this_segment, x_this_segment, y_this_segment, pred_this_segment):
    global  iii, num_predicted_showers, num_real_showers, num_fakes_g, num_found_g, num_missed_g, found_truth_energies_energies, found_truth_energies_found

    iii+=1
    uniques = tf.unique(classes_this_segment)[0].numpy()
    truth_energies_this_segment = y_this_segment[:, 1]


    beta_all = pred_this_segment[:, -6]
    is_spectator = np.logical_not(y_this_segment[:, 14])
    is_spectator = np.logical_and(is_spectator, beta_all>0.1)
    is_spectator = np.logical_and(is_spectator, classes_this_segment>=0)


    beta_all_filtered = beta_all[is_spectator==1]
    y_all_filtered = y_this_segment[is_spectator==1]


    clustering_coords_all = pred_this_segment[:, -2:]
    clustering_coords_all_filtered = clustering_coords_all[is_spectator==1, :]

    labels = find_uniques_from_betas(beta_all_filtered, clustering_coords_all_filtered, dist_threshold=0.8)
    num_showers_this_segment = len(np.unique(classes_this_segment))
    # print(num_showers_this_segment, len(np.unique(labels)))

    classes_all_filtered = classes_this_segment[is_spectator==1]


    predicted_showers_this_segment = len(np.unique(labels))
    num_predicted_showers.append(predicted_showers_this_segment)

    num_real_showers.append(num_showers_this_segment)


    unique_labels = np.unique(labels)
    unique_classes = np.unique(classes_all_filtered)


    found = dict()
    found_energies = dict()
    unique_classes_this_segment, indices = np.unique(classes_this_segment, return_index=True)
    # for x in unique_classes_this_segment:
    for i in range(len(unique_classes_this_segment)):
        x = unique_classes_this_segment[i]
        found[x]=False
        found_energies[x] = truth_energies_this_segment[indices[i]]


    found_predicted = dict()
    for x in np.unique(unique_labels):
        found_predicted[x]=False

    num_fakes = 0
    num_found = 0
    num_missed = 0

    for x in unique_classes:
        classes = classes_all_filtered[classes_all_filtered==x]
        betas = beta_all_filtered[classes_all_filtered==x]


        this_class = classes[np.argmax(betas)]
        label_max = labels[classes_all_filtered==x][np.argmax(betas)]
        # y_this = y_all_filtered[np.argmax(betas)]
        # print(y_this[:, 1], [])

        if found[this_class]==True:
            pass
        else:
            found[this_class]=True
            found_predicted[label_max]=True
            num_found+=1

    for x in unique_labels:
        # labels_x = labels[labels==x]
        # betas = beta_all_filtered[labels==x]
        #
        # this_class = classes[np.argmax(betas)]

        if found_predicted[x]==True:
            pass
        else:
            num_fakes+=1


    for k,v in found.items():
        found_truth_energies_energies.append(found_energies[k])
        if v==False:
            found_truth_energies_found.append(False)
            num_missed += 1
        else:
            found_truth_energies_found.append(True)

    # print(num_found, num_missed, num_fakes)

    num_found_g.append(num_found)
    num_missed_g.append(num_missed)
    num_fakes_g.append(num_fakes)

    print(num_showers_this_segment, num_found, num_missed, num_fakes, predicted_showers_this_segment)

    # if iii == 11:
    #     plt.scatter(clustering_coords_all_filtered[:, 0], clustering_coords_all_filtered[:, 1], c=classes_all_filtered, cmap='hsv')
    #     plt.savefig('clustering_coords.png')
    #     0/0

    return






truth_energies = []
def analyse_one_file(features, predictions, truth):
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

    i += 1





def make_plots(pdfpath):
    global truth_energies, found_truth_energies_found, found_truth_energies_energies
    global  num_real_showers, num_predicted_showers
    global  num_found_g, num_missed_g, num_fakes_g

    truth_energies = np.concatenate(truth_energies, axis=0)
    truth_energies[truth_energies>250] = 300

    found_truth_energies_energies = np.array(found_truth_energies_energies)
    found_truth_energies_found = np.array(found_truth_energies_found)

    e_bins = [0,10,20,30,40,50,60,70,80,90,200]

    centers = []
    mean = []

    print(found_truth_energies_found)
    print(found_truth_energies_energies)
    for i in range(len(e_bins)-1):
        l = e_bins[i]
        h = e_bins[i+1]

        this_energies = np.argwhere(np.logical_and(found_truth_energies_energies>l, found_truth_energies_energies<h))

        filtered_found = found_truth_energies_found[this_energies].astype(np.float)
        m = np.mean(filtered_found)
        mean.append(m)
        centers.append(l+5)

    with PdfPages(pdfpath) as pdf:
        plt.figure()
        plt.hist(truth_energies, bins=50, histtype='step')
        plt.xlabel("Truth shower energy")
        plt.ylabel("Frequency")
        plt.title('Truth energies')
        pdf.savefig()

        print(this_energies, centers)
        plt.figure()
        plt.plot(centers, mean, linewidth=0.7, marker='o')
        plt.xticks(centers)

        plt.xlabel('Shower energy')
        plt.ylabel('% found')
        plt.title('Function of energy')

        pdf.savefig()


        #
        # 0/0
        # print(num_real_showers, num_predicted_showers)


        plt.figure()
        plt.hist(num_real_showers, bins=np.arange(0,50), histtype='step')
        plt.hist(num_predicted_showers, bins=np.arange(0,70), histtype='step')
        plt.xlabel('Num showers')
        plt.ylabel('Frequency')
        plt.legend(['Real showers','Predicted showers'])
        plt.title('Histogram of predicted/real number of showers')
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
        plt.hist(num_real_showers, bins=30, histtype='step')
        plt.xlabel('Num showers')
        plt.ylabel('Frequency')
        plt.legend(['Found','Missed', 'Fakes', 'Real number of showers'])
        plt.title('Histogram of found/missed/fakes')
        pdf.savefig()



        bins = np.linspace(0,1,11)
        num_real_showers = np.array(num_real_showers, np.float)
        plt.figure()
        plt.hist(num_found_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
        plt.hist(num_missed_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
        plt.hist(num_fakes_g/num_real_showers, bins=30, range=(0,1.1),histtype='step')
        plt.xlabel('(Num found/missed/fakes) / Total showers')
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


        f_found_g = num_found_g/num_real_showers
        f_missed_g = num_missed_g/num_real_showers
        f_fakes_g = num_fakes_g/num_real_showers

        for i in np.unique(num_real_showers):
            if i<=0:
                continue

            print("XYZ", i, np.mean(f_found_g[num_real_showers==i]))
            print("ABC", i, np.mean(f_missed_g[num_real_showers==i]))
            print("DEF", i, np.mean(f_fakes_g[num_real_showers==i]))

            x_num_real.append(i)
            x_fraction_found.append(np.mean(f_found_g[num_real_showers==i]))
            x_fraction_missed.append(np.mean(f_missed_g[num_real_showers==i]))
            x_fraction_fakes.append(np.mean(f_fakes_g[num_real_showers==i]))


            y_fraction_found.append(np.var(f_found_g[num_real_showers==i]))
            y_fraction_missed.append(np.var(f_missed_g[num_real_showers==i]))
            y_fraction_fakes.append(np.var(f_fakes_g[num_real_showers==i]))

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
        plt.plot(x_num_real, x_fraction_found, linewidth=0.7, marker='o')
        plt.plot(x_num_real, x_fraction_missed, linewidth=0.7, marker='o')
        plt.plot(x_num_real, x_fraction_fakes, linewidth=0.7, marker='o')
        plt.xlabel('Num showers')
        plt.ylabel('Fraction (mean)')
        plt.legend(['Found','Missed', 'Fakes'])
        pdf.savefig()



        plt.figure()
        plt.plot(x_num_real, y_fraction_found, linewidth=0.7, marker='o')
        plt.plot(x_num_real, y_fraction_missed, linewidth=0.7, marker='o')
        plt.plot(x_num_real, y_fraction_fakes, linewidth=0.7, marker='o')
        plt.xlabel('Num showers')
        plt.ylabel('Fraction (variance)')
        plt.legend(['Found','Missed', 'Fakes'])
        pdf.savefig()


def main(files, pdfpath):
    for file in files:
        with gzip.open(file, 'rb') as f:
            data_dict = pickle.load(f)
            analyse_one_file(data_dict['features'], data_dict['predicted'], data_dict['truth'])

    make_plots(pdfpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('output', help='Output directory with .bin.gz files (all will be analysed) or a text file containing lest of those which are to be analysed')
    parser.add_argument('-p', help='Path of the pdf file. Otherwise will be produced in the output directory.', default='')
    args = parser.parse_args()

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
    main(files_to_be_tested, pdfpath)
