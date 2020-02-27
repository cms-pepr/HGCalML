# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import time



def cumsum_including_last(V):
    return tf.concat(([0],tf.cumsum(V)), axis=0)


def construct_indicing(M, N):
    # Symnbols with ' are ragged dimensions
    # In the following comments, the [] are batch elements. So the first batch element contains 4 vertices and 3 showers
    # The second batch element contains 5 vertices and 2 showers. The third batch element also contains 5 vertices and 2
    # showers.
    # M ∈ Z+^[B, V', S'] - this will be [0,0,0,1,1,1,2,2,2,3,3,3,],[0,0,1,1,2,2,3,3,4,4,5,5,],[0,0,1,1,2,2,3,3,4,4,5,5,]
    # N ∈ Z+^[B, V', S'] - this will be [0,1,2,0,1,2,0,1,2,0,1,2,],[0,1,0,1,0,1,0,1,0,1,0,1,],[0,1,0,1,0,1,0,1,0,1,0,1,]
    # So we have to go from
    # [0,4,10,16,] to these (which is the original row splits) (S)
    # Obviously we also have maximum showers in each batch element (M)
    # [3,2,2,]
    # Also the number of vertices
    # [4,6,6,]
    # Multiply and get this:
    # [12,12,12]
    # or ex sum (S2)
    # [0,12,24,36]

    # Generate sequence to 36 and split by ex sum
    # [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    # [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,],[12,13,14,15,16,17,18,19,20,21,22,23,],[24,25,26,27,28,29,30,31,32,33,34,35]
    # Divide by M
    # [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,],[ 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,],[ 8, 8, 8, 9, 9, 9,10,10,10,11,11,11,]
    # Modulo by M
    # [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,],[ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],[ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,]
    # Voilaa!
    # Now need the batch indexing tensor which should be simple
    # Makes ones vector of length equal to last element of S2 and then use S2 as splits
    # [1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]
    # Multiply by [0,1,2] which is simply range nd to batch elements
    M = tf.cast(M, tf.int32)
    N = tf.cast(N, tf.int32)
    S2 = cumsum_including_last(M*N)
    b = M.shape[0]
    # print(b)

    m = S2[-1]
    sequence_36 = tf.RaggedTensor.from_row_splits(tf.range(0, m, dtype=tf.int32),S2)- tf.cumsum(M*N, exclusive=True)[..., tf.newaxis]


    divided_by_M = tf.math.floordiv(sequence_36,  M[..., tf.newaxis])
    modulo_by_M = tf.math.floormod(sequence_36, M[..., tf.newaxis])

    batch_indexing_tensor = tf.RaggedTensor.from_row_splits(tf.ones(shape=(m,), dtype=tf.int32), S2) * tf.range(0,b, dtype=tf.int32)[..., tf.newaxis]
    # print(batch_indexing_tensor)
    # #
    # # print(sequence_36)
    # print(batch_indexing_tensor)
    # print(divided_by_M)
    # print(modulo_by_M)

    return divided_by_M, modulo_by_M, batch_indexing_tensor



def sort_by_labels_then_beta(output_space, beta_values, labels_classes, row_splits):
    # Sort the output space first by classes and then by beta values (will be useful later)
    value_row_ids = tf.RaggedTensor.from_row_splits(labels_classes, row_splits).value_rowids()
    # TODO: Adjust the values later for now this should be more than fine


    # Have to go to double precision to do proper sorting
    sorting_indices = tf.argsort((1-tf.cast(beta_values, tf.float64))+tf.cast(labels_classes,tf.float64)*100000.+tf.cast(value_row_ids, tf.float64)*9000000)[..., tf.newaxis]
    # print(sorting_indices)

    output_space = tf.gather_nd(output_space, sorting_indices)
    beta_values = tf.gather_nd(beta_values, sorting_indices)
    labels_classes = tf.gather_nd(labels_classes, sorting_indices)

    return output_space, beta_values, labels_classes




def sort_by_labels_then_beta_naive(output_space, beta_values, labels_classes, row_splits):
    labels_classes = tf.cast(labels_classes, tf.int32)


    #### TEST 1 - SORTING ###
    output_space_ragged_unsorted = tf.RaggedTensor.from_row_splits(output_space, row_splits)
    betas_ragged_unsorted = tf.RaggedTensor.from_row_splits(beta_values, row_splits)
    labels_ragged_unsorted = tf.RaggedTensor.from_row_splits(labels_classes, row_splits)

    sorted_output_space = []
    sorted_beta_values = []
    sorted_labels_classes = []
    for i in range(output_space_ragged_unsorted.shape[0]):
        segment_output = output_space_ragged_unsorted[i]
        segment_betas = betas_ragged_unsorted[i]
        segment_labels = labels_ragged_unsorted[i]

        # First sort by labels just in case they are not sorted
        indices = tf.argsort(segment_labels, direction='ASCENDING')[..., tf.newaxis]
        segment_output = tf.gather_nd(segment_output, indices)
        segment_betas = tf.gather_nd(segment_betas, indices)
        segment_labels = tf.gather_nd(segment_labels, indices)

        segment_output_sorted = []
        segment_betas_sorted = []
        segment_labels_sorted = []

        segment_output_ragged = tf.RaggedTensor.from_value_rowids(segment_output, value_rowids=segment_labels)
        segment_betas_ragged = tf.RaggedTensor.from_value_rowids(segment_betas, value_rowids=segment_labels)
        segment_labels_ragged = tf.RaggedTensor.from_value_rowids(segment_labels, value_rowids=segment_labels)
        for j in range(segment_betas_ragged.shape[0]):
            indices = tf.argsort(segment_betas_ragged[j], direction='DESCENDING')[..., tf.newaxis]
            segment_output_sorted.append(tf.gather_nd(segment_output_ragged[j], indices))
            segment_betas_sorted.append(tf.gather_nd(segment_betas_ragged[j], indices))
            segment_labels_sorted.append(tf.gather_nd(segment_labels_ragged[j], indices))

        segment_output_sorted = tf.concat(segment_output_sorted, axis=0)
        segment_betas_sorted = tf.concat(segment_betas_sorted, axis=0)
        segment_labels_sorted = tf.concat(segment_labels_sorted, axis=0)

        sorted_output_space.append(segment_output_sorted)
        sorted_beta_values.append(segment_betas_sorted)
        sorted_labels_classes.append(segment_labels_sorted)

    sorted_output_space = tf.concat(sorted_output_space, axis=0)
    sorted_beta_values = tf.concat(sorted_beta_values, axis=0)
    sorted_labels_classes = tf.concat(sorted_labels_classes, axis=0)
    sorted_labels_classes = tf.cast(sorted_labels_classes, tf.float32)


    return sorted_output_space, sorted_beta_values, sorted_labels_classes



def evaluate_loss_with_test_case(output_space, beta_values, labels_classes, row_splits, Q_MIN=0.1, S_B=1):

    """
    ####################################################################################################################
    # Implements OBJECT CONDENSATION for ragged tensors
    # For more, look into the paper
    # https://arxiv.org/pdf/2002.03605.pdf
    ####################################################################################################################

    In this implementation:
    1. segment refers to one batch element i.e. one batch element might consist of multiple showers
    2. batch size is number of segments or batch size which is denoted by B


    :param output_space: the clustering space (float32)
    :param beta_values: beta values (float 32)
    :param labels_classes: Labels: -1 for background [0 - num clusters) for foreground clusters
    :param row_splits: row splits to construct ragged tensor such that it separates all the batch elements
    :param Q_MIN: Q_min hyper parameter
    :param S_B: s_b hyper parameter
    :return:
    """

    # Initially, the background is -1 we are just making it 0 instead
    labels_classes = labels_classes +1

    output_space_n, beta_values_n, labels_classes_n = sort_by_labels_then_beta_naive(output_space, beta_values, labels_classes, row_splits)
    output_space, beta_values, labels_classes = sort_by_labels_then_beta(output_space, beta_values, labels_classes, row_splits)

    x = tf.norm(output_space_n-output_space) + tf.norm(beta_values_n-beta_values) + tf.norm(labels_classes_n-labels_classes)
    assert float(x) < 0.00001


    # Separate all the segments in a ragged format
    output_space_ragged_segments_only = tf.RaggedTensor.from_row_splits(output_space, row_splits)
    beta_values_ragged_segments_only = tf.RaggedTensor.from_row_splits(beta_values, row_splits)
    labels_classes_ragged_segments_only = tf.RaggedTensor.from_row_splits(labels_classes, row_splits)

    # The segment below constructs two row splits so we can split by segments and then by clusters
    # Maximum number of clusters in each of the segments ∈ Z+^B
    M = tf.reduce_max(labels_classes_ragged_segments_only, axis=-1)+1
    running_sum_M_exclusive_without_last_element = tf.math.cumsum(M, exclusive=True)
    row_splits_inner = (labels_classes_ragged_segments_only + running_sum_M_exclusive_without_last_element[..., tf.newaxis]).values
    row_splits_inner = tf.cast(row_splits_inner, tf.int32)
    sum_of_clusters_in_batch = tf.reduce_sum(M)
    row_splits_outer = tf.concat((running_sum_M_exclusive_without_last_element, [sum_of_clusters_in_batch]), axis=0)
    row_splits_outer = tf.cast(row_splits_outer, tf.int32)

    beta_values_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(beta_values, row_splits_inner)
    beta_values_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(beta_values_ragged_segments_and_clusters, row_splits=row_splits_outer)
    output_space_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(output_space, row_splits_inner)
    output_space_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(output_space_ragged_segments_and_clusters, row_splits=row_splits_outer)
    labels_classes_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(labels_classes, row_splits_inner)
    labels_classes_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(labels_classes_ragged_segments_and_clusters, row_splits=row_splits_outer)



    # This is used to pick first elements for each cluster of each segment using gather nd (which will have highest
    #  beta values)
    max_values_indices_0 = beta_values_ragged_segments_and_clusters.value_rowids()
    max_values_indices_1 = (tf.RaggedTensor.from_row_splits(tf.range(0, sum_of_clusters_in_batch), row_splits=row_splits_outer) - running_sum_M_exclusive_without_last_element[..., tf.newaxis]).values
    max_values_indices_1 = tf.cast(max_values_indices_1, tf.int32)
    max_values_indices_2 = tf.zeros_like(max_values_indices_1, dtype=tf.int32)
    max_values_indices = tf.concat((max_values_indices_0[..., tf.newaxis], max_values_indices_1[..., tf.newaxis], max_values_indices_2[..., tf.newaxis]), axis=-1)

    # Collect the x(s) with highest beta value from each cluster
    highest_beta_x_ragged = tf.gather_nd(output_space_ragged_segments_and_clusters, max_values_indices)
    highest_beta_x_ragged = tf.RaggedTensor.from_row_splits(highest_beta_x_ragged, row_splits_outer)

    # Convert betas to charge
    charge_values_ragged_segments_and_clusters = tf.pow(tf.math.atan(beta_values_ragged_segments_and_clusters), 2) + Q_MIN
    charge_values_ragged_batch_segments = tf.RaggedTensor.from_row_splits(charge_values_ragged_segments_and_clusters.values.values, row_splits)


    # N is number of vertices in each segment
    N = row_splits[1:] - row_splits[:-1]
    M = tf.cast(M, tf.int32)

    attractive_loss, repulsive_loss = find_loss_values_naive(N, beta_values_ragged_segments_and_clusters, output_space_ragged_segments_and_clusters, labels_classes_ragged_segments_and_clusters, charge_values_ragged_segments_and_clusters)

    # Collect the highest beta value from each cluster
    highest_beta_ragged = tf.gather_nd(beta_values_ragged_segments_and_clusters, max_values_indices)
    highest_beta_ragged = tf.RaggedTensor.from_row_splits(highest_beta_ragged, row_splits_outer)

    highest_charge_ragged = tf.gather_nd(charge_values_ragged_segments_and_clusters, max_values_indices)
    highest_charge_ragged = tf.RaggedTensor.from_row_splits(highest_charge_ragged, row_splits_outer)

    vertices_indexing_tensor, representative_vertex_indexing_tensor, batch_indexing_tensor = construct_indicing(M, N)
    shower_indexing_tensor_with_batch = tf.concat((batch_indexing_tensor[..., tf.newaxis], vertices_indexing_tensor[..., tf.newaxis]), axis=-1)
    max_indexing_tensor_with_batch = tf.concat((batch_indexing_tensor[..., tf.newaxis], representative_vertex_indexing_tensor[..., tf.newaxis]), axis=-1)

    x_for_V = tf.gather_nd(output_space_ragged_segments_only, shower_indexing_tensor_with_batch)
    q_for_V = tf.gather_nd(charge_values_ragged_batch_segments, shower_indexing_tensor_with_batch)
    cluster_ids_for_V = tf.gather_nd(labels_classes_ragged_segments_only, shower_indexing_tensor_with_batch)
    x_alpha_for_V = tf.gather_nd(highest_beta_x_ragged, max_indexing_tensor_with_batch)
    q_alpha_for_V = tf.gather_nd(highest_charge_ragged, max_indexing_tensor_with_batch)


    # Compute the attractive potential
    V_attractive = tf.reduce_sum(tf.pow(output_space_ragged_segments_and_clusters - tf.expand_dims(highest_beta_x_ragged, axis=2), 2), axis=-1)
    V_attractive = V_attractive * tf.cast(tf.not_equal(labels_classes_ragged_segments_and_clusters,0), tf.float32) * charge_values_ragged_segments_and_clusters
    V_attractive = tf.reduce_sum(V_attractive * highest_charge_ragged[..., tf.newaxis], axis=-1)
    V_attractive = tf.reduce_sum(V_attractive, axis=1) / tf.cast(N, tf.float32)
    V_attractive = tf.reduce_mean(V_attractive)

    # Compute the repulsive potential
    V_repulsive = tf.reduce_sum(tf.pow(x_for_V-x_alpha_for_V, 2), axis=-1)
    V_repulsive = tf.maximum(0, 1 - V_repulsive)
    V_repulsive = V_repulsive * tf.cast(tf.not_equal(representative_vertex_indexing_tensor,0), tf.float32)
    V_repulsive = V_repulsive * q_alpha_for_V
    V_repulsive = q_for_V * V_repulsive * tf.cast(tf.not_equal(tf.cast(representative_vertex_indexing_tensor, tf.float32),cluster_ids_for_V), tf.float32)
    V_repulsive = tf.reduce_sum(V_repulsive, axis=-1) / tf.cast(N, tf.float32)

    V_repulsive = tf.reduce_mean(V_repulsive)

    L_v = V_repulsive + V_attractive

    assert bool(np.isclose(float(V_attractive), float(attractive_loss))) and bool(np.isclose(float(V_repulsive), float(repulsive_loss)))

    L_b_first_term = tf.reduce_mean(tf.reduce_mean(1-highest_beta_ragged, axis=-1), axis=0)

    L_b_second_term = tf.reduce_sum(tf.cast(tf.equal(labels_classes_ragged_segments_only, 0), tf.float32)*beta_values_ragged_segments_only, axis=-1)
    L_b_second_term =  L_b_second_term / tf.reduce_sum(tf.cast(tf.equal(labels_classes_ragged_segments_only, 0), tf.float32), axis=-1)
    L_b_second_term = tf.reduce_mean(L_b_second_term)

    return S_B*L_b_second_term + L_b_first_term + L_v




def find_loss_values_naive(N, beta_values_ragged_segments_and_clusters, output_space_ragged_segments_and_clusters, labels_classes_ragged_segments_and_clusters, charge_values_ragged_segments_and_clusters):

    # Attractive loss
    b = beta_values_ragged_segments_and_clusters.shape[0]


    loss_values = []
    loss_values_repulsive = []

    for i in range(b):
        beta_values_ragged_clusters = beta_values_ragged_segments_and_clusters[i]
        output_space_ragged_clusters = output_space_ragged_segments_and_clusters[i]
        labels_classes_ragged_clusters = labels_classes_ragged_segments_and_clusters[i]
        charge_values_ragged_clusters = charge_values_ragged_segments_and_clusters[i]

        c = beta_values_ragged_clusters.shape[0]

        max_charges = []
        max_x = []

        indices = []

        repulsive_losses = []

        stuff = []
        for k in range(c):
            charqe_alpha_k_index = tf.argmax(beta_values_ragged_clusters[k])
            indices.append(charqe_alpha_k_index)
            max_charges.append(charge_values_ragged_clusters[k, charqe_alpha_k_index])
            max_x.append(output_space_ragged_clusters[k, charqe_alpha_k_index])


            repulsive_loss_k = tf.maximum(0, 1 - tf.reduce_sum(tf.pow(max_x[k][tf.newaxis, ...] - output_space_ragged_clusters.values, 2), -1))

            repulsive_loss_k = max_charges[k][tf.newaxis,...]*repulsive_loss_k
            repulsive_loss_k = charge_values_ragged_clusters.values * repulsive_loss_k * tf.cast(tf.not_equal(labels_classes_ragged_clusters.values, k), tf.float32)
            stuff.append(tf.not_equal(labels_classes_ragged_clusters.values, k)[..., tf.newaxis])

            repulsive_loss_k = repulsive_loss_k * float(k != 0)
            repulsive_loss_k = tf.reduce_sum(repulsive_loss_k)

            repulsive_losses.append(repulsive_loss_k)

        repulsive_losses = tf.convert_to_tensor(repulsive_losses)
        # if i==0:
        #     print("Hello", tf.reduce_sum(repulsive_losses))
        loss_values_repulsive.append(tf.reduce_sum(repulsive_losses))

        max_charges = tf.convert_to_tensor(max_charges)
        max_x = tf.convert_to_tensor(max_x)
        max_x = tf.RaggedTensor.from_row_splits(tf.gather_nd(max_x, output_space_ragged_clusters.value_rowids()[..., tf.newaxis]), output_space_ragged_clusters.row_splits)
        max_charges = tf.RaggedTensor.from_row_splits(tf.gather_nd(max_charges, output_space_ragged_clusters.value_rowids()[..., tf.newaxis]), output_space_ragged_clusters.row_splits)

        loss_value = tf.reduce_sum(tf.pow((max_x - output_space_ragged_clusters), 2), axis=-1)

        loss_value = loss_value * max_charges * charge_values_ragged_clusters
        loss_value = tf.reduce_sum(loss_value, axis=-1)
        loss_value = loss_value * tf.cast(tf.not_equal(tf.range(0,c), 0), tf.float32)
        loss_value = tf.reduce_sum(loss_value)
        loss_value = loss_value / tf.cast(N[i], tf.float32)

        loss_values.append(loss_value)


    loss_values = tf.convert_to_tensor(loss_values)
    loss_values_repulsive = tf.convert_to_tensor(loss_values_repulsive)/ tf.cast(N, tf.float32)

    return tf.reduce_mean(loss_values), tf.reduce_mean(loss_values_repulsive)


def evaluate_loss(output_space, beta_values, labels_classes, row_splits, Q_MIN=0.1, S_B=1):
    """
    ####################################################################################################################
    # Implements OBJECT CONDENSATION for ragged tensors
    # For more, look into the paper
    # https://arxiv.org/pdf/2002.03605.pdf
    ####################################################################################################################

    In this implementation:
    1. segment refers to one batch element i.e. one batch element might consist of multiple showers
    2. batch size is number of segments or batch size which is denoted by B


    :param output_space: the clustering space (float32)
    :param beta_values: beta values (float 32)
    :param labels_classes: Labels: -1 for background [0 - num clusters) for foreground clusters
    :param row_splits: row splits to construct ragged tensor such that it separates all the batch elements
    :param Q_MIN: Q_min hyper parameter
    :param S_B: s_b hyper parameter
    :return:
    """


    labels_classes = tf.cast(labels_classes, tf.int32)

    # Initially, the background is -1 we are just making it 0 instead
    labels_classes = labels_classes +1

    output_space, beta_values, labels_classes = sort_by_labels_then_beta(output_space, beta_values, labels_classes, row_splits)

    # Separate all the segments in a ragged format
    output_space_ragged_segments_only = tf.RaggedTensor.from_row_splits(output_space, row_splits)
    beta_values_ragged_segments_only = tf.RaggedTensor.from_row_splits(beta_values, row_splits)
    labels_classes_ragged_segments_only = tf.RaggedTensor.from_row_splits(labels_classes, row_splits)

    # The segment below constructs two row splits so we can split by segments and then by clusters
    # Maximum number of clusters in each of the segments ∈ Z+^B
    M = tf.reduce_max(labels_classes_ragged_segments_only, axis=-1)+1

    running_sum_M_exclusive_without_last_element = tf.math.cumsum(M, exclusive=True)

    row_splits_inner = (labels_classes_ragged_segments_only + running_sum_M_exclusive_without_last_element[..., tf.newaxis]).values
    row_splits_inner = tf.cast(row_splits_inner, tf.int32)

    sum_of_clusters_in_batch = tf.reduce_sum(M)
    row_splits_outer = tf.concat((running_sum_M_exclusive_without_last_element, [sum_of_clusters_in_batch]), axis=0)
    row_splits_outer = tf.cast(row_splits_outer, tf.int32)

    beta_values_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(beta_values, row_splits_inner)
    beta_values_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(beta_values_ragged_segments_and_clusters, row_splits=row_splits_outer)
    output_space_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(output_space, row_splits_inner)
    output_space_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(output_space_ragged_segments_and_clusters, row_splits=row_splits_outer)
    labels_classes_ragged_segments_and_clusters = tf.RaggedTensor.from_value_rowids(labels_classes, row_splits_inner)
    labels_classes_ragged_segments_and_clusters = tf.RaggedTensor.from_row_splits(labels_classes_ragged_segments_and_clusters, row_splits=row_splits_outer)




    # This is used to pick first elements for each cluster of each segment using gather nd (which will have highest
    #  beta values)
    max_values_indices_0 = beta_values_ragged_segments_and_clusters.value_rowids()
    max_values_indices_1 = (tf.RaggedTensor.from_row_splits(tf.range(0, sum_of_clusters_in_batch), row_splits=row_splits_outer) - running_sum_M_exclusive_without_last_element[..., tf.newaxis]).values
    max_values_indices_1 = tf.cast(max_values_indices_1, tf.int32)
    max_values_indices_2 = tf.zeros_like(max_values_indices_1, dtype=tf.int32)
    max_values_indices = tf.concat((max_values_indices_0[..., tf.newaxis], max_values_indices_1[..., tf.newaxis], max_values_indices_2[..., tf.newaxis]), axis=-1)


    # Collect the x(s) with highest beta value from each cluster
    highest_beta_x_ragged = tf.gather_nd(output_space_ragged_segments_and_clusters, max_values_indices)
    highest_beta_x_ragged = tf.RaggedTensor.from_row_splits(highest_beta_x_ragged, row_splits_outer)

    # Convert betas to charge
    charge_values_ragged_segments_and_clusters = tf.pow(tf.math.atan(beta_values_ragged_segments_and_clusters), 2) + Q_MIN
    charge_values_ragged_batch_segments = tf.RaggedTensor.from_row_splits(charge_values_ragged_segments_and_clusters.values.values, row_splits)


    # N is number of vertices in each segment
    N = row_splits[1:] - row_splits[:-1]
    M = tf.cast(M, tf.int32)

    # Collect the highest beta value from each cluster
    highest_beta_ragged = tf.gather_nd(beta_values_ragged_segments_and_clusters, max_values_indices)
    highest_beta_ragged = tf.RaggedTensor.from_row_splits(highest_beta_ragged, row_splits_outer)

    highest_charge_ragged = tf.gather_nd(charge_values_ragged_segments_and_clusters, max_values_indices)
    highest_charge_ragged = tf.RaggedTensor.from_row_splits(highest_charge_ragged, row_splits_outer)

    vertices_indexing_tensor, representative_vertex_indexing_tensor, batch_indexing_tensor = construct_indicing(M, N)
    shower_indexing_tensor_with_batch = tf.concat((batch_indexing_tensor[..., tf.newaxis], vertices_indexing_tensor[..., tf.newaxis]), axis=-1)
    max_indexing_tensor_with_batch = tf.concat((batch_indexing_tensor[..., tf.newaxis], representative_vertex_indexing_tensor[..., tf.newaxis]), axis=-1)

    shower_indexing_tensor_with_batch = tf.cast(shower_indexing_tensor_with_batch, tf.int64)

    x_for_V = tf.gather_nd(output_space_ragged_segments_only, shower_indexing_tensor_with_batch)
    q_for_V = tf.gather_nd(charge_values_ragged_batch_segments, shower_indexing_tensor_with_batch)
    cs_for_V = tf.gather_nd(labels_classes_ragged_segments_only, shower_indexing_tensor_with_batch)
    cs_for_V = tf.cast(cs_for_V, tf.float32)
    x_alpha_for_V = tf.gather_nd(highest_beta_x_ragged, max_indexing_tensor_with_batch)
    q_alpha_for_V = tf.gather_nd(highest_charge_ragged, max_indexing_tensor_with_batch)


    # Compute the attractive potential
    V_attractive = tf.reduce_sum(tf.pow(output_space_ragged_segments_and_clusters - tf.expand_dims(highest_beta_x_ragged, axis=2), 2), axis=-1)
    V_attractive = V_attractive * tf.cast(tf.not_equal(labels_classes_ragged_segments_and_clusters,0), tf.float32) * charge_values_ragged_segments_and_clusters
    V_attractive = tf.reduce_sum(V_attractive * highest_charge_ragged[..., tf.newaxis], axis=-1)
    V_attractive = tf.reduce_sum(V_attractive, axis=1) / tf.cast(N, tf.float32)
    V_attractive = tf.reduce_mean(V_attractive)

    # Compute the repulsive potential
    V_repulsive = tf.reduce_sum(tf.pow(x_for_V-x_alpha_for_V, 2), axis=-1)
    V_repulsive = tf.maximum(0, 1 - V_repulsive)
    V_repulsive = V_repulsive * tf.cast(tf.not_equal(representative_vertex_indexing_tensor,0), tf.float32)
    V_repulsive = V_repulsive * q_alpha_for_V
    V_repulsive = q_for_V * V_repulsive * tf.cast(tf.not_equal(tf.cast(representative_vertex_indexing_tensor, tf.float32),cs_for_V), tf.float32)
    V_repulsive = tf.reduce_sum(V_repulsive, axis=-1) / tf.cast(N, tf.float32)
    V_repulsive = tf.reduce_mean(V_repulsive)

    L_v = V_repulsive + V_attractive


    # TODO: Fix it to exclude the background
    # TODO: done but it hasn't been fully tested as of yet.
    # Test it again just to make sure
    masque_pour_foreground_showers = tf.cast(tf.RaggedTensor.from_row_splits(max_values_indices_1!=0, row_splits_outer), tf.float32)
    L_b_first_term = tf.reduce_mean(tf.reduce_sum((1-highest_beta_ragged)*masque_pour_foreground_showers, axis=-1)/tf.reduce_sum(masque_pour_foreground_showers), axis=0)


    L_b_second_term = tf.reduce_sum(tf.cast(tf.equal(labels_classes_ragged_segments_only, 0), tf.float32)*beta_values_ragged_segments_only, axis=-1)
    num_noise = tf.reduce_sum(tf.cast(tf.equal(labels_classes_ragged_segments_only, 0), tf.float32), axis=-1)
    L_b_second_term =  tf.cast(tf.greater(num_noise,0), tf.float32) * L_b_second_term / tf.maximum(num_noise, 1)
    L_b_second_term = tf.reduce_mean(L_b_second_term)


    # print(L_b_first_term, L_b_second_term, V_attractive, V_repulsive)

    loss = S_B*L_b_second_term + L_b_first_term + L_v

    parts = (float(L_b_first_term.numpy()), float(L_b_second_term.numpy()), float(V_attractive.numpy()), float(V_repulsive.numpy()))
    return loss, parts


def remove_zero_length_elements_from_ragged_tensors(row_splits):
    lengths = row_splits[1:] - row_splits[:-1]
    row_splits = tf.concat(([0], tf.cumsum(tf.gather_nd(lengths, tf.where(tf.not_equal(lengths, 0))))), axis=0)
    return row_splits
