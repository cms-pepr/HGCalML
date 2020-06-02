# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import time




def remove_zero_length_elements_from_ragged_tensors(row_splits):
    lengths = row_splits[1:] - row_splits[:-1]
    row_splits = tf.concat(([0], tf.cumsum(tf.gather_nd(lengths, tf.where(tf.not_equal(lengths, 0))))), axis=0)
    return row_splits



# create a few with fixed number and then make an if statement selecting the right one
# @tf.function


# bucket this guy with padded inputs maybe?
# @tf.function
def _parametrised_instance_loop(max_instances,
                                instance_ids,
                                no_noise_mask,
                                x_s,
                                classes_s,
                                beta_s,
                                q_s,
                                extra_beta_weights=None,
                                no_beta_norm=False,
                                payload_loss=None):
    # get an idea of all the shapes

    # K      print('instance_ids',tf.shape(instance_ids))
    # V x 2  print('x_s',tf.shape(x_s))
    # V      print('classes_s',tf.shape(classes_s))
    # V      print('beta_s',tf.shape(beta_s))
    # 0      print('num_vertices',tf.shape(num_vertices))
    # V      print('q_s',tf.shape(q_s))
    # V      no_noise_mask

    # move to convention of at least one feature axis

    def gather_for_obj_from_vert(v_prop, ids):
        return tf.gather_nd(tf.tile(tf.expand_dims(v_prop, axis=0), [kalpha.shape[0], 1, 1]), ids, batch_dims=1)

    # instance ids > 0, do not include noise
    # classes_s is 0 for noise
    # create Mki: V x K matrix
    M = tf.expand_dims(instance_ids, axis=1) - tf.expand_dims(classes_s, axis=0)
    M = tf.where(tf.abs(M) < 0.1, tf.zeros_like(M) + 1., tf.zeros_like(M))
    # K x V
    # print('M',M.shape)
    #
    # Here add payload losses. they should already be in a per-vertex loss form, so V, containing a sum of all loss terms
    #
    
    #
    #
    #
    # if padding is applied, otherwise it's clear that it's an object
    is_obj_k = tf.reduce_max(M, axis=1)
    # K
    # print('is_obj_k',is_obj_k.shape)
    Nobj = tf.reduce_sum(is_obj_k, axis=0)
    # print('Nobj',Nobj.shape)
    
    full_pll = None
    if payload_loss is not None:
        payload_loss = tf.expand_dims(payload_loss, axis=0)
        per_obj_pll = M * payload_loss
        per_obj_beta = M * tf.math.atanh(tf.expand_dims(beta_s,axis=0))**2
        per_obj_pll *= per_obj_beta
        per_obj_beta_sum = tf.reduce_sum(per_obj_beta, axis=1) + 1e-6
        per_obj_pll_sum = tf.reduce_sum(per_obj_pll, axis=1)
        full_pll = tf.reduce_sum(per_obj_pll_sum/per_obj_beta_sum, axis=0)
        full_pll /= (Nobj + 1e-3)

    Ntotal = tf.cast(tf.shape(beta_s)[0], dtype='float32')

    kalpha = tf.argmax(M * tf.expand_dims(no_noise_mask * beta_s, axis=0), axis=1)
    kalpha = tf.expand_dims(kalpha, axis=1)
    # K x 1
    # print('kalpha',kalpha.shape)

    # gather everything
    q_kalpha = gather_for_obj_from_vert(tf.expand_dims(q_s, axis=1),
                                        kalpha)  # tf.gather_nd(tf.tile(tf.expand_dims(q_s,axis=0),[Nobj,1]) ,kalpha,batch_dims=1) # K x 1
    # q_kalpha = tf.expand_dims(q_kalpha, axis=1) # K x 1 x 1
    # print('q_kalpha',q_kalpha.shape)

    beta_kalpha = gather_for_obj_from_vert(tf.expand_dims(beta_s, axis=1), kalpha)
    beta_kalpha = tf.expand_dims(beta_kalpha, axis=1)  # K x 1 x 1
    # print('beta_kalpha',beta_kalpha.shape)

    x_kalpha = gather_for_obj_from_vert(x_s, kalpha)
    x_kalpha = tf.expand_dims(x_kalpha, axis=1)  # K x 1 x 2
    # print('x_kalpha',x_kalpha.shape)
    x_s = tf.expand_dims(x_s, axis=0)
    # print('x_s',x_s.shape)

    distance = tf.sqrt(tf.reduce_sum((x_kalpha - x_s) ** 2, axis=-1) + 1e-6)  # K x V , d (tf.sqrt(0)) problem
    # print('distance',distance)

    F_att = q_kalpha * tf.expand_dims(q_s, axis=0) * distance ** 2 * M  # K x V
    # print('F_att',F_att.shape)
    F_att = is_obj_k * tf.reduce_sum(F_att, axis=1)  # K
    # print('F_att',F_att.shape)
    L_att = tf.reduce_sum(F_att, axis=0) / (Ntotal + 1e-6)
    # print('L_att',L_att.shape)

    F_rep = q_kalpha * tf.expand_dims(q_s, axis=0) * tf.nn.relu(1. - distance) * (1. - M)  # K x V
    F_rep = is_obj_k * tf.reduce_sum(F_rep, axis=1)  # K
    L_rep = tf.reduce_sum(F_rep, axis=0) / (Ntotal + 1e-6)

    weights_ka = 1.
    if extra_beta_weights is not None:
        weights_ka = gather_for_obj_from_vert(tf.expand_dims(extra_beta_weights,axis=1),kalpha)
        weights_ka = tf.squeeze(weights_ka ,axis=1)

    # check broadcasting here
    L_beta = tf.reduce_sum(weights_ka* is_obj_k * (1 - tf.squeeze(tf.squeeze(beta_kalpha, axis=1), axis=1)))
    if not no_beta_norm:
        L_beta /= (Nobj + 1e-6)

    n_noise = tf.reduce_sum((1. - no_noise_mask))
    L_suppnoise = L_beta*0.
    if n_noise > 0:
        L_suppnoise = tf.reduce_sum((1. - no_noise_mask) * beta_s, axis=0) / (n_noise + 1e-6)

    # print(L_att, L_rep, L_beta, L_suppnoise)
    # return V_att_segment, V_rep_segment, L_beta_f_segment
    return L_att, L_rep, L_beta, L_suppnoise, full_pll


def padded_parallel_instance_loop(instance_ids,
                                  no_noise_mask,
                                  x_s,
                                  classes_s,
                                  beta_s,
                                  q_s):
    # check length, pad add to predefined tf.function
    # get an idea of all the shapes

    # K      print('instance_ids',tf.shape(instance_ids))
    # V x 2  print('x_s',tf.shape(x_s))
    # V      print('classes_s',tf.shape(classes_s))
    # V      print('beta_s',tf.shape(beta_s))
    # V      print('q_s',tf.shape(q_s))
    # V      no_noise_mask

    # we just need to pad instance_ids with -1
    #
    #  still doesn't allow for tf function unfortunately

    return _parametrised_instance_loop(tf.shape(instance_ids)[0], instance_ids, no_noise_mask, x_s, classes_s, beta_s,
                                       q_s)

    for count in [20, 40, 80, 100, 200]:
        if (tf.shape(instance_ids)[0] <= count):
            if (tf.shape(instance_ids)[0] < count):
                instance_ids = tf.pad(instance_ids, [[0, 20 - tf.shape(instance_ids)[0]]], mode='CONSTANT',
                                      constant_values=-1)
            maxinst = tf.convert_to_tensor(count, dtype=tf.int64)
            return _parametrised_instance_loop(maxinst, instance_ids, no_noise_mask, x_s, classes_s, beta_s, q_s)

    return _parametrised_instance_loop(tf.shape(instance_ids)[0], instance_ids, no_noise_mask, x_s, classes_s, beta_s,
                                       q_s)


counter = 0


def indiv_object_condensation_loss_2(output_space, beta_values, labels_classes, row_splits, spectators, 
                                     payload_loss,
                                     Q_MIN=0.1, S_B=1,
                                     energyweights=None,
                                     no_beta_norm=False):
    """
    ####################################################################################################################
    # Implements OBJECT CONDENSATION for ragged tensors
    # For more, look into the paper
    # https://arxiv.org/pdf/2002.03605.pdf
    ####################################################################################################################

    :param output_space: the clustering space (float32)
    :param beta_values: beta values (float 32)
    :param labels_classes: Labels: -1 for background [0 - num clusters) for foreground clusters
    :param row_splits: row splits to construct ragged tensor such that it separates all the batch elements
    :param Q_MIN: Q_min hyper parameter
    :param S_B: s_b hyper parameter
    :return:
    """
    global counter
    print("started call ", counter, "batch size", len(output_space))

    labels_classes += 1

    batch_size = row_splits.shape[0] - 1

    V_att = tf.constant(0., tf.float32)
    V_rep = tf.constant(0., tf.float32)

    L_beta_f = tf.constant(0., tf.float32)
    L_beta_s = tf.constant(0., tf.float32)

    payload_loss_full = tf.constant(0., tf.float32)
    
    beta_values = tf.clip_by_value(beta_values, 0. + 1e-5, 1. - 1e-5)

    for b in tf.range(batch_size):
        x_s = output_space[row_splits[b]:row_splits[b + 1]]
        classes_s = labels_classes[row_splits[b]:row_splits[b + 1]]
        beta_s = beta_values[row_splits[b]:row_splits[b + 1]]
        payload_loss_seg = payload_loss[row_splits[b]:row_splits[b + 1]]
        
        
        e_weights = 1.
        if energyweights is not None:
            e_weights = energyweights[row_splits[b]:row_splits[b + 1]]
        
        # e_weights not used for now!
        #num_vertices = tf.cast(row_splits[b + 1] - row_splits[b], tf.float32)

        spectators_s = spectators[row_splits[b]:row_splits[b + 1]]

        # Now filter TODO: Test
        x_s = x_s[spectators_s==0]
        classes_s = classes_s[spectators_s==0]
        beta_s = beta_s[spectators_s==0]
        payload_loss_seg = payload_loss_seg[spectators_s==0]
        if energyweights is not None:
            e_weights = e_weights[spectators_s==0]
            
        if len(x_s) == 0:
            print("Warning >>>>NO TRUTH ASSOCIATED VERTICES IN THIS SEGMENT<<<< (just a warning though)")
            continue


        q_s = e_weights * ( tf.math.atanh(beta_s) ** 2 + Q_MIN ) 
        #scaling per instance! just changes instance weighting towards high energy particles

        instance_ids, _ = tf.unique(tf.reshape(classes_s, (-1,)))

        if len(instance_ids) < 1:
            print("Warning >>>>NO INSTANCES IN WINDOW<<<< (just a warning though)")
            continue

        instance_ids = tf.where(instance_ids < 0.1, tf.zeros_like(instance_ids) - 1., instance_ids)
        # instance_ids = tf.sort(instance_ids) #why?

        no_noise_mask = tf.where(classes_s > 0.1, tf.zeros_like(classes_s) + 1., tf.zeros_like(classes_s))
        # beta_maxs = []

        V_att_segment, V_rep_segment, L_beta_f_segment, L_beta_s_segment, segment_payload_loss = _parametrised_instance_loop(
            tf.shape(instance_ids)[0],
            instance_ids,
            no_noise_mask,
            x_s,
            classes_s,
            beta_s,
            q_s,
            e_weights,
            no_beta_norm,
            payload_loss_seg)

        L_beta_f_segment = tf.where(tf.math.is_nan(L_beta_f_segment), 0., L_beta_f_segment)
        L_beta_s_segment = tf.where(tf.math.is_nan(L_beta_s_segment), 0., L_beta_s_segment)
        
        V_att_segment = tf.where(tf.math.is_nan(V_att_segment), 0., V_att_segment)
        V_rep_segment = tf.where(tf.math.is_nan(V_rep_segment), 0., V_rep_segment)

        L_beta_f += L_beta_f_segment
        L_beta_s += L_beta_s_segment

        V_att += V_att_segment
        V_rep += V_rep_segment
        
        payload_loss_full += tf.where(tf.math.is_nan(segment_payload_loss), 0., segment_payload_loss) 

    batch_size = float(batch_size)
    V_att = V_att / (batch_size + 1e-5)
    V_rep = V_rep / (batch_size + 1e-5)
    L_beta_s = L_beta_s / (batch_size + 1e-5)
    L_beta_f = L_beta_f / (batch_size + 1e-5)

    print("finished call ", counter)
    counter += 1

    return V_att, V_rep, L_beta_s, L_beta_f, payload_loss_full


def object_condensation_loss(output_space, beta_values, labels_classes, row_splits, Q_MIN=0.1, S_B=1):
    V_att, V_rep, L_beta_s, L_beta_f, payload_loss_full = indiv_object_condensation_loss(output_space, beta_values, labels_classes,
                                                                      row_splits, Q_MIN, S_B)

    losses = float(V_att.numpy()), float(V_rep.numpy()), float(L_beta_f.numpy()), float(L_beta_s.numpy())

    return V_att + V_rep + L_beta_s + L_beta_f, losses
