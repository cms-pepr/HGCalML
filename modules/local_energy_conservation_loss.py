import tensorflow as tf
from oc_helper_ops import SelectWithDefault
from oc_helper_ops import CreateMidx

def get_local_energy_conservation_loss_per_batch_element(
        beta,
        x,
        q_min,
        object_weights,  # V x 1 !!
        truth_idx,
        is_spectator,
        payload_loss,
        hit_energy,
        t_energy,
        pred_energy,
        n_shower_neighbours=3,
        noise_q_min=None,
        distance_scale=None,
        use_mean_x=0.,
        kalpha_damping_strength=0.,
        beta_gradient_damping=0.,
        soft_q_scaling=True,
):
    tf.assert_equal(True, is_spectator >= 0.)
    tf.assert_equal(True, beta >= 0.)

    # set all spectators invalid here, everything scales with beta, so:
    if beta_gradient_damping > 0.:
        beta = beta_gradient_damping * tf.stop_gradient(beta) + (1. - beta_gradient_damping) * beta
    beta_in = beta
    beta = tf.clip_by_value(beta, 0., 1. - 1e-4)
    beta *= (1. - is_spectator)
    qraw = tf.math.atanh(beta) ** 2

    is_noise = tf.where(truth_idx < 0, tf.zeros_like(truth_idx, dtype='float32') + 1., 0.)  # V x 1
    if noise_q_min is not None:
        q_min = (1. - is_noise) * q_min + is_noise * noise_q_min

    if soft_q_scaling:
        qraw = tf.math.atanh(beta_in / 1.002) ** 2  # beta_in**4 *20.
        beta = beta_in * (1. - is_spectator)  # no need for clipping

    q = (qraw + q_min) * (1. - is_spectator)  # V x 1

    N = tf.cast(beta.shape[0], dtype='float32')

    Msel, M_not, N_per_obj = CreateMidx(truth_idx, calc_m_not=True)
    # use eager here
    if Msel is None:
        # V_att, V_rep, Noise_pen, B_pen, pll, too_much_B_pen
        print('>>> WARNING: Event has no objects, only noise! Will return zero loss. <<<')
        zero_tensor = tf.reduce_mean(q, axis=0) * 0.
        zero_payload = tf.reduce_mean(payload_loss, axis=0) * 0.
        return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_payload, zero_tensor

    N_per_obj = tf.cast(N_per_obj, dtype='float32')  # K x 1
    K = tf.cast(Msel.shape[0], dtype='float32')

    padmask_m = SelectWithDefault(Msel, tf.zeros_like(beta_in) + 1., 0.)  # K x V-obj x 1
    x_m = SelectWithDefault(Msel, x, 0.)  # K x V-obj x C
    beta_m = SelectWithDefault(Msel, beta, 0.)  # K x V-obj x 1
    is_spectator_m = SelectWithDefault(Msel, is_spectator, 0.)  # K x V-obj x 1
    q_m = SelectWithDefault(Msel, q, 0.)  # K x V-obj x 1
    object_weights_m = SelectWithDefault(Msel, object_weights, 0.)
    distance_scale_m = SelectWithDefault(Msel, distance_scale, 1.)

    tf.assert_greater(distance_scale_m, 0., message="predicted distances must be greater zero")

    kalpha_m = tf.argmax((1. - is_spectator_m) * beta_m, axis=1)  # K x 1

    x_kalpha_m = tf.gather_nd(x_m, kalpha_m, batch_dims=1)  # K x C
    if use_mean_x > 0:
        x_kalpha_m_m = tf.reduce_sum(q_m * x_m * padmask_m, axis=1)  # K x C
        x_kalpha_m_m = tf.math.divide_no_nan(x_kalpha_m_m, tf.reduce_sum(q_m * padmask_m, axis=1) + 1e-9)
        x_kalpha_m = use_mean_x * x_kalpha_m_m + (1. - use_mean_x) * x_kalpha_m

    if kalpha_damping_strength > 0:
        x_kalpha_m = kalpha_damping_strength * tf.stop_gradient(x_kalpha_m) + (
                1. - kalpha_damping_strength) * x_kalpha_m

    # perform kNN using x_kalpha_m to get closest neighbour matrix
    shower_neighbour_matrix = do_knn(x_kalpha_m, n_shower_neighbours, max_shower_dist) # K x (1 + n_shower_neighbours)

    # calculate TRUTH energy sum of all neighbour TRUTH showers + the shower itself
    truth_obj_hit_e = SelectWithDefault(Msel, t_energy, 0.)  # K x V-obj x 1
    truth_obj_dep_e = tf.reduce_sum(truth_obj_hit_e, axis=1)  # K x 1
    neighbour_shower_energy_matrix_truth = SelectWithDefault(shower_neighbour_matrix, truth_obj_dep_e, 0.) # K x (1 + n_shower_neighbours)
    local_shower_energy_sum_truth = tf.reduce_sum(neighbour_shower_energy_matrix_truth, axis=1)  # K x 1

    # calculate PREDICTED energy sum of all neighbour RECO showers + the shower itself
    pred_obj_hit_e = SelectWithDefault(Msel, pred_energy, 0.)  # K x V-obj x 1
    pred_obj_dep_e = tf.reduce_sum(pred_obj_hit_e, axis=1)  # K x 1
    neighbour_shower_energy_matrix_pred = SelectWithDefault(shower_neighbour_matrix, pred_obj_dep_e, 0.) # K x (1 + n_shower_neighbours)
    local_shower_energy_sum_pred = tf.reduce_sum(neighbour_shower_energy_matrix_pred, axis=1)  # K x 1

    # TODO calculate loss (as the sum of energy differences for each shower?)
    shower_energy_sq_diff = tf.math.squared_difference(local_shower_energy_sum_truth, local_shower_energy_sum_pred)
    local_energy_conservation_loss = tf.reduce_sum(shower_energy_sq_diff, axis=0)

    return local_energy_conservation_loss


    # FIXME OPEN QUESTIONS BELOW
    # truth RECO deposited energy
    # FIXME is the following needed?: nt_idx = t_idx + 1  # make noise object
    # obj_hit_e = SelectWithDefault(Msel, hit_energy, 0.) # K x V-obj x 1
    # obj_dep_e = tf.reduce_sum(obj_hit_e, axis=1)  # K x 1

    # FIXME I think I don't need to do this... since I need energy per shower only
    # _, idxs, _ = tf.unique_with_counts(truth_idx[:, 0])  # for backgather, same indices as in objsel
    # idxs = tf.expand_dims(idxs, axis=1)
    # scat_dep_e = tf.gather_nd(obj_dep_e, idxs)

    # FIXME from calc_energy_correction_factor_loss function
    # FIXME # calo-like
    # FIXME ediff = (t_energy - pred_energy * dep_energies) / tf.sqrt(t_energy + 1e-3)
