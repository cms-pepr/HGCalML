import tensorflow as tf
from oc_helper_ops import SelectWithDefault
from oc_helper_ops import CreateMidx
from select_knn_op import SelectKnn


def get_local_energy_conservation_loss_per_batch_element(
        x,
        truth_idx,
        hit_energy,
        t_energy,
        pred_energy_corrections,
        max_shower_dist=30,  # TODO think what value is the best
        n_shower_neighbours=3
):

    # create hit-shower association matrix (each row contains all hits associated to one shower)
    Msel, _, _ = CreateMidx(truth_idx, calc_m_not=False)
    if Msel is None:
        print('>>> WARNING: Event has no objects, only noise! Will return zero loss. <<<')
        return None

    x_m = SelectWithDefault(Msel, x, 0.)  # K x V-obj x C
    hit_counter = tf.where(x_m[:, :, 0] != 0., 1, 0)  # K x V-obj
    x_m_sum = tf.reduce_sum(x_m, axis=1)  # K x C
    n_hits_per_shower = tf.reduce_sum(hit_counter, axis=1)  # K
    n_hits_per_shower = tf.expand_dims(n_hits_per_shower, axis=1)  # K x 1
    n_hits_per_shower = tf.cast(n_hits_per_shower, dtype='float32')  # K x 1
    # get shower centers as geometrical mean of hit coordinates
    x_showers = tf.math.divide_no_nan(x_m_sum, n_hits_per_shower)  # K x C

    # perform kNN using x_m to get closest neighbour matrix
    shower_neighbour_matrix, _ = SelectKnn(n_shower_neighbours,  # K x (1 + n_shower_neighbours)
                                           x_showers, tf.constant([0, Msel.shape[0]], dtype="int32"),
                                           max_radius=max_shower_dist, tf_compatible=True)

    # for backgather, the same indices as in Msel
    _, idxs, _ = tf.unique_with_counts(truth_idx[:, 0])
    idxs = tf.expand_dims(idxs, axis=1)  # V x 1

    # calculate deposited energy per shower
    hit_energies_per_shower = SelectWithDefault(Msel, hit_energy, 0.)  # K x V-obj x 1
    hit_energy_sum_per_shower = tf.reduce_sum(hit_energies_per_shower, axis=1)  # K x 1
    scat_energy_deposited = tf.gather_nd(hit_energy_sum_per_shower, idxs)  # V x 1

    # calculate PREDICTED energy sum of all neighbour RECO showers + the shower itself
    predicted_energy = scat_energy_deposited * pred_energy_corrections  # V x 1
    neighbour_shower_energy_matrix_predicted = SelectWithDefault(shower_neighbour_matrix, predicted_energy, 0.)  # K x (1 + n_shower_neighbours)
    local_shower_energy_sum_predicted = tf.reduce_sum(neighbour_shower_energy_matrix_predicted, axis=1)  # K x 1
    scat_local_energy_predicted = tf.gather_nd(local_shower_energy_sum_predicted, idxs)  # V x 1

    # calculate TRUTH energy sum of all neighbour TRUTH showers + the shower itself
    neighbour_shower_energy_matrix_truth = SelectWithDefault(shower_neighbour_matrix, t_energy, 0.)  # K x (1 + n_shower_neighbours)
    local_shower_energy_sum_truth = tf.reduce_sum(neighbour_shower_energy_matrix_truth, axis=1)  # K x 1
    scat_local_energy_truth = tf.gather_nd(local_shower_energy_sum_truth, idxs)  # V x 1

    ediff = (scat_local_energy_truth - scat_local_energy_predicted) / tf.sqrt(scat_local_energy_truth + 1e-3)  # V x 1

    return ediff
