import tensorflow as tf
import numpy as np
from LossLayers import LLFullObjectCondensation
import random

#Small test to check if the loss function is working as expected afer I changed the oc-Kernals

def random_loss(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    llFOBC = LLFullObjectCondensation(scale=1.,
                use_energy_weights=True,
                record_metrics=True,
                print_loss=True,
                name="ExtendedOCLoss",
                implementation = "hinge",
                beta_loss_scale = 1.0,
                too_much_beta_scale = 0.0,
                energy_loss_weight = 0.4,
                classification_loss_weight = -0.4,
                position_loss_weight =  0.0,
                timing_loss_weight = 0.0,
                q_min = 1.0,
                use_average_cc_pos = 0.9999)

    # Adjusting the inputs to have random values but similar to the provided examples
    length = np.random.randint(1000, 1100)


    pred_beta = tf.random.uniform((length, 1), minval=0.996, maxval=1.0, dtype=tf.float32)
    pred_ccoords = tf.random.uniform((length, 3), minval=-76.6, maxval=292.2, dtype=tf.float32)
    pred_distscale = tf.ones((length, 1), dtype=tf.float32)
    pred_energy = tf.random.uniform((length, 1), minval=1.009, maxval=1.194, dtype=tf.float32)
    pred_energy_low_quantile = tf.zeros((length, 1), dtype=tf.float32)
    pred_energy_high_quantile = tf.random.uniform((length, 1), minval=0, maxval=165, dtype=tf.float32)
    pred_pos = tf.random.uniform((length, 2), minval=-29.99, maxval=166.66, dtype=tf.float32)
    pred_time = tf.random.uniform((length, 1), minval=7.211, maxval=10.004, dtype=tf.float32)
    pred_time_unc = tf.random.uniform((length, 1), minval=0.974, maxval=3.322, dtype=tf.float32)
    pred_id = tf.random.uniform((length, 6), minval=0, maxval=8, dtype=tf.float32)
    rechit_energy = tf.random.uniform((length, 1), minval=13.72, maxval=4103.71, dtype=tf.float32)
    t_idx = tf.random.uniform((length, 1), minval=1, maxval=8, dtype=tf.int32)
    t_energy = tf.random.uniform((length, 1), minval=2302.9, maxval=6639.13, dtype=tf.float32)
    t_pos = tf.random.uniform((length, 3), minval=-0.0027, maxval=0.0011, dtype=tf.float32)
    t_time = tf.zeros((length, 1), dtype=tf.float32)
    t_pid = tf.random.uniform((length, 1), minval=-211, maxval=211, dtype=tf.int32)
    t_spectator_weights = tf.zeros((length, 1), dtype=tf.float32)
    t_fully_contained = tf.ones((length, 1), dtype=tf.int32)
    t_rec_energy = tf.random.uniform((length, 1), minval=812.48, maxval=2587.7, dtype=tf.float32)
    t_is_unique = tf.random.uniform((length, 1), minval=0, maxval=1, dtype=tf.int32)

    #randomly generate rowsplits for testing
    # Generate three random integers between 1 and length-1, ensuring they are sorted
    num_splits = np.random.randint(1, 3)
    random_splits = sorted(np.random.randint(8, length, size=num_splits))

    # Append 0 at the beginning and 'length' at the end to form valid row splits
    rowsplits = [0] + random_splits + [length]

    #rowsplits = [0] + [length]
    
    # Convert to a TensorFlow constant
    rowsplits = tf.constant(rowsplits, dtype=tf.int32)


    # Grouping inputs
    inputs = [pred_beta, pred_ccoords, pred_distscale,
            pred_energy, pred_energy_low_quantile, pred_energy_high_quantile,
            pred_pos, pred_time, pred_time_unc, pred_id,
            rechit_energy,
            t_idx, t_energy, t_pos, t_time, t_pid, t_spectator_weights, t_fully_contained, t_rec_energy,
            t_is_unique,
            rowsplits]


    loss = llFOBC.loss(inputs)  # Adjust this call based on how the loss function is actually used
    print(loss)
    return loss


for i in [42,300,500,1,565656,400432,342342,3242342]:
    random_loss(i)