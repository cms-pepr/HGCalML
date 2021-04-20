
from oc_helper_ops import CreateMidx, SelectWithDefault
import tensorflow as tf

nvert=20

truth_idxs = tf.random.uniform((nvert,1), 0, 6, dtype='int32', seed=0) - 1 #for noise
features =  tf.random.uniform((nvert,1),seed=0)


selidx,mnot,cperunique = CreateMidx(truth_idxs, calc_m_not=True)

#just a small consistency check



#print(truth_idxs)
#print(selidx)
#print(mnot)
#print(cperunique)

beta_m  = SelectWithDefault(selidx, features, -1.)

kalpha_m = tf.argmax(beta_m,axis=1) 
#print(beta_m, kalpha_m)

#print(tf.gather_nd(beta_m,kalpha_m, batch_dims=1))

#now test the whole loss

from object_condensation import oc_per_batch_element, oc_per_batch_element_old

'''
oc_per_batch_element(
        beta,
        x,
        q_min,
        object_weights, # V x 1 !!
        truth_idx,
        is_spectator,
        payload_loss,
        S_B=1.,
        payload_weight_function = None,  #receives betas as K x V x 1 as input, and a threshold val
        payload_weight_threshold = 0.8,
        use_mean_x = False,
        cont_beta_loss=False,
        prob_repulsion=False,
        phase_transition=False,
        phase_transition_double_weight=False,
        alt_potential_norm=False,
        cut_payload_beta_gradient=False
        ):
'''

nvert=1000
diffs=[]
for seed in range(40):

    ccoords = tf.random.uniform((nvert,3), dtype='float32', seed=seed)
    beta = tf.random.uniform((nvert,1), dtype='float32', seed=seed)
    truth_idxs = tf.random.uniform((nvert,1), 0, 6, dtype='int32', seed=seed) - 1 
    is_spectator = tf.zeros_like(beta)
    payload_loss = tf.random.uniform((nvert,3), dtype='float32', seed=seed)
    object_weights = tf.zeros_like(beta) + 1.
    
    V_att, V_rep, Noise_pen, B_pen, pll, to_much_B_pen = oc_per_batch_element(beta, ccoords, 0.5, object_weights, truth_idxs, is_spectator, payload_loss,
                               S_B=1.,phase_transition=True, prob_repulsion=True,alt_potential_norm=True)
    
    
    
    V_att_old, V_rep_old, Noise_pen_old, B_pen_old, pll_old, to_much_B_pen_old = oc_per_batch_element_old(beta, ccoords, 0.5, object_weights, truth_idxs, is_spectator, payload_loss,
                               S_B=1.,phase_transition=True, prob_repulsion=True,alt_potential_norm=True)

    
    dVatt = V_att-V_att_old
    dVrep = V_rep-V_rep_old
    dNoise_pen = Noise_pen-Noise_pen_old
    diffs+=dVatt.numpy().flatten().tolist()
    diffs+=dVrep.numpy().flatten().tolist()
    diffs+=dNoise_pen.numpy().flatten().tolist()
    #the others are expected to differ

print(diffs)    
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.hist(diffs)
plt.xlabel("∆(old,new)")
plt.savefig("diff_in_oc_new.pdf")



