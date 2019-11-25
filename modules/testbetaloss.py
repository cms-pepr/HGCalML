
import DeepJetCore

from betaLosses import get_arb_loss

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()



row_splits = np.array([0,5,9,13,19]+[0]*13+[5], np.int)
ccoords = np.random.normal(size=(19, 2))*2
beta = np.random.normal(size=(19,))*10.
beta = z = 1/(1 + np.exp(-beta))
cluster_asso = np.floor(np.random.normal(size=(19,))*3)
cluster_asso[cluster_asso<0] = -1

print('ccoords',ccoords)

print('row_splits',row_splits)

print('beta',beta)

print('cluster_asso',cluster_asso)


ccoords = tf.constant(ccoords)
beta = tf.constant(beta)
row_splits = tf.constant(row_splits)
cluster_asso = tf.constant(cluster_asso)


tf.Print(ccoords,[ccoords],'ccoords ')

# n is prediction but I am setting it to all zeros

n = tf.constant(np.where(cluster_asso< 0, np.zeros_like(cluster_asso)+1, np.zeros_like(cluster_asso)))

attractive_loss, rep_loss, min_beta_loss = get_arb_loss(ccoords, row_splits, beta, n, cluster_asso)

print(attractive_loss)

#with tf.compat.v1.Session() as sess:
#    _,_,_ = sess.run([attractive_loss, rep_loss, min_beta_loss])


