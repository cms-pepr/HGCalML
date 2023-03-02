
import tensorflow as tf
from binned_select_knn_op import BinnedSelectKnn
import numpy as np

import time
from push_knn_op import PushKnn
from push_knn_op import _tf_push_knn as tf_push_knn

def make_data(nvert, nk, nf):
    
    f = tf.constant(np.random.rand(nvert,nf),dtype='float32')
    w = tf.constant(np.random.rand(nvert,nk),dtype='float32')
    
    c = tf.constant(np.random.rand(nvert,3),dtype='float32') #just to get neighbours
    nidx, _ = BinnedSelectKnn(nk+1, c, tf.constant([0,nvert],dtype='int32'))
    
    return tf.constant(f),tf.constant(w),nidx[:,1:]
    
    


def run_operation(operation, watch, **kwargs):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(watch)
        out = operation(**kwargs)
    return out, t_newop.gradient
    
    
def run_test(n_vert, n_k, n_f):
    f,w,nidx = make_data(n_vert, n_k, n_f)
    
    print('running with',n_vert, n_k, n_f)
    
    if False:
        f = tf.constant([
            [1., 1./2., 1./4],
            [2., 2./2., 2./4],
            [3., 3./2., 3./4],
            [4., 4./2., 4./4]
            ])
        
        w = tf.constant([
            [.5, 1.],
            [1., 1.],
            [2., 1.],
            [1., 1.]
            ])
        
        nidx = tf.constant([
            [1, 2],
            [3, -1],
            [2, -1],
            [-1, -1]
            ])
    
    
        print('w',w)
        print('f',f)
        print('nidx', nidx)
    
    
    o, gw = run_operation(tf_push_knn, w,  w=w,f=f,nidx=nidx)
    
    tf_w_grad = gw( o, w )
    
    o, gw = run_operation(PushKnn, w,  w=w,f=f,nidx=nidx)
    
    w_grad = gw( o, w )
    
    diff = (tf_w_grad-w_grad)/tf.where(tf_w_grad == 0, 1e-3,tf_w_grad)
    diff = tf.reduce_max(tf.abs(diff))
    print('weight gradient diff',diff)
    
    if diff.numpy() > 1e-3:
        raise RuntimeError("weight grad diff too big")
    
    o, gw = run_operation(tf_push_knn, f,  w=w,f=f,nidx=nidx)
    
    tf_w_grad = gw( o, f )
    
    o, gw = run_operation(PushKnn, f,  w=w,f=f,nidx=nidx)
    
    w_grad = gw( o, f )
    
    diff = (tf_w_grad-w_grad)/tf.where(tf_w_grad == 0, 1e-3,tf_w_grad)
    diff = tf.reduce_max(tf.abs(diff))
    print('feature gradient diff',diff)
    
    if diff.numpy() > 1e-3:
        raise RuntimeError("feat grad diff too big")


for i in range(10):
    run_test((i+2)*100, 10+4*i, 24+2*i)

exit()

st = time.time()
tf_out = tf_push_knn(w,f,nidx)
print('tf took', time.time()-st,'s')
st = time.time()
k_out = PushKnn(w,f,nidx)
print('pushknn', time.time()-st,'s')

#print(tf_out)
#print(k_out)

diff = (tf_out-k_out)/tf.where(tf_out == 0, 1e-3,tf_out)
maxdiff = tf.reduce_max(diff)
print(maxdiff)

