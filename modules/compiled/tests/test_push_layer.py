
import tensorflow as tf
from binned_select_knn_op import BinnedSelectKnn
import numpy as np
from GraphCondensationLayers import PushUp, GraphCondensation

import time
from push_knn_op import PushKnn
from push_knn_op import _tf_push_knn as tf_push_knn

def make_data(nvert, nk, nf):
    
    f = tf.constant(np.random.rand(nvert,nf),dtype='float32')
    w = tf.constant(np.random.rand(nvert,nk),dtype='float32')
    
    c = tf.constant(np.random.rand(nvert,3),dtype='float32') #just to get neighbours
    nidx, _ = BinnedSelectKnn(nk+1, c, tf.constant([0,nvert],dtype='int32'))
    
    return tf.constant(f),tf.constant(w),nidx[:,1:]


def simple_replacement_sum(f,trans,weight = None, select=False, mode='sum', add_self=False):
    #most simple way of implementing it
    out = (f*0.).numpy()
    wsum = (f[:,0:1]*0.).numpy()
    if weight is None:
        weight = f[:,0:1]*0. + 1.
        
    if add_self:
        for i_f in range(out.shape[1]):
            for i_v in range(f.shape[0]):
                if trans['nidx_down'][i_v,0] < 0:#up
                    out[i_v,i_f] += f[i_v,i_f] * weight[i_v] * trans['weights_down'][i_v,0]
                    wsum[i_v] +=  weight[i_v] * trans['weights_down'][i_v,0]
        
    for i_f in range(out.shape[1]):
        for i_v in range(f.shape[0]):
            for i_n in range(trans['nidx_down'].shape[1]):
                nidx = trans['nidx_down'][i_v,i_n]
                if nidx < 0:
                    continue
                out[nidx,i_f] += f[i_v,i_f] * weight[i_v] * trans['weights_down'][i_v,i_n]
                wsum[nidx] += weight[i_v] * trans['weights_down'][i_v,i_n] 
                
    if mode == 'mean':
        for i_f in range(out.shape[1]):
            for i_v in range(f.shape[0]):
                if wsum[i_v]:
                    out[i_v,i_f] /= wsum[i_v] / out.shape[1] #divide by nfeat that was added above
            
    
    out = tf.gather_nd(out, trans['sel_idx_up'])
    return out
    

def simple_data(randomise = False):
    f = tf.constant([
            [1., 1./2., 1./4],
            [2., 2./2., 2./4],
            [3., 3./2., 3./4],
            [4., 4./2., 4./4],
            [10.,11.,12.]
            ])
    
    wf = tf.constant([
        [2],
        [1.5],
        [3],
        [0.5],
        [3.]
        ],dtype='float32')
        
    w = tf.constant([
        [1.5, 2.],
        [1., 1.],
        [2., 1.],
        [20., 1.],
        [2., 1.]
        ])
    
    nidx = tf.constant([
        [1, 3],
        [-1, -1],
        [1, -1],
        [-1, -1],
        [-1, -1]
        ])
    
    if randomise:
        f = tf.constant(np.random.rand(*list(f.shape)),dtype='float32')+1.
        wf = tf.constant(np.random.rand(*list(wf.shape)),dtype='float32')+0.2
        w = tf.constant(np.random.rand(*list(w.shape)),dtype='float32')+0.1 #make it all numerically stable
    '''
    'rs_down',
    'rs_up',
    'nidx_down',
    'distsq_down', #in case it's needed
    'sel_idx_up', # -> can also be used to scatter
    'weights_down'
    '''
    trans = GraphCondensation()
    trans['rs_down'] = tf.constant([0,4],dtype='int32')
    trans['rs_up'] = tf.constant([0,2],dtype='int32')
    trans['nidx_down'] = nidx
    trans['distsq_down'] = tf.abs(w)
    trans['sel_idx_up'] = tf.constant([[1],[3],[4]],dtype='int32')
    trans['weights_down'] = w
    
        
    return f, tf.abs(wf), trans
    
f, wf, trans = simple_data(True)

#print(PushUp(mode='sum')(f,trans))
#simple_replacement_sum(f,trans)
#
#print(PushUp(mode='sum',add_self=True)(f,trans))
#simple_replacement_sum(f,trans,add_self=True)
#

#print(PushUp(mode='sum')(f,trans, weight = wf))
#simple_replacement_sum(f,trans, weight = wf)
#
#print(PushUp(mode='sum',add_self=True)(f,trans, weight = wf))
#simple_replacement_sum(f,trans, weight = wf,add_self=True)
#
#exit()

#print(PushUp(mode='mean')(f,trans))
#simple_replacement_sum(f,trans,mode='mean')
#
#f, wf, trans = simple_data(randomise=True)
#print(PushUp(mode='mean')(f,trans))
#simple_replacement_sum(f,trans,mode='mean')

#exit()
pu = PushUp(mode='mean')(f,trans,weight = wf)
spu = simple_replacement_sum(f,trans,weight = wf,mode='mean')
print(pu)
print(spu)
print(pu-spu)

pu =  PushUp(mode='mean',add_self=True)(f,trans,weight = wf)
spu = simple_replacement_sum(f,trans,weight = wf,mode='mean',add_self=True)

print(pu)
print(spu)
print(pu-spu)










