

import tensorflow as tf
import numpy as np
import time
from unique_indices_op import UniqueIndices as uidx


a = tf.constant( [ 1,1,1,4,4,-9,8,3,1,1  ], dtype='int32' )
rs = tf.constant( [ 0, 5, 10], dtype='int32' )

print(uidx(a,rs))



#now performance test

a = np.random.randint(0,20, size=100000)
print(a.shape)
a = tf.constant(a, dtype='int32')
rs = tf.constant([0,40000,100000], dtype='int32')

t = uidx(a)
st = time.time()
for _ in range(5):
    t = uidx(a,rs)
print('time passed:', (time.time()-st)/5.)
print(t[0], t[1])

g = tf.gather_nd(a, t[0])
print(g)