
import tensorflow as tf
from tensorflow.python.framework import ops

_index_replacer = tf.load_op_library('rs_offset_adder.so')

def RSOffsetAdder(idx, row_splits):
    '''
    Adds offsets to indices in idx, and returns the result.
    The offsets equal the row split offsets. Indices < 0 are ignored.
    '''
    # checks if idx is dim None x 1, and rs is None, both are TF tensors
    if ((idx.shape.ndims == 2 and idx.shape[1] == 1) or idx.shape.ndims < 2) and idx.shape.ndims == 1:
        return _index_replacer.RSOffsetAdder(idx=idx, row_splits=row_splits)
    else:
        raise ValueError('idx must be a 1D tensor or a 2D tensor with shape None x 1')

@ops.RegisterGradient("RSOffsetAdder")
def _RSOffsetAdderGrad(op, gradin):
    return None,None
  


def test(print_debug = False):
    # a quick test in eager mode
    import numpy as np
    import tensorflow as tf
    idx = tf.constant([0,-1,1,-1,4,5,-1,2,1,100], dtype=tf.int32)
    rs = tf.constant([0,3,6,10], dtype=tf.int32)
    idx = RSOffsetAdder(idx, rs)

    # expected output: [  0  -1   1  -1   7   8  -1   8   7 106]
    if print_debug:
        print(idx.numpy())

    multiplier = 1000
    #more rigorous test implemeting same in numpy with a loop over rowsplits and 10*multiplier random indices
    idx = np.random.randint(-1, 100, 10*multiplier, dtype=np.int32)
    rs = multiplier * np.array([0,3,6,10], dtype=np.int32)
    kernel_idx = RSOffsetAdder(tf.constant(idx), tf.constant(rs))
    np_idx = idx.copy()
    for i in range(1, len(rs)+1):
        if i == len(rs):
            np_idx[rs[i-1]:]  = np.where( np_idx[rs[i-1]:] >=0 , np_idx[rs[i-1]:] + rs[i-1], np_idx[rs[i-1]:])
        else:
            np_idx[rs[i-1]:rs[i]]  = np.where( np_idx[rs[i-1]:rs[i]] >=0 , np_idx[rs[i-1]:rs[i]] + rs[i-1], np_idx[rs[i-1]:rs[i]])
    return np.all(np_idx == kernel_idx.numpy())
