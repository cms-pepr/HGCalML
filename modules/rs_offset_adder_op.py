
import tensorflow as tf
from tensorflow.python.framework import ops

_index_replacer = tf.load_op_library('rs_offset_adder.so')

def RSOffsetAdder(idx, rs):
    '''
    Replaced indices in to_be_replaced, and returns the result.
    replacements are organised in the following way:
    
    replacements[i] contains the index j that i should be replaced with
    
    '''
    
    return _index_replacer.RSOffsetAdder(idx, rs)

@ops.RegisterGradient("RSOffsetAdder")
def _RSOffsetAdderGrad(op, gradin):
    return None,None
  
