
import tensorflow as tf
from tensorflow.python.framework import ops

_index_replacer = tf.load_op_library('index_replacer.so')

def IndexReplacer(to_be_replaced, replacements):
    '''
    Replaced indices in to_be_replaced, and returns the result.
    replacements are organised in the following way:
    
    replacements[i] contains the index j that i should be replaced with
    
    '''
    
    return _index_replacer.IndexReplacer(to_be_replaced=to_be_replaced,replacements=replacements)

@ops.RegisterGradient("IndexReplacer")
def _IndexReplacerGrad(op, gradin):
    
    return None,None
  
