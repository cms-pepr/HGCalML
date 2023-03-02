
import tensorflow as tf
from tensorflow.python.framework import ops

_uidx = tf.load_op_library('unique_indices.so')

def UniqueIndices(labels, row_splits = None):
    '''
    Returns:
    - indices of the unique labels
    - row splits for the unique indices (only if row splits given as input)
    
    
    determine the indices (!) of the unique labels (int32), to be used with gather.
    The indices are global and span over multiple row splits.
    
    In case there are multiple same indices in the input, the index to the 
    unique index should not be considered deterministic
    
    
    '''
    if row_splits is None:
        row_splits = tf.constant([0, labels.shape[0]], dtype='int32')
        return _uidx.UniqueIndices(input_labels=labels,row_splits=row_splits)[0]
    
    return _uidx.UniqueIndices(input_labels=labels,row_splits=row_splits)

@ops.RegisterGradient("UniqueIndices")
def _UniqueIndicesGrad(op, gradin, gradin2):
    
    return None,None
  
