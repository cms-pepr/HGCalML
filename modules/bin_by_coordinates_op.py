
import tensorflow as tf
from tensorflow.python.framework import ops

_bin_by_coordinates = tf.load_op_library('bin_by_coordinates.so')

def BinByCoordinates():
    
    return _bin_by_coordinates.BinByCoordinates()

@ops.RegisterGradient("BinByCoordinates")
def _BinByCoordinatesGrad(op, ):
    
    return None
  
