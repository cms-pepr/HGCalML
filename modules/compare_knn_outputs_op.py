
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

compare_knn_outputs_op = tf.load_op_library('compare_knn_outputs.so')

def CompareKnnOutputs(inTensor1, inTensor2):
    '''
    returns scaleFactor * inTensor
    '''

    return compare_knn_outputs_op.CompareKnnOutputs(in_tensor1=inTensor1, in_tensor2=inTensor2)
