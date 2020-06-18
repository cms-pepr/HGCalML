import tensorflow as tf
from tensorflow.python.framework import ops


_bc_op = tf.load_op_library('build_condensates.so')

def BuildCondensates(ccoords, betas, features, row_splits, radius=0.8, min_beta=0.1):
    '''
    betas are sum(V) x 1
    '''
    #get sorted etc here
    beta_sorting=[]
    for e in tf.range(len(row_splits)-1):
        sorted = tf.argsort(betas[row_splits[e]:row_splits[e+1],0],axis=0)
        beta_sorting.append(sorted)
    
    beta_sorting = tf.concat(beta_sorting,axis=0)
    print(beta_sorting)

    summed_features, asso_idx = _bc_op.BuildCondensates(ccoords=ccoords, betas=betas, beta_sorting=beta_sorting, 
                                   features=features, row_splits=row_splits, radius=radius, 
                                   min_beta=min_beta)
    return summed_features, asso_idx
    
    