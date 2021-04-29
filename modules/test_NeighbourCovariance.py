

def test_NeighbourCovariance():
    from scipy.stats import random_correlation as randc
    from numpy.random import multivariate_normal
    import numpy as np
    
    import matplotlib.pyplot as plt
    
    nvert=200
    
    np.random.seed(514)
    x = randc.rvs((.2, 1.8))
    print(x)
    coords = multivariate_normal([1.,2.], x, size=nvert)
    
    print('coords in',coords.shape)
    
    #plt.scatter(coords[:,0],coords[:,1])
    #plt.show()
    
    import tensorflow as tf
    from neighbour_covariance_op import NeighbourCovariance
    
    coordinates = tf.constant(coords,dtype='float32')
    n_idxs = tf.tile(tf.expand_dims(tf.range(nvert),axis=0),[nvert,1])
    print(n_idxs)
    distsq = tf.cast(n_idxs, dtype='float32')/100.
    features = tf.constant(np.random.rand(nvert,2),dtype='float32')+1e-2
    
    cov, mean_C = NeighbourCovariance(coordinates, distsq, features, n_idxs)
    
    #expected shapes: V x F x 2 * 2, V x F x 2
    print(cov.shape, mean_C.shape)
    
    print('op neighbour covariance and mean')
    print(cov[0],'\n',mean_C[0]) #all the same
    
    print('numpy neighbour covariance')
    print(np.cov(coords, aweights=features[:,0]*np.exp(-distsq[0]), rowvar=False, ddof=0))
    print(np.cov(coords, aweights=features[:,1]*np.exp(-distsq[0]), rowvar=False, ddof=0))
    


test_NeighbourCovariance()