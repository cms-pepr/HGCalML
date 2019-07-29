import tensorflow as tf


def transform_coordinates(x, other_options):
    
    xcoord = x[:,:,0:1]
    ycoord = x[:,:,1:2]
    zcoord = x[:,:,2:3]
    
    otherfeatures = x[:,:,3:]
    
    #transform to spherical coordinates
    r = tf.math.sqrt( xcoord**2 + ycoord**2 + zcoord**2 )
    theta = tf.math.acos( zcoord / r )
    phi = tf.math.atan( ycoord / xcoord )
    
    
    ##replace nan values with 0 to deal with divergences
    thetazeros = tf.zeros(theta.shape)
    phizeros = tf.zeros(phi.shape)    
    theta = tf.where(tf.is_nan(theta), thetazeros, theta)
    phi = tf.where(tf.is_nan(phi), phizeros, phi)     
    
    ##if eta would be used, need to build in a check that theta is not 0
    #eta = -tf.math.log( tf.math.tan( theta / 2) )   
    
    #replace cartesian coordinates with spherical ones calculated above    
    #by concatenating tensors along last axis (the features axis, in our example axis 2)
    xtrans = tf.concat( [r,theta,phi,otherfeatures], -1)
        
    return xtrans

    
#a version that works by passing the index of the coordinate, but only works on tensors with last dimension 3
def transform_coordinates_v2(x, feature_index_x, feature_index_y, feature_index_z):
    
    xcoord = x[:,:,feature_index_x:feature_index_x+1]
    ycoord = x[:,:,feature_index_y:feature_index_y+1]
    zcoord = x[:,:,feature_index_z:feature_index_z+1]
	        
    #transform to spherical coordinates
    r = tf.math.sqrt( xcoord**2 + ycoord**2 + zcoord**2 )
    theta = tf.math.acos( zcoord / r )
    phi = tf.math.atan( ycoord / xcoord )
    
    
    ##replace nan values with 0 to deal with divergences
    thetazeros = tf.zeros(theta.shape)
    phizeros = tf.zeros(phi.shape)    
    theta = tf.where(tf.is_nan(theta), thetazeros, theta)
    phi = tf.where(tf.is_nan(phi), phizeros, phi)     
    
    ##if eta would be used, need to build in a check that theta is not 0
    #eta = -tf.math.log( tf.math.tan( theta / 2) )   
    
    #replace cartesian coordinates with spherical ones calculated above    
    #by concatenating tensors along last axis (the features axis, in our example axis 2)
    xtrans = tf.concat( [r,theta,phi], -1)
        
    return xtrans    

    
def dummy(x, other_options):
    return tf.identity( x )


def printAndEval(prtext, x, sess):
    
    print(prtext,x.shape)
    ##print(x)
    result = sess.run(x)
    print(result)


# example Batches x Nodes x SpatialCoordinates tensor
test_spatial_global = tf.constant([[[1, 2, 3],
                                    [1.1, 2.1, 3.1],
                                    [1, 1.9, 3],
                                    [0, 0, 0],
                                    [0, 4, 2]],
                                    [[11, 12, 13],
                                     [11.1, 12.1, 13.1],
                                     [11, 11.9, 13],
                                     [10, 13, 12],
                                     [10, 14, 12]]],dtype=tf.float32)


# example Batches x Nodes x Features tensor
# Features represents e.g. SpatialCoordinates, energy, and rechit id
test_features_global = tf.constant([[[1, 2, 3, 17.1, 412],
                                    [1.1, 2.1, 3.1, 29.5, 522],
                                    [1, 1.9, 3, 0.4, 1704],
                                    [0, 0, 0, 14.9, 489],
                                    [0, 4, 2, 0.0, 128]],
                                    [[11, 12, 13, 14.9, 12],
                                     [11.1, 12.1, 13.1, 37.8, 1630],
                                     [11, 11.9, 13, 20.3, 1381],
                                     [10, 13, 12, 11.7, 945],
                                     [10, 14, 12, 6.4, 398]]],dtype=tf.float32)




print('input shape spatial features: ',test_spatial_global.shape)
print('input shape all features: ',test_features_global.shape)



with tf.Session() as sess:


    someoptions = ''
    
    input_spatial = dummy(test_spatial_global, someoptions)
    input_features = dummy(test_features_global, someoptions) 
    
    transform_spatial = transform_coordinates(test_spatial_global, someoptions)
    transform_features = transform_coordinates(test_features_global, someoptions)        
    transform_spatial_v2 = transform_coordinates_v2(test_features_global, 1, 2, 3)
    
    
    sess.run(tf.global_variables_initializer())


    ##print('output shape: ',transform.shape) 
    
    printAndEval('input tensor (spatial features)',input_spatial,sess)
    printAndEval('input tensor (all features)',input_features,sess)    
       
    printAndEval('transformed tensor (spatial features)',transform_spatial,sess)    
    printAndEval('transformed tensor (all features)',transform_features,sess)
    printAndEval('transformed tensor v2 (output spatial features)',transform_spatial_v2,sess)        
    
    
    
    
    
    
