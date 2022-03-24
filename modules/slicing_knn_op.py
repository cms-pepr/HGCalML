
import tensorflow as tf
from tensorflow.python.framework import ops
import globals as gl
from oc_helper_ops import SelectWithDefault

'''
Wrap the module
'''

_nknn_op = tf.load_op_library('slicing_knn.so')

def check_tuple(in_tuple, tuple_name: str, tuple_type, checkValue=True):
    if not isinstance(tuple_type,tuple):
        tuple_type = (tuple_type, )
    if in_tuple is None:
        raise ValueError("<", tuple_name, "> argument is not specified!")
    if len(in_tuple)!=2:
        raise ValueError("<", tuple_name, "> argument has to be tuple of size 2!")
    if not isinstance(in_tuple[0], tuple_type) or not isinstance(in_tuple[1], tuple_type) or not (type(in_tuple[0])==type(in_tuple[1])):
        raise ValueError("<", tuple_name, "> argument has to be of type Tuple[",tuple_type,",",tuple_type,"]!", 'but is',
                         type(in_tuple[0]),'and',type(in_tuple[1]))
    if not checkValue and ((in_tuple[0]<0) or (in_tuple[1]<0)):
        raise ValueError("<", tuple_name, "> tuple has to contain only positive values!")



def SlicingKnn(K : int, coords, row_splits, features_to_bin_on=None, 
               n_bins=None, bin_width=None, return_n_bins: bool=False,
               min_bins=[3,3]):
    '''
    Perform kNN search with slicing method

    @type K: int
    @param K: number of neighbours to search for

    @type coords: tf.Tensor
    @param coords: coordinate tensor

    @type row_splits: tf.Tensor
    @param row_splits: row splits tensor

    @type features_to_bin_on: Tuple[int, int]
    @param features_to_bin_on: indices of features to bin on

    @type n_bins: Tuple[int, int]
    @param n_bins: number of bins to split phase space for kNN search

    @type bin_width: Tuple[float, float] or Tuple[tf.Variable, tf.Variable]
    @param bin_width: width of phase-space bins
    
    @type return_n_bins: bool
    @param return_n_bins: also returns the total number of bins used
    
    @type min_bins: list
    @param min_bins: minimum binning (in 2D)
    
    '''

    #  start_time_int = time.time()

    # type and values check for input parameters
    check_tuple(features_to_bin_on,"features_to_bin_on",int)
    n_features = coords.shape[1]
    if (features_to_bin_on[0]>=n_features) or (features_to_bin_on[1]>=n_features) or (features_to_bin_on[0]==features_to_bin_on[1]):
        raise ValueError("Value error for <features_to_bin_on>!")
    if ((n_bins is None) and (bin_width is None)) or ((n_bins is not None) and (bin_width is not None)):
        raise ValueError("Specify either <n_bins> OR <bin_width> argument but not both!")
    if n_bins is None:
        check_tuple(bin_width,"bin_width",(float,tf.Variable),checkValue=not isinstance(bin_width,tf.Variable))
    else:
        check_tuple(n_bins,"n_bins",int)

    # select only 2 dimensions that will be used for binning
    r_coords = tf.gather(coords,features_to_bin_on,axis=1)

    # find min/max of selected coordinates
    r_coords = tf.transpose(r_coords) # since tf.map_fn apply fn to each element unstacked on axis 0

    r_max = tf.map_fn(tf.math.reduce_max, r_coords, fn_output_signature=tf.float32)
    r_min = tf.map_fn(tf.math.reduce_min, r_coords, fn_output_signature=tf.float32)

    # add safety margin to the phase-space for binning
    r_diff = tf.add(r_max,-1*r_min)
    r_max = tf.add(r_max,0.00001*r_diff)
    r_min = tf.add(r_min,-0.00001*r_diff)
    r_diff = tf.add(r_max,-1*r_min)
    

    # calculate n_bins if bin_width is given
    if bin_width is not None:
        if not isinstance(bin_width[0], tf.Variable): #already checked both are the same
            bin_width = tf.constant(bin_width)
        else:
            bin_width = [tf.expand_dims(a,axis=0) for a in bin_width]
            bin_width = tf.concat(bin_width,axis=0)
        _n_bins = tf.math.maximum(tf.constant(min_bins, dtype=tf.int32),
                tf.math.minimum(
                    tf.cast(tf.math.ceil(tf.multiply(r_diff,1.0/bin_width)),tf.int32), 
                    tf.constant([50,50], dtype=tf.int32))) # limit the number of bins to min 3x3 and max 50x50
    else:
        _n_bins = tf.constant(n_bins, dtype=tf.int32) # cast tuple to Tensor to match required argument type
    
    idx, dist = _nknn_op.SlicingKnn(n_neighbours=K, coords=coords, row_splits=row_splits, n_bins=_n_bins, features_to_bin_on=features_to_bin_on, coord_min=r_min, coord_max=r_max)
    
    with tf.control_dependencies([
        tf.assert_equal(tf.range(tf.shape(idx)[0]), idx[:,0]),
        tf.assert_less(idx, row_splits[-1]),
        tf.assert_less(-2, idx)
        ]):
        
        if gl.knn_ops_use_tf_gradients:
            ncoords = SelectWithDefault(idx, coords, 0.)
            dist = (ncoords[:,0:1,:]-ncoords)**2
            dist = tf.reduce_sum(dist,axis=2)
            dist = tf.where(idx<0, 0., dist)
        
        if return_n_bins:
            return idx, dist, tf.reduce_prod(_n_bins)
        return idx, dist



_sknn_grad_op = tf.load_op_library('select_knn_grad.so')

@ops.RegisterGradient("SlicingKnn")
def _SlicingKnnGrad(op, gradidx, dstgrad):

    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]

    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)

    return coord_grad, None, None, None, None #no grad for row_splits, features_to_bin_on, n_bins and and bin_width




'''
notes:
The following example for 3D space coordinates.
Only two custom ops are needed in total, both rather simple

# bins are organised as a flat index indexing back to 4D: [row_splits, nbins, nbins, nbins]

unit_distance=some_value
nbins = tf.max(coords/unit_distance)
bin_no = custom_op_assign_bins(coords, row_splits, nbins=nbins) #nbins same in all dimensions


_, resort_indices, n_per_bin = tf.unique_with_counts(bin_no) 

# add a leading zero to n_per_bin
n_per_bin = tf.concatenate([row_splits[0], n_per_bin],axis=0)

# make it row split like
n_per_bin = tf.cumsum(n_per_bin)

# now tf.gather_nd(some_tensor, resort_indices) will resort everything
# and n_per_bin will define the boundaries of the bins in row split format:
# lower bound: n_per_bin[i], upper bound: n_per_bin[i+1]

sorted_coords = tf.gather_nd(coords,resort_indices)

dist, idx = custom_op_bin_knn(sorted_coords, n_per_bin, unit_distance)

# the above kNN is a very minor extension of SelectKNN where only the loop is restricted within a certain range
# This range is given by n_per_bin[b], n_per_bin[b+1]. 
#
# the index b is determined by a bin-stepper class (C++), that will be the heart of this kernel
# This class should be a helper and templated in a separate header file, since also future other kernels might
# want to use it.
#
# The bin stepper should be able to step an ND cube surface in a grid, where the cube sides are configurable. So e.g.
# it has a 'radius', where radius=0 means, it would give the index of the origin bin,
# radius=1 means, it would return the flat indices for the nearest neighbours to the origin bin, 
# radius=2 would mean it would return the flat indices of the neighbours of the nearest neighbour bins etc.
# 
# This class must be stateful, such that the following can happen
#

int radius=0;
while(true){
    binstepper stepper(origin_bin, dimensions, radius);
    int bin = stepper.get()
    for(vertices[bin] to vertices[bin+1]){
       determine kNN and max_distance.
    }
    if(stepper.done()){//stepped through all bins of a cube with side=radius
        if(unit_distance>2.*max_distance
           || stepper.no_more_bins()  //this can happen if the whole coordinate space has been scanned (radius>=nbins/2)
        ){ //might need to be adapted, just a guess right now
            break; //done
        }
        else{
            radius+=1
        }
    }
}

# use tf.scatter_nd(resort_indices,...) to get back to old format, 
# possibly do some simple index reshuffling for neighbour indices

'''







