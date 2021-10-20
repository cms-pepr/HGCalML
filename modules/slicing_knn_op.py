
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Wrap the module
'''

_nknn_op = tf.load_op_library('slicing_knn.so')

def check_tuple(in_tuple, tuple_name: str, tuple_type: type):
    if in_tuple is None:
        raise ValueError("<", tuple_name, "> argument is not specified!")
    if len(in_tuple)!=2:
        raise ValueError("<", tuple_name, "> argument has to be tuple of size 2!")
    if (type(in_tuple[0]) is not tuple_type) or (type(in_tuple[1]) is not tuple_type):
        raise ValueError("<", tuple_name, "> argument has to be of type Tuple[",tuple_type,",",tuple_type,"]!")
    if (in_tuple[0]<0) or (in_tuple[1]<0):
        raise ValueError("<", tuple_name, "> tuple has to contain only positive values!")



def SlicingKnn(K : int, coords, row_splits, features_to_bin_on=None, n_bins=None, bin_width=None):
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

    @type bin_width: Tuple[float, float]
    @param bin_width: width of phase-space bins
    '''

    # type and values check for input parameters
    check_tuple(features_to_bin_on,"features_to_bin_on",int)
    n_features = coords.shape[1]
    if (features_to_bin_on[0]>=n_features) or (features_to_bin_on[1]>=n_features) or (features_to_bin_on[0]==features_to_bin_on[1]):
        raise ValueError("Value error for <features_to_bin_on>!")
    if ((n_bins is None) and (bin_width is None)) or ((n_bins is not None) and (bin_width is not None)):
        raise ValueError("Specify either <n_bins> OR <bin_width> argument but not both!")
    if n_bins is None:
        check_tuple(bin_width,"bin_width",float)
    else:
        check_tuple(n_bins,"n_bins",int)

    # features to do 2d phase-space binning on
    bin_f1 = features_to_bin_on[0]
    bin_f2 = features_to_bin_on[1]

    # find min/max in tensor taking into account row_splits
    # TODO is creation of ragged tensor an expensive operation (memory and time wise)?
    coords_ragged = tf.RaggedTensor.from_row_splits(values=coords, row_splits=row_splits)
    r_max = tf.map_fn(tf.math.argmax, coords_ragged, fn_output_signature=tf.int64)
    r_min = tf.map_fn(tf.math.argmin, coords_ragged, fn_output_signature=tf.int64)

    # contains minimum and maximum coordinates of two first dimentions in coords tensor
    _phase_space_bin_boundary = []
    if n_bins is None:
        n_bins = [float('inf'),float('inf')]

    for i_split in range(0,len(row_splits)-1):
        min_coords = r_min[i_split]
        max_coords = r_max[i_split]

        _min = coords[min_coords[bin_f1]+row_splits[i_split].numpy()][bin_f1].numpy()
        _max = coords[max_coords[bin_f1]+row_splits[i_split].numpy()][bin_f1].numpy()
        _phase_space_bin_boundary.append((_min-0.00001*(_max-_min)).item())
        _phase_space_bin_boundary.append((_max+0.00001*(_max-_min)).item())

        _min = coords[min_coords[bin_f2]+row_splits[i_split].numpy()][bin_f2].numpy()
        _max = coords[max_coords[bin_f2]+row_splits[i_split].numpy()][bin_f2].numpy()
        _phase_space_bin_boundary.append((_min-0.00001*(_max-_min)).item())
        _phase_space_bin_boundary.append((_max+0.00001*(_max-_min)).item())

        # find n_bins for the current batch
        if bin_width is not None:
            n_bins_1 = int((_phase_space_bin_boundary[-3] - _phase_space_bin_boundary[-4]) / bin_width[0]) + 1
            n_bins_2 = int((_phase_space_bin_boundary[-1] - _phase_space_bin_boundary[-2]) / bin_width[1]) + 1
            if n_bins_1<n_bins[0]:
                n_bins[0] = n_bins_1
            if n_bins_2<n_bins[1]:
                n_bins[1] = n_bins_2

    if type(n_bins) is list:
        n_bins = tuple(n_bins)

    return _nknn_op.SlicingKnn(n_neighbours=K, coords=coords, row_splits=row_splits, n_bins=n_bins, features_to_bin_on=features_to_bin_on, phase_space_bin_boundary=_phase_space_bin_boundary)


_sknn_grad_op = tf.load_op_library('select_knn_grad.so')

@ops.RegisterGradient("SlicingKnn")
def _SelectKnnGrad(op, gradidx, dstgrad):

    coords = op.inputs[0]
    indices = op.outputs[0]
    distances = op.outputs[1]

    coord_grad = _sknn_grad_op.SelectKnnGrad(grad_distances=dstgrad, indices=indices, distances=distances, coordinates=coords)

    #  return coord_grad, None, None #no grad for row splits and masking values
    return coord_grad
