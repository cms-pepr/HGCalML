
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("BinnedSelectKnn")
    .Attr("n_neighbours: int")
    .Attr("tf_compatible: bool")
    .Attr("use_direction: bool")
    .Input("coords: float")
    .Input("bin_idx: int32")
    .Input("dim_bin_idx: int32")
    .Input("bin_boundaries: int32")
    .Input("n_bins: int32")
    .Input("bin_width: float")
    .Input("direction: int32")
    .Output("indices: int32")
    .Output("distances: float32");

