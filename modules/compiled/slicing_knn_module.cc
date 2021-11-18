#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SlicingKnn")
    .Attr("features_to_bin_on: list(int) >= 2")
    .Attr("n_neighbours: int")
    .Input("coords: float32")
    .Input("row_splits: int32")
    .Input("n_bins: int32")
    .Input("coord_min: float32")
    .Input("coord_max: float32")
    .Output("indices: int32")
    .Output("distances: float32");
