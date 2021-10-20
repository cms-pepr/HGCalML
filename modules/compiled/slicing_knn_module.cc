#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SlicingKnn")
    .Attr("n_bins: list(int) >= 2")
    .Attr("features_to_bin_on: list(int) >= 2")
    .Attr("n_neighbours: int")
    .Attr("phase_space_bin_boundary: list(float) >= 4")
    .Input("coords: float32")
    .Input("row_splits: int32")
    .Output("indices: int32")
    .Output("distances: float32");
