#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("New5Knn")
    .Attr("n_neighbours: int")
    .Input("coords: float32")
    .Input("n_vtx_per_bin_cumulative: int32")
    .Input("bin_neighbours: int32")
    .Input("vtx_bin_assoc: int32")
    .Output("indices: int32")
    .Output("distances: float32");
