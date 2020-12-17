#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("New3Knn")
    .Attr("n_neighbours: int")
    .Attr("tf_compatible: bool")
    .Attr("max_radius: float")
    .Attr("n_bins_x: int")
    .Attr("n_bins_y: int")
    .Input("coords: float32")
    .Input("row_splits: int32")
    .Output("indices: int32")
    .Output("distances: float32");

