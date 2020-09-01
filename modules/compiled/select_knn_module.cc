#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SelectKnn")
    .Attr("n_neighbours: int")
    .Attr("tf_compatible: bool")
    .Attr("max_radius: float")
    .Attr("mask_mode: int")
    .Input("coords: float32")
    .Input("row_splits: int32")
    .Input("mask: int32")
    .Output("indices: int32")
    .Output("distances: float32");



