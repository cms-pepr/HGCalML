#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("LocalCluster")
    .Input("neighbour_idxs: int32") //change to distances!!
    .Input("hierarchy_idxs: int32")
    .Input("global_idxs: int32")
    .Input("row_splits: int32")
    .Output("out_row_splits: int32")
    .Output("selection_idxs: int32")
    .Output("backscatter_idxs: int32");



