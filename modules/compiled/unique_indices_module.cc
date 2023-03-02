
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("UniqueIndices")
    .Input("input_labels: int32")
    .Input("row_splits: int32")
    .Output("indices: int32")
    .Output("unique_row_splits: int32");

