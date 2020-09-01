#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SelectThreshold")
    .Attr("threshold: float")
    .Input("th_value: float32")
    .Input("rowsplits: int32")
    .Output("scatter_idxs: int32")
    .Output("new_rowsplits: int32");



