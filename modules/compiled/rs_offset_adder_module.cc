
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RSOffsetAdder")
    .Input("t_dx: int32")
    .Input("row_splits: int32")
    .Output("output: int32"); 
