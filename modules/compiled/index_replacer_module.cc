
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("IndexReplacer")
    .Input("to_be_replaced: int32")
    .Input("replacements: int32")
    .Output("output: int32"); 
