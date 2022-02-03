
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BinByCoordinates")
    .Input("input: float") 
    .Attr("attr: float")
    .Output("output: int32"); 
