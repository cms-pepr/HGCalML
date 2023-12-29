#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("CompareKnnOutputs")
    // .Attr("scale_factor: int")
    .Input("in_tensor1: int32")
    .Input("in_tensor2: int32")
    .Output("out_tensor: int32")
;                                    
