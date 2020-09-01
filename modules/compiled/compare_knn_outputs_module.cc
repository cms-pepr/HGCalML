#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("CompareKnnOutputs")
    // .Attr("scale_factor: int")
    .Input("in_tensor1: int32")
    .Input("in_tensor2: int32")
    .Output("out_tensor: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0)); // requires that output with idx 0
            return Status::OK();           // should have the same shape as
    })                                     // input with idx 0.
;                                    
