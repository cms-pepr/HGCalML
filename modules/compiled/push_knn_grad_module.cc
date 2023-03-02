#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("PushKnnGrad")
    .Input("grad: float32")
    .Input("weights: float32")
    .Input("features: float32")
    .Input("indices: int32")
    .Output("weight_grad: float32")
    .Output("feature_grad: float32");

