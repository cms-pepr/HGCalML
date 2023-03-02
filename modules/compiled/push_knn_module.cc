#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("PushKnn")
    .Input("weights: float32") //change to distances!!
    .Input("features: float32")
    .Input("indices: int32")
    .Output("out_features: float32");



