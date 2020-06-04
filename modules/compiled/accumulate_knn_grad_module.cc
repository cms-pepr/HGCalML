#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("AccumulateKnnGrad")
    .Input("grad_from_out_features: float32")
    .Input("distances: float32")
    .Input("features: float32")
    .Input("neigh_indices: int32")
    .Input("max_feat_indices: int32")
    .Output("out_grad_distances: float32")
    .Output("out_grad_features: float32");
    //.Output("grad_indices: int32");



