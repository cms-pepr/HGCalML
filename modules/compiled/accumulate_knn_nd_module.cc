#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("AccumulateKnnNd")
    .Attr("n_moments: int")
    .Input("coords: float32")
    .Input("features: float32")
    .Input("indices: int32")
    .Output("out_features: float32")
    .Output("out_max_idxs: int32")
    .Output("out_feature_sum: float32");



