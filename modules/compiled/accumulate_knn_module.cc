#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("AccumulateKnn")
    .Attr("n_moments: int")
    .Attr("mean_and_max: bool")
    .Input("distances: float32") //change to distances!!
    .Input("features: float32")
    .Input("indices: int32")
    .Output("out_features: float32")
    .Output("out_max_idxs: int32");



