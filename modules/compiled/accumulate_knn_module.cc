#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("accumulate_knn")
    .Attr("n_moments: int")
    .Input("coords: float32")
    .Input("features: float32")
    .Output("out_features: float32");



