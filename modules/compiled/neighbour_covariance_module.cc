#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("NeighbourCovariance")
    .Input("coordinates: float32")
    .Input("features: float32")
    .Input("n_idxs: int32")
    .Output("covariance: float32")
    .Output("means: float32");



