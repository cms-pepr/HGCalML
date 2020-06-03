#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SelectKnnGrad")
    .Input("grad_distances: float32")
    .Input("indices: int32")
    .Input("distances: float32")
    .Input("coordinates: float32")
    .Output("grad_coords: float32");



