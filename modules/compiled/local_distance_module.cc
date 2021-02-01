#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("LocalDistance")
    .Input("coordinates: float32")
    .Input("neighbour_idxs: int32")
    .Output("distances: float32");



