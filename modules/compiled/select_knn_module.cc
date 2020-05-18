#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("SelectKnn")
    .Attr("n_neighbours: int")
    .Input("coords: float32")
    .Output("indices: int32");



