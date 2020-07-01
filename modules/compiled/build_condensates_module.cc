#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("BuildCondensates")
    .Attr("radius: float")
    .Attr("min_beta: float")
    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("beta_sorting: int32")
    .Input("features: float32")
    .Input("row_splits: int32")
    .Output("summed_features: float32")
    .Output("asso_idx: int32");



