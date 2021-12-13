#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("AssignToCondensates")
    .Attr("radius: float")
    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("c_point_idx: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32");



