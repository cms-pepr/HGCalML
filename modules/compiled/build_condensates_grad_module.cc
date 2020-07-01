#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("BuildCondensatesGrad")
    .Input("sumfeat_grad: float32")
    .Input("asso_idx: int32")
    .Output("feat_grad: float32");



