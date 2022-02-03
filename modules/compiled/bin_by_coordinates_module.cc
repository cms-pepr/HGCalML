
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BinByCoordinates")
    .Input("coordinates: float")
    .Input("row_splits: int32")
    .Input("bin_width: float")
    .Input("nbins: int32")//same in all dimensions
    .Output("output: int32"); 
