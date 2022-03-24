
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BinByCoordinates")
    .Attr("calc_n_per_bin: bool")
    .Input("coordinates: float")
    .Input("row_splits: int32")
    .Input("bin_width: float")
    .Input("nbins: int32")
    .Output("bin_assignment: int32") //non-flat
    .Output("flat_bin_assignment: int32") //flat
    .Output("nperbin: int32"); //corresponding to the flat assignment
