#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("LatentSpaceGrid")
    .Attr("size: float")
    .Attr("min_cells: int") //same for all dimensions
    .Input("coords: float32")
    .Input("min_coord: float32") //per dimension
    .Input("max_coord: float32")
    .Input("row_splits: int32")
    .Output("select_idxs: int32")
    .Output("pseudo_rowsplits: int32")
    .Output("n_cells_per_rs_coord: int32")
    .Output("vert_to_cell: int32");



