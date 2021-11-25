#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*

REGISTER_OP("LocalGroup")
    .Input("neighbour_idxs: int32") // V x K
    .Input("hierarchy_idxs: int32") // V x 1  per group
    .Input("hierarchy_score: float") // V x 1 score per group
    .Attr("score_threshold: float")
    .Input("global_idxs: int32") // V x 1
    .Input("row_splits: int32") // RS
    .Output("out_row_splits: int32") //RS'
    .Output("selection_neighbour_idxs: int32") //V' x K'
    .Output("backscatter_idxs: int32") //V
    .Output("n_in_group: int32"); //V x 1

*/

REGISTER_OP("LocalGroup")
    .Input("neighbour_idxs: int32") // V x K
    .Input("hierarchy_idxs: int32") // V x 1  per group
    .Input("hierarchy_score: float") // V x 1 score per group
    .Attr("score_threshold: float")
    .Input("row_splits: int32") // RS
    .Output("out_row_splits: int32") //RS'
    .Output("directed_neighbour_indices: int32") //V x K
    .Output("selection_indices: int32") //V' x 1
    .Output("backgather_idxs: int32"); //V





