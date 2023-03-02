#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("BinnedAssignToCondensates")
    .Attr("beta_threshold: float")
    .Attr("assign_by_max_beta: bool")

    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("beta: float32")
    .Input("no_condensation_mask: int32")
    .Input("bins_flat: int32")
    .Input("bin_splits: int32")
    .Input("indices_to_filtered: int32")
    .Input("original_indices: int32")
    .Input("row_splits: int32")
    .Input("n_bins: int32")
    .Input("bin_widths: float32")
    .Input("ccoords_h: float32")
    .Input("dist_h: float32")
    .Input("beta_h: float32")
    .Input("no_condensation_mask_h: int32")
    .Input("row_splits_h: int32")
    .Output("assigned_h: int32")
    .Output("assigned: int32")
    .Output("alpha_indices: int32")
    .Output("association: int32")
    .Output("n_condensates: int32");



