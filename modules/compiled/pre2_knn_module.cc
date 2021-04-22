#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("Pre2Knn")
    .Attr("n_bins_x: int")
    .Attr("n_bins_y: int")
    .Input("coords: float32")
    .Output("sorted_coords: float32")
    .Output("auxaliry_knn_arrays: int32") // 1D array containing following
                                          // values/arrays [n_bins_x, n_bins_y,
                                          // n_vtx_per_bin_cumulative[n_bins_x*n_bins_y+1],
                                          // bin_neighbours[9*n_bins_x*n_bins_y],
                                          // bin_vtx_assoc[V]]
    ;

