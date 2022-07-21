
#ifndef ASSIGN_TO_CONDENSATES_BINNED_KERNEL_H
#define ASSIGN_TO_CONDENSATES_BINNED_KERNEL_H

namespace tensorflow {
namespace functor {

/*
 *
    .Attr("radius: float")
    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("c_point_idx: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32");

 */

template<typename Device, typename dummy>
struct AssignToCondensatesBinnedOpFunctor {
    void operator()(

            const Device &d,

            const float *d_ccoords,
            const float *d_dist,
            const int *c_point_idx,

            const int *row_splits,

            int *asso_idx,

            const float radius,
            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_condensates
            );
};

//template<typename Device, typename dummy>
//struct BinnedCondensatesFinderOpFunctor {
//    void operator() (
//            const Device &d);
////            const float* ccoords_h,
////            const float* dist_h,
////            const float* beta_h,
////            const int* bins_flat_h,
////            const int* bin_splits_h,
////            const int* n_bins_h,
////            const float* bin_widths_h);
//};


template<typename Device, typename dummy>
struct BinnedCondensatesFinderOpFunctor {
    void operator()(
            const Device &d,
            const int dimensions,
            float*max_search_dist_binning_h,
            int*condensates_assigned_h,
            int*condensates_dominant_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            const int* bin_splits_h,
            const int* n_bins_h,
            const float* bin_widths_h,
            int* high_assigned_status_h,
            const int*row_splits_h,
            const float* ccoords,
            const float* dist,
            const float* beta,
            const int* indices_to_filtered,
            int* assigned,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int*row_splits,
            const int num_rows
            );
};


}  // namespace functor
}  // namespace tensorflow

#endif //ASSIGN_TO_CONDENSATES_KERNEL_H

