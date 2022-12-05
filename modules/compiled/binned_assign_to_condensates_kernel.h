
#ifndef ASSIGN_TO_CONDENSATES_BINNED_KERNEL_H
#define ASSIGN_TO_CONDENSATES_BINNED_KERNEL_H

namespace tensorflow {
namespace functor {


template<typename Device, typename dummy>
struct BinnedCondensatesFinderOpFunctor {
    void operator()(
            const Device &d,
            const int dimensions,
            const int dimensions_binning,
            int*condensates_assigned_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            const int* no_condensation_mask_h,
            int* high_assigned_status_h,
            const int* original_indices_h,
            const int*row_splits_h,
            const float* ccoords,
            const float* dist,
            const int* no_condensation_mask,
            const int* indices_to_filtered,
            float* assignment_min_distance,
            int* assignment,
            int* association,
            int* n_condensates,
            int* alpha_indices,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int*row_splits,
            const int num_rows,
            const bool assign_by_max_beta
            );
};


}  // namespace functor
}  // namespace tensorflow

#endif //ASSIGN_TO_CONDENSATES_KERNEL_H

