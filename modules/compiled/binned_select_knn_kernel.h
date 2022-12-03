
#ifndef BINNED_SELECT_KNN_KERNEL_H
#define BINNED_SELECT_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct BinnedSelectKnnOpFunctor {
    void operator()(
            const Device &d,

            const float * d_coord,
            const int * d_bin_idx,
            const int * d_direction,
            const int * d_dim_bin_idx,

            const int * d_bin_boundaries,
            const int * d_n_bins,

            const float* d_bin_width,

            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_bin_dim,

            const int n_bboundaries,
            bool tf_compat,
            bool use_direction
    );



};

}  // namespace functor
}  // namespace tensorflow

#endif //BINNED_SELECT_KNN_KERNEL_H
