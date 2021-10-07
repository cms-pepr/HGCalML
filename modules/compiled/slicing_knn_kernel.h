
#ifndef SLICING_KNN_KERNEL_H
#define SLICING_KNN_KERNEL_H
#include <vector>

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct SlicingKnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coords_sorted, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            std::vector<float> phase_space_bin_boundary,
            const int n_rs,

            std::vector<int> n_bins,
            std::vector<int> features_to_bin_on
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SLICING_KNN_KERNEL_H

