
#ifndef SLICING_KNN_KERNEL_H
#define SLICING_KNN_KERNEL_H
#include <vector>

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct SlicingKnnOpFunctor {
    void operator()(const Device &d,

            const float *d_coords_sorted, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!
            const int* d_n_bins, // accessible only on GPU!!!
            const float* d_coords_min, // accessible only on GPU!!!
            const float* d_coords_max, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            const int n_rs,

            std::vector<int> features_to_bin_on
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SLICING_KNN_KERNEL_H

