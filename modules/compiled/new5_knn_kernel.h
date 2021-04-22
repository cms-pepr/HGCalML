
#ifndef NEW5_KNN_KERNEL_H
#define NEW5_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct New5KnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coords_sorted, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,

            const int n_bins,
            const int* n_vtx_per_bin_cumulative, // size: n_bins_x*n_bins_y+1
            const int* bin_neighbours, // size: 9*n_bins_x*n_bins_y, bin itself + up to 8 neighbour bins
            const int* vtx_bin_assoc // size: V
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //NEW5_KNN_KERNEL_H

