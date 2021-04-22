
#ifndef PRE2_KNN_KERNEL_H
#define PRE2_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct Pre2KnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coords, // accessible only on GPU!!!
            float *d_coords_sorted,
            int *auxaliry_knn_arrays,
            const int V, // # of vertices
            const int n_coords,
            const int n_bins_x,
            const int n_bins_y
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //PRE2_KNN_KERNEL_H

