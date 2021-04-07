
#ifndef NEIGHBOUR_COVARIANCE_KERNEL_H
#define NEIGHBOUR_COVARIANCE_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct NeighbourCovarianceOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coords,
            const float *d_feats,
            const int* d_n_dixs,

            float * d_covariance,
            float * d_means,


            const int n_vert,
            const int n_coords,
            const int n_feat,
            const int n_neigh,
            const int n_vert_out
            );
};


}  // namespace functor
}  // namespace tensorflow

#endif //NEIGHBOUR_COVARIANCE_KERNEL_H

