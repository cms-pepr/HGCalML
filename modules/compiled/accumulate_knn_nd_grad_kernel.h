#include "accumulate_knn_kernel.h"
#ifndef ACCUMULATE_KNN_ND_GRAD_KERNEL_H
#define ACCUMULATE_KNN_ND_GRAD_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct AccumulateKnnNdGradOpFunctor {
    void operator()(
            const Device &d,

            const float *d_grad_from_out_features,
            const float *d_grad_from_sum_features,

            const float *d_coord, // sum(V) x S
            const float *d_feat, // sum(V) x F
            const float *d_orig_out_feat,
            const float *d_orig_out_feat_sum,
            const int *d_max_feat_indices,
            const int * d_neigh_indices,

            float *d_out_grad_coords,
            float *d_out_grad_features,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,
            const int n_grad_from_out_feat,
            const int n_moments);
};

}  // namespace functor
}  // namespace tensorflow

#endif //ACCUMULATE_KNN_GRAD_KERNEL_H

