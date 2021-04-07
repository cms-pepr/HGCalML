
#ifndef ACCUMULATE_KNN_GRAD_KERNEL_H
#define ACCUMULATE_KNN_GRAD_KERNEL_H

#include "accumulate_knn_kernel.h"

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct AccumulateKnnGradOpFunctor {
    void operator()(
            const Device &d,

            const float *d_grad_from_out_features,
            const float *d_distances, // sum(V) x S
            const float *d_feat, // sum(V) x F
            const int *d_max_feat_indices,
            const int * d_neigh_indices,

            float *d_out_grad_distances,
            float *d_out_grad_features,

            int n_vert,
            int n_neigh,
            int n_feat,

            int n_grad_from_out_feat,

            int n_moments,
            bool mean_and_max);
};

}  // namespace functor
}  // namespace tensorflow

#endif //ACCUMULATE_KNN_GRAD_KERNEL_H

