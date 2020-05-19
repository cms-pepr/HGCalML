// accumulate_knn_kernel.h
#ifndef ACCUMULATE_KNN_ND_KERNEL_H
#define ACCUMULATE_KNN_ND_KERNEL_H

#include "accumulate_knn_kernel.h"
//for this define #define ACCUMULATE_KNN_EXPONENT 1

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct AccumulateKnnNdOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coord,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,
            int *d_out_maxidxs,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,

            const int n_out_feat,

            const int n_moments);
};


}  // namespace functor
}  // namespace tensorflow

#endif //ACCUMULATE_KNN_ND_KERNEL_H
