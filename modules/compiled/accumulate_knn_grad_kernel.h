
#ifndef ACCUMULATE_KNN_GRAD_KERNEL_H
#define ACCUMULATE_KNN_GRAD_KERNEL_H

namespace tensorflow {

namespace functor {


template<typename Device, typename dummy>
struct AccumulateKnnGradOpFunctor {
    void operator()(
            const Device &d,

            const float *d_grad_from_out_features,
            const float *d_coord, // sum(V) x S
            const float *d_feat, // sum(V) x F
            const int *d_max_feat_indices,
            const int * d_neigh_indices,

            float *d_out_grad_coords,
            float *d_out_grad_features,

            int n_vert,
            int n_neigh,
            int n_coords,
            int n_feat,

            int n_grad_from_out_feat,

            int n_moments);
};

}  // namespace functor

}  // namespace tensorflow

#endif //ACCUMULATE_KNN_GRAD_KERNEL_H

