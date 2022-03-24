// accumulate_knn_kernel.h
#ifndef ACCUMULATE_KNN_KERNEL_H
#define ACCUMULATE_KNN_KERNEL_H


namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct AccumulateKnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_distances,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,
            int *d_out_maxidxs,

            int n_vert,
            int n_neigh,
            int n_feat,

            int n_out_feat,

            int n_moments,
            bool mean_and_max);
};


}  // namespace functor
}  // namespace tensorflow

#endif //ACCUMULATE_KNN_KERNEL_H
