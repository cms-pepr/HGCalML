// push_knn_kernel.h
#ifndef PUSH_KNN_KERNEL_H
#define PUSH_KNN_KERNEL_H


namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct PushKnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_weights,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,

            int n_vert,
            int n_neigh,
            int n_feat);
};


}  // namespace functor
}  // namespace tensorflow

#endif //PUSH_KNN_KERNEL_H
