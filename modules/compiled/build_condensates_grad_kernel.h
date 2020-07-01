
#ifndef BUILD_CONDENSATES_GRAD_KERNEL_H
#define BUILD_CONDENSATES_GRAD_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct BuildCondensatesGradOpFunctor {
    void operator()(

            const Device &d,

            const float *sum_features_grad,
            const int *asso_idx,
            float *features_grad,

            const int n_vert,
            const int n_feat
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BUILD_CONDENSATES_KERNEL_H

