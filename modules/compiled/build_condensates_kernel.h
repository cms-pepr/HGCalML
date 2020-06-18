
#ifndef BUILD_CONDENSATES_KERNEL_H
#define BUILD_CONDENSATES_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct BuildCondensatesOpFunctor {
    void operator()(

            const Device &d,

            const float *d_ccoords,
            const float *d_betas,
            const int *beta_sorting,
            const float *features,
            const int *row_splits,


            float *summed_features,
            int *asso_idx,

            const int n_vert,
            const int n_feat,
            const int n_ccoords,

            const int n_rs,

            const float radius,
            const float min_beta
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BUILD_CONDENSATES_KERNEL_H

