
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
            const float *d_dist,
            const float *d_tosum,
            const int *row_splits,

            int *asso_idx,
            int *is_cpoint,
            float * temp_betas,
            int *n_condensates,
            float *temp_tosum,
            float *summed,

            const int n_vert,
            const int n_ccoords,
            const int n_sumf,

            const int n_rs,

            const float radius,
            const float min_beta,
            const bool soft,
            const bool sum
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BUILD_CONDENSATES_KERNEL_H

