
#ifndef OC_HELPER_M_INDICES_KERNEL_H
#define OC_HELPER_M_INDICES_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct MIndicesMaxUqOpFunctor { //just because access needs to be different for GPU and CPU
    void operator()(
            const Device &d,
            const int *d_maxunique,
            int * n_max_per_unique
            );
};

template<typename Device, typename dummy>
struct MIndicesOpFunctor {
    void operator()(
            const Device &d,

            const int *d_truthidx,
            const int *d_unique_idx,
            const int *rs,

            int * out_idx,
            int * m_not,

            const int n_vert,
            const int n_unique,
            const int n_max_per_unique,
            const int n_rs,
            const int n_max_in_rs,

            bool calc_m_not

            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //OC_HELPER_M_INDICES_KERNEL_H

