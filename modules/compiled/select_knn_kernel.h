
#ifndef SELECT_KNN_KERNEL_H
#define SELECT_KNN_KERNEL_H

namespace tensorflow {
namespace functor {
namespace selknn{
enum mask_mode_en{mm_none, mm_acc, mm_scat};
enum mask_logic_en{ml_xor, ml_and};
}
template<typename Device, typename dummy>
struct SelectKnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coord,
            const int* d_row_splits,
            const int* mask,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            selknn::mask_mode_en mask_mode,
            selknn::mask_logic_en mask_logic
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SELECT_KNN_KERNEL_H

