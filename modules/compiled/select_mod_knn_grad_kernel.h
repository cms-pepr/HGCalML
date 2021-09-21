
#ifndef SELECT_MOD_KNN_GRAD_KERNEL_H
#define SELECT_MOD_KNN_GRAD_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct SelectModKnnGradOpFunctor {
    void operator()(
            const Device &d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,
            const float *d_coord_mod,

            float * d_grad_coord,
            float * d_grad_coord_mod,

            const int n_vert,
            const int n_neigh,
            const int n_coords
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SELECT_KNN_KERNEL_H

