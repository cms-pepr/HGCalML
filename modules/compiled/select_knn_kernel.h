
#ifndef SELECT_KNN_KERNEL_H
#define SELECT_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct SelectKnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coord,
            int *d_indices,

            const int n_vert,
            const int n_neigh,
            const int n_coords
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SELECT_KNN_KERNEL_H

