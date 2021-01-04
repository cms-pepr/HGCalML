// local_distance_kernel.h
#ifndef LOCAL_DISTANCE_KERNEL_H
#define LOCAL_DISTANCE_KERNEL_H

namespace tensorflow {
namespace functor {


template<typename Device, typename dummy>
struct LocalDistanceOpFunctor {
    void operator()(
            const Device &d,

            const int *d_neigh_idxs,
            const float *d_coords,

            float * d_distances,

            const int n_coords,
            const int n_in_vert,
            const int n_out_vert,
            const int n_neigh
    );



};



}  // namespace functor
}  // namespace tensorflow

#endif //LOCAL_CLUSTER_KERNEL_H
