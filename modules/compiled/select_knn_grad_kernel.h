
#ifndef SELECT_KNN_GRAD_KERNEL_H
#define SELECT_KNN_GRAD_KERNEL_H

namespace tensorflow {
namespace functor {

/*
.Input("grad_distances: float32")
.Input("indices: int32")
.Input("distances: float32")
.Input("coordinates: float32")
.Output("grad_coords: float32")
*/

template<typename Device, typename dummy>
struct SelectKnnGradOpFunctor {
    void operator()(
            const Device &d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,

            float * d_grad_coord,

            const int n_vert,
            const int n_neigh,
            const int n_coords
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SELECT_KNN_KERNEL_H

