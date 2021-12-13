
#ifndef ASSIGN_TO_CONDENSATES_KERNEL_H
#define ASSIGN_TO_CONDENSATES_KERNEL_H

namespace tensorflow {
namespace functor {

/*
 *
    .Attr("radius: float")
    .Input("ccoords: float32")
    .Input("dist: float32")
    .Input("c_point_idx: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32");

 */

template<typename Device, typename dummy>
struct AssignToCondensatesOpFunctor {
    void operator()(

            const Device &d,

            const float *d_ccoords,
            const float *d_dist,
            const int *c_point_idx,

            const int *row_splits,

            int *asso_idx,

            const float radius,
            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_condensates
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //ASSIGN_TO_CONDENSATES_KERNEL_H

