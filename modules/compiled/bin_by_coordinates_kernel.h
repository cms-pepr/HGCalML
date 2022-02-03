
#ifndef BIN_BY_COORDINATES_KERNEL_H
#define BIN_BY_COORDINATES_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct BinByCoordinatesOpFunctor {
    void operator()(

            const Device &d,
            const float * d_coords,
            const int * d_rs,

            const float * d_binswidth, //singleton
            const int* n_bins,//singleton

            int * d_assigned_bin,

            int n_vert,
            int n_coords,
            int n_rs
    );



};

}  // namespace functor
}  // namespace tensorflow

#endif //BIN_BY_COORDINATES_KERNEL_H
