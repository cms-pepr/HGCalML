
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
            int * d_flat_assigned_bin,
            int * d_n_per_bin,

            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_total_bins,
            const bool calc_n_per_bin
    );



};


template<typename Device, typename dummy>
struct BinByCoordinatesNbinsHelperOpFunctor {
    void operator()(
            const Device &d,
            const int * n_bins,
            int * out_tot_bins,
            int n_nbins,
            int nrs
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BIN_BY_COORDINATES_KERNEL_H



