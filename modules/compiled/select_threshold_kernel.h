
#ifndef SELECT_THRESHOLD_KERNEL_H
#define SELECT_THRESHOLD_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct SelectThresholdOpFunctor {
    void operator()(
            const Device &d,

            const float *d_th_values,
            const int* d_row_splits,

            int *d_scatter_idxs,
            int *d_new_rowsplits,

            int * n_scatter_idxs,

            const int n_vert,

            const int n_rs,

            const float threshold
            );
};

template<typename Device, typename dummy>
struct CopyOutputSelectThresholdOpFunctor {
    void operator()(
            const Device &d,

            int *d_scatter_idxs,

            int *d_tmp_scatter_idxs,

            int  n_scatter_idxs

    );
};

}  // namespace functor
}  // namespace tensorflow

#endif //SELECT_KNN_KERNEL_H

