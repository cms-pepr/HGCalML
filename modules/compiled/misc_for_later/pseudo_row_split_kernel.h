
#ifndef PSEUDO_ROW_SPLIT_KERNEL_H
#define PSEUDO_ROW_SPLIT_KERNEL_H

namespace tensorflow {
namespace functor {

/*
 *
    .Input("asso_idx: int32")
    .Input("n_per_rs: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("pseudo_rs: int32")
 */

template<typename Device, typename dummy>
struct PseudoRowSplitOpFunctor {
    void operator()(

            const Device &d,

            const int *d_asso_idx,
            const int *d_n_per_rs,
            const int *row_splits,

            int *asso_idx,
            int *pseudo_rs,

            const int n_vert,
            const int n_pseudo_rs,
            const int n_rs

            );
};

template<typename Device, typename dummy>
struct PseudoRowSplitCountOpFunctor {
    void operator()(

            const Device &d,

            const int *d_n_per_rs,

            int& ntotal,
            const int n_prs

            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BUILD_CONDENSATES_KERNEL_H

