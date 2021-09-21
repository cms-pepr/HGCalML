// local_group_kernel.h
#ifndef LOCAL_GROUP_KERNEL_H
#define LOCAL_GROUP_KERNEL_H

namespace tensorflow {
namespace functor {

/*
 * works exclusively on global indices (not caring about row splits).
 * the row splits are assumed to be accounted for in the input indices of the neighbours
 * (d_neigh_idxs)
 *
 * The performance can be increased by a small factor <10 if row splits
 * are used for parallelisation anyway, because we know vertices from differen row splits
 * don't talk to each other, we can parallelise along this dimension.
 * So far, this is not implemented, but foreseen in the interface
 *
 */
template<typename Device, typename dummy>
struct LocalGroupOpFunctor {
    void operator()(
            const Device &d,

            const int *d_neigh_idxs,
            const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op

            const float * d_hierarchy_score,

            const float score_threshold,

            const int * d_global_idxs, //global index of each vertex: V x 1, not global dimension!
            const int * d_row_splits,  //keeps dimensions: N_rs x 1

            int * mask,
            int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
            int * n_sel_vtx,
            int * d_out_row_splits,

            const int n_in_vert,
            const int n_neigh,
            const int n_row_splits,

            //globals for bookkeeping. dimension n_global_vert_g!
            int *d_out_backscatter, //which global index each vertex is associated to V x 1
            int n_global_vert_g
    );



};

//adapt row splits through simple copies

template<typename Device, typename dummy>
struct LocalGroupTruncateOpFunctor {
    void operator()(
            const Device &d,

            const int *d_in_selection_idxs, //which ones to keep
            int *d_out_selection_idxs,
            int *d_out_ningroup,
            int newsize,
            int n_neigh
    );
//needs a truncate functor, too? or do this with mallocs?
};


}  // namespace functor
}  // namespace tensorflow

#endif //LOCAL_CLUSTER_KERNEL_H
