
#ifndef LATENT_SPACE_GRID_KERNEL_H
#define LATENT_SPACE_GRID_KERNEL_H

namespace tensorflow {
namespace functor {
template<typename Device, typename dummy>
struct LatentSpaceGetGridSizeOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coords,
            const int *row_splits,

            const float *max_coords, //one per rs and dim
            const float *min_coords, //one per rs and dim

            const int n_coords,
            const int n_rs,

            //calculates:
            int * n_cells_tot_per_rs,
            float * adj_cell_sizes,
            int * n_cells_per_rs_coord,
            int & n_pseudo_rs,

            const float size,
            int min_cells
            );
};

template<typename Device, typename dummy>
struct LatentSpaceGridOpFunctor {
    void operator()(

            const Device &d,

            const float *d_coords,
            const int *row_splits,

            const float *max_coords, //one per rs and dim
            const float *min_coords, //one per rs and dim


            const int * n_cells_tot_per_rs,
            const float * adj_cell_sizes,
            const int * n_cells_per_rs_coord,
            const int  n_pseudo_rs,

            //calculates
            int * asso_vert_to_global_cell, // (nv) maps each vertex to global cell index.
            int * n_vert_per_global_cell, //almost the same as pseudo_rs
            int * n_vert_per_global_cell_filled, //almost the same as pseudo_rs

            int * resort_idxs,
            int * pseudo_rs,

            const int n_vert,
            const int n_coords,
            const int n_rs,

            const float size
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //BUILD_CONDENSATES_KERNEL_H

