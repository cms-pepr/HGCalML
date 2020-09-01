
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "latent_space_grid_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

//helpers here, static

//scales with ic(fully) and rs(partially)
static void get_n_cells(
        const float *d_coords,
        const int *row_splits,

        const float *max_coords, //one per rs and dim
        const float *min_coords, //one per rs and dim

        int * n_cells_tot_per_rs,
        float * adj_cell_sizes,
        int * n_cells_per_rs_coord,

        const int n_coords,
        const int n_rs,

        const int irs,

        const float size,
        const int min_cells
){

    //printf("get_n_cells\n");

    int nthisrs=1;
    for(int ic=0;ic<n_coords;ic++){

        float max = max_coords[I2D(irs,ic,n_coords)];
        float min = min_coords[I2D(irs,ic,n_coords)];
        int n_ic = (max-min)/size;
        if(n_ic < min_cells)
            n_ic = min_cells;
        nthisrs *= n_ic;

       // printf("ic %d: n_ic %d\n",ic,n_ic);

        adj_cell_sizes[I2D(irs,ic,n_coords)] = (max-min)/(float)n_ic * 1.00001;//avoid edge effects
        n_cells_per_rs_coord[I2D(irs,ic,n_coords)] = n_ic;
    }

    //printf("ntot per rs %d: %d\n",irs,nthisrs);
    n_cells_tot_per_rs[irs] = nthisrs;
}

static void assign_v_to_cells(
        const float *d_coords,
        const int *row_splits,

        const float *min_coords, //one per rs and dim
        const int * n_cells_per_rs_coord,
        const int * n_cells_per_rs,
        const float * adj_cell_sizes,

        int * asso_vert_to_global_cell, //1D C-format
        int * n_vert_per_global_cell,

        const int n_coords,
        const int irs){ //1D C-format

   // printf("assign_v_to_cells\n");

    size_t startvert = row_splits[irs];
    size_t endvert = row_splits[irs+1];

    //potential opt FIXME
    int cellidx_offset=0;
    for(size_t i=0;i<irs;i++){
        cellidx_offset +=  n_cells_per_rs[i];
    }

   // printf("offset %d\n",cellidx_offset);

    for (int iv = startvert; iv < endvert; iv++) {

        int cellidx = cellidx_offset;//make this a rs offset
        int multiplier=1;

        for (int ic = n_coords - 1; ic > -1; ic--){//reverse for indexing


            const float& csize = adj_cell_sizes[I2D(irs,ic,n_coords)];
            float normcoord = d_coords[I2D(iv,ic,n_coords)] - min_coords[I2D(irs,ic,n_coords)];  //maybe add this offset in a different kernel TBI
            int thisidx = normcoord / csize;

            cellidx += multiplier*thisidx;
            multiplier *= n_cells_per_rs_coord[I2D(irs,ic,n_coords)];

        }
        asso_vert_to_global_cell[iv] = cellidx;
        //make atomic for parallel!
        n_vert_per_global_cell[cellidx] += 1;

    }


}


static void make_pseudo_rs(
        const int * n_vert_per_global_cell,
        int * pseudo_rs,
        const int n_cells){

    //printf("make_pseudo_rs\n");

    int total=0;
    int ice=0;
    for(;ice<n_cells;ice++){
        pseudo_rs[ice]=total;
        //printf("nvert per cell %d : %d\n",ice,n_vert_per_global_cell[ice]);
        total+=n_vert_per_global_cell[ice];
    }
    pseudo_rs[n_cells]=total;

}


static void make_resort_idxs(
        const int * pseudo_rs,
        const int * asso_vert_to_global_cell,

        int * n_vert_per_global_cell_filled,
        int * resort_idxs,

        const int n_cell,
        const int n_vert
        ){

   // printf("make_resort_idxs\n");

    for(size_t ice=0;ice<n_cell;ice++){//parallel over ice!

        size_t offset = pseudo_rs[ice];


        for(size_t iv=0;iv<n_vert;iv++){

            if( asso_vert_to_global_cell[iv] == ice){ //parallel over ice avoids bad atomic here
                size_t idx = n_vert_per_global_cell_filled[ice];
                resort_idxs[offset + idx] = iv;
                n_vert_per_global_cell_filled[ice] += 1;
            }
        }
    }
}


template<typename dummy>
struct LatentSpaceGetGridSizeOpFunctor<CPUDevice, dummy>  {
    void operator()(
            const CPUDevice &d,

            const float *d_coords,
            const int *row_splits,

            const float *max_coords, //one per rs and dim
            const float *min_coords, //one per rs and dim

            const int n_coords,
            const int n_rs,

            //calculates:
            int * n_cells_tot_per_rs,       // temp rs
            float * adj_cell_sizes,         // temp rs x c
            int * n_cells_per_rs_coord,     // temp rs x c
            int & n_pseudo_rs,

            const float size,
            int min_cells

            ){



        //copy rs to cpu here

        for(size_t irs=0;irs<n_rs-1;irs++){

            size_t nvertrs = row_splits[irs+1]-row_splits[irs];

            get_n_cells(d_coords,row_splits,max_coords,min_coords,
                    n_cells_tot_per_rs,adj_cell_sizes,n_cells_per_rs_coord,
                    n_coords,n_rs,irs,size,min_cells);
        }

        //cpu!
        n_pseudo_rs=1;//rs format
        for(size_t i=0;i<n_rs-1;i++){
            n_pseudo_rs+=n_cells_tot_per_rs[i];
        }

    }
};


// CPU specialization
template<typename dummy>
struct LatentSpaceGridOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

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

            const float size) {

        for(size_t i=0;i<n_pseudo_rs-1;i++)
            n_vert_per_global_cell[i]=0;

        for(size_t irs=0;irs<n_rs-1;irs++){

            assign_v_to_cells(d_coords,row_splits,min_coords,n_cells_per_rs_coord,n_cells_tot_per_rs,
                    adj_cell_sizes,
                    asso_vert_to_global_cell,n_vert_per_global_cell,
                    n_coords,irs);

            /*
             * int * asso_v_to_cell_idx, //1D C-format
               int * n_vert_per_global_cell,
             */

        }
        make_pseudo_rs(n_vert_per_global_cell, pseudo_rs,n_pseudo_rs-1);
        //this can just run in some thread as long as pseudo_rs are not used


        for(int i=0;i<n_pseudo_rs-1;i++)
            n_vert_per_global_cell_filled[i]=0;

        make_resort_idxs(pseudo_rs,asso_vert_to_global_cell,n_vert_per_global_cell_filled,resort_idxs,
                n_pseudo_rs-1.,n_vert);
      //  int * n_vert_per_global_cell_filled,
      //  int * resort_idxs,

        //now we have the minimal ingredients:
        // resort_idxs and pseudo_rs

    }
};

template<typename Device>
class LatentSpaceGridOp : public OpKernel {
public:
    explicit LatentSpaceGridOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                context->GetAttr("size", &size_));
        OP_REQUIRES_OK(context,
                context->GetAttr("min_cells", &min_cells_));

    }


    void Compute(OpKernelContext *context) override {

        /*
         *
         *   .Attr("size: float")
    .Attr("min_cells: int") //same for all dimensions

    .Input("coords: float32")
    .Input("min_coord: float32") //per dimension
    .Input("max_coord: float32")
    .Input("row_splits: int32")

    .Output("select_idxs: float32")
    .Output("pseudo_rowsplits: int32");
         */

        const Tensor &t_coords = context->input(0);
        const Tensor &t_min_coords = context->input(1);
        const Tensor &t_max_coords = context->input(2);
        const Tensor &t_row_splits = context->input(3);


        const int n_vert = t_coords.dim_size(0);
        const int n_coords = t_coords.dim_size(1);

        const int n_rs = t_row_splits.dim_size(0);


        TensorShape shape_nrs;
        shape_nrs.AddDim(n_rs);
        Tensor t_n_cells_tot_per_rs;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,shape_nrs,&t_n_cells_tot_per_rs));


        TensorShape shape_nrsmone_nc;
        shape_nrsmone_nc.AddDim(n_rs-1);
        shape_nrsmone_nc.AddDim(n_coords);

        Tensor t_adj_cell_sizes;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_FLOAT ,shape_nrsmone_nc,&t_adj_cell_sizes));

        Tensor * t_n_cells_per_rs_coord=NULL;
        //OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,shape_nrs_nc,&t_n_cells_per_rs_coord));
        OP_REQUIRES_OK(context, context->allocate_output(2, shape_nrsmone_nc, &t_n_cells_per_rs_coord));


        int n_pseudo_rs=0;


        LatentSpaceGetGridSizeOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_coords.flat<float>().data() , // , const float *d_coords,
                t_row_splits.flat<int>().data() ,//const int *row_splits,

                t_max_coords.flat<float>().data() , //const float *max_coords, //one per rs and dim
                t_min_coords.flat<float>().data() ,  //const float *min_coords, //one per rs and dim

                n_coords,
                n_rs,

                //calculates:
                t_n_cells_tot_per_rs.flat<int>().data() ,//int * n_cells_tot_per_rs,       // temp
                t_adj_cell_sizes.flat<float>().data() , //float * adj_cell_sizes,         // temp
                t_n_cells_per_rs_coord->flat<int>().data(), //int * n_cells_per_rs_coord,     // temp
                n_pseudo_rs,

                size_,
                min_cells_

        );



        TensorShape shape_nvert;
        shape_nvert.AddDim(n_vert);
        Tensor *t_resort_idxs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape_nvert, &t_resort_idxs));


        TensorShape shape_nprs;
        shape_nprs.AddDim(n_pseudo_rs);
        Tensor *t_pseudo_rs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, shape_nprs, &t_pseudo_rs));


        //temps

        Tensor *t_asso_vert_to_global_cell=NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, shape_nvert, &t_asso_vert_to_global_cell));

        TensorShape shape_ncells;
        shape_ncells.AddDim(n_pseudo_rs-1);

        Tensor t_n_vert_per_global_cell;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,shape_ncells,&t_n_vert_per_global_cell));

        Tensor t_n_vert_per_global_cell_filled;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,shape_ncells,&t_n_vert_per_global_cell_filled));

        //Tensor *t_resort_inv = NULL;
        //OP_REQUIRES_OK(context, context->allocate_output(1, shape_t_resort, &t_resort_inv));
        //
        ////resort: standard sorting -> pseudo row split sorting
        ////resort_inv: pseudo row split sorting -> standard sorting
        //
        //Tensor *cells_to_orig_sorting = NULL;
        //OP_REQUIRES_OK(context, context->allocate_output(2, shape_t_resort, &cells_to_orig_sorting));
        //
        ////contains cell index (global) for each vertex



        //create asso_cell_idx tensor, dim(V)

        /*
         * first, we need to get the number of cells.
         * this already requires a templated call
         */
        //row split loop internal

        LatentSpaceGridOpFunctor<Device, int>()(
                context->eigen_device<Device>(),                                   //            const float min_beta                 / const float min_beta

                t_coords.flat<float>().data() , // , const float *d_coords,
                t_row_splits.flat<int>().data() ,//const int *row_splits,

                t_max_coords.flat<float>().data() , //const float *max_coords, //one per rs and dim
                t_min_coords.flat<float>().data() ,  //const float *min_coords, //one per rs and dim



                t_n_cells_tot_per_rs.flat<int>().data() ,//int * n_cells_tot_per_rs,       // temp
                t_adj_cell_sizes.flat<float>().data() , //float * adj_cell_sizes,         // temp
                t_n_cells_per_rs_coord->flat<int>().data(), //int * n_cells_per_rs_coord,     // temp
                n_pseudo_rs,

                //calculates
                t_asso_vert_to_global_cell->flat<int>().data(), // (nv) maps each vertex to global cell index.
                t_n_vert_per_global_cell.flat<int>().data(), //almost the same as pseudo_rs
                t_n_vert_per_global_cell_filled.flat<int>().data(), //almost the same as pseudo_rs

                t_resort_idxs->flat<int>().data() ,
                t_pseudo_rs->flat<int>().data() ,

                n_vert,
                n_coords,
                n_rs,

                size_

        );




    }

private:
    float size_;
    int min_cells_;
};

REGISTER_KERNEL_BUILDER(Name("LatentSpaceGrid").Device(DEVICE_CPU), LatentSpaceGridOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct LatentSpaceGetGridSizeOpFunctor<GPUDevice, int>;
extern template struct LatentSpaceGridOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("LatentSpaceGrid").Device(DEVICE_GPU), LatentSpaceGridOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
