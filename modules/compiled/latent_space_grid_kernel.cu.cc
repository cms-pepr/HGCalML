//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU


#include "tensorflow/core/framework/op_kernel.h"
#include "latent_space_grid_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {



typedef Eigen::GpuDevice GPUDevice;

__global__
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

        const int irs_dummy,

        const float size,
        const int min_cells
){


    int irs =  blockIdx.x * blockDim.x + threadIdx.x;
    if(irs >= n_rs-1)
        return;

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

__global__
static void set_n_vert_per_global_cell_zero(
        int* n_vert_per_global_cell,
        int* n_vert_per_global_cell_filled,
        int ncells){

    int ic =  blockIdx.x * blockDim.x + threadIdx.x;
    if(ic >= ncells)
        return;
    n_vert_per_global_cell[ic]=0;
    n_vert_per_global_cell_filled[ic]=0;
}


__global__
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


    size_t startvert = row_splits[irs];
    size_t endvert = row_splits[irs+1];

    //for(int iv=startvert;iv<endvert;iv++){
    int iv =  blockIdx.x * blockDim.x + threadIdx.x + startvert;
    if(iv >= endvert)
        return;

    //potential opt FIXME
    int cellidx_offset=0;
    for(size_t i=0;i<irs;i++){
        cellidx_offset +=  n_cells_per_rs[i];
    }





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
    //move this to a cell parallel loop
    //n_vert_per_global_cell[cellidx]+=1;
    atomicAdd(&(n_vert_per_global_cell[cellidx]) , 1);
    //}
}


__global__
static void make_pseudo_rs(
        const int * n_vert_per_global_cell,
        int * pseudo_rs,
        const int n_cells){


   // printf("make_pseudo_rs");
    int total=0;
    int ice=0;
    for(;ice<n_cells;ice++){
        pseudo_rs[ice]=total;
      //  printf("nvert per cell %d : %d\n",ice,n_vert_per_global_cell[ice]);
        total+=n_vert_per_global_cell[ice];
    }
    pseudo_rs[n_cells]=total;
   // printf("total vert %d\n",total);

}


__global__
static void make_resort_idxs(
        const int * pseudo_rs,
        const int * asso_vert_to_global_cell,

        int * n_vert_per_global_cell_filled,
        int * resort_idxs,

        const int n_cell,
        const int n_vert
        ){

    size_t ice= blockIdx.x * blockDim.x + threadIdx.x;
    if(ice >= n_cell)
        return;


    size_t offset = pseudo_rs[ice];

    for(size_t iv=0;iv<n_vert;iv++){

        if( asso_vert_to_global_cell[iv] == ice){ //parallel over ice avoids bad atomic here
            size_t idx = n_vert_per_global_cell_filled[ice];
            resort_idxs[offset + idx] = iv; //offset guarantees no race
            n_vert_per_global_cell_filled[ice] += 1;
        }
    }

}


template<typename dummy>
struct LatentSpaceGetGridSizeOpFunctor<GPUDevice, dummy>  {
    void operator()(
            const GPUDevice &d,

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


        grid_and_block b(n_rs,4);

        get_n_cells<<<b.grid(), b.block()>>>(d_coords,row_splits,max_coords,min_coords,
                n_cells_tot_per_rs,adj_cell_sizes,n_cells_per_rs_coord,
                n_coords,n_rs,0,size,min_cells);


        cudaDeviceSynchronize();

        std::vector<int> cpu_n_cells_tot_per_rs(n_rs-1);
        cudaMemcpy(&cpu_n_cells_tot_per_rs.at(0),n_cells_tot_per_rs,(n_rs-1)*sizeof(int),cudaMemcpyDeviceToHost);

        //cpu!
        n_pseudo_rs=1;//rs format
        for(size_t i=0;i<n_rs-1;i++){
            n_pseudo_rs+=cpu_n_cells_tot_per_rs[i];
        }


    }
};


// CPU specialization
template<typename dummy>
struct LatentSpaceGridOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

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

        // not needed init with 0 for(size_t i=0;i<n_pseudo_rs-1;i++)
        // not needed init with 0     n_vert_per_global_cell[i]=0;

        grid_and_block gb2(n_pseudo_rs-1, 512);

        set_n_vert_per_global_cell_zero<<<gb2.grid(),gb2.block()>>>(
                n_vert_per_global_cell,n_vert_per_global_cell_filled,n_pseudo_rs-1);


        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost); //Async if needed, but these are just a few kB


        for(size_t irs=0;irs<n_rs-1;irs++){

            int nvert = cpu_rowsplits.at(irs+1) - cpu_rowsplits.at(irs);

            grid_and_block gb(nvert,768);

            assign_v_to_cells<<<gb.grid(),gb.block()>>>(d_coords,row_splits,min_coords,n_cells_per_rs_coord,n_cells_tot_per_rs,
                    adj_cell_sizes,
                    asso_vert_to_global_cell,n_vert_per_global_cell,
                    n_coords,irs);


            cudaDeviceSynchronize();
        }

        make_pseudo_rs<<<1,1>>>(n_vert_per_global_cell, pseudo_rs,n_pseudo_rs-1);
        //this can just run in some thread as long as pseudo_rs are not used

        cudaDeviceSynchronize();

        //for(int i=0;i<n_pseudo_rs-1;i++)
        //    n_vert_per_global_cell_filled[i]=0;

        grid_and_block cell_gb(n_pseudo_rs-1, 512);

        make_resort_idxs<<<cell_gb.grid(),cell_gb.block()>>>(pseudo_rs,asso_vert_to_global_cell,n_vert_per_global_cell_filled,resort_idxs,
                n_pseudo_rs-1.,n_vert);
      //  int * n_vert_per_global_cell_filled,
      //  int * resort_idxs,

        cudaDeviceSynchronize();

        //now we have the minimal ingredients:
        // resort_idxs and pseudo_rs

    }
};

template struct LatentSpaceGetGridSizeOpFunctor<GPUDevice, int>;
template struct LatentSpaceGridOpFunctor<GPUDevice, int>;
}//functor
}//tf

#endif  // GOOGLE_CUDA
