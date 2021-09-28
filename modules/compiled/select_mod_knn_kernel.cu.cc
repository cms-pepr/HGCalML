//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "select_mod_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

namespace gpu{
__device__
static float calculateDistance(size_t i_v, size_t j_v, const float * d_coord,
        const float * d_coord_mod,
        size_t n_coords,
        float maxrad){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float modi = 0;
        float modj = 0;
        float coorddist = 0;
        for(size_t j=0;j<n_coords;j++){ //contract over axis 1
            float Rivji = d_coord_mod[I3D(i_v, j, i, n_coords, n_coords)];
            coorddist += Rivji * (d_coord[I2D(i_v,j,n_coords)]-d_coord[I2D(j_v,j,n_coords)]);
            if(maxrad && coorddist > maxrad)
                return 10.*maxrad;//early stop
        }
        float dist = coorddist;//modi-modj;
        distsq += dist*dist;
    }
    return distsq;
}


__device__
static int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

    maxdist=0;
    int maxidx=0;
    if(n_neigh < 2)
        return maxidx;
    for(size_t n=1;n<n_neigh;n++){ //0 is self
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

__global__
static void set_defaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;
    const size_t n =  blockIdx.y * blockDim.y + threadIdx.y;
    if(n >= n_neigh)
        return;

    if(n){
        if(tf_compat)
            d_indices[I2D(i_v,n,n_neigh)] = i_v;
        else
            d_indices[I2D(i_v,n,n_neigh)] = -1;
    }
    else{
        d_indices[I2D(i_v,n,n_neigh)] = i_v;
    }
    d_dist[I2D(i_v,n,n_neigh)] = 0;


}


__global__
static void select_knn_kernel(
        const float *d_coord,
        const float *d_coord_mod,
        const int* d_row_splits,
        const int* d_mask,
        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int j_rs,
        const bool tf_compat,
        const float max_radius,
        selknn::mask_mode_en mask_mode,
        selknn::mask_logic_en mask_logic) {

    /*
     *
     * Likely this can be improved a lot by using  registers per i_v for R.
     * But requires different compiled functions for each dimension choice up to a reasonable number
     * -> TBI
     *
     */


    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    if(i_v >= end_vert || i_v>=n_vert)
        return;//this will be a problem with actual RS

    if(mask_mode != selknn::mm_none){
        if(mask_logic == selknn::ml_and){
            if(!d_mask[i_v])
                return;
        }
        else{
            if(mask_mode == selknn::mm_scat && d_mask[i_v])
                return;
            else if(mask_mode == selknn::mm_acc && !d_mask[i_v])
                return;
        }
    }

    //protection against n_vert<n_neigh
    size_t nvert_in_row = end_vert - start_vert;
    size_t max_neighbours = n_neigh;
    //set default to self
    if(nvert_in_row<n_neigh){
        max_neighbours=nvert_in_row;
    }


    size_t nfilled=1;
    size_t maxidx_local=0;
    float maxdistsq=0;

    for(size_t j_v=start_vert;j_v<end_vert;j_v++){
        if(i_v == j_v)
            continue;

        if(mask_mode != selknn::mm_none){
            if(mask_logic == selknn::ml_and){
                if(!d_mask[j_v])
                    continue;
            }
            else{
                if(mask_mode == selknn::mm_scat && !d_mask[j_v])
                    continue;
                else if(mask_mode == selknn::mm_acc && d_mask[j_v])
                    continue;
            }
        }
        //fill up
        float distsq = calculateDistance(i_v,j_v,d_coord,d_coord_mod,n_coords, maxdistsq);
        if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
            d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
            d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
            if(distsq > maxdistsq){
                maxdistsq = distsq;
                maxidx_local = nfilled;
            }
            nfilled++;
            continue;
        }
        if(distsq < maxdistsq){// automatically applies to max radius
            //replace former max
            d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
            d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;
            //search new max
            maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        }
    }
    //__syncthreads();

}
}

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SelectModKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
            const float *d_coordmod,
            const int* d_row_splits,
            const int* d_mask,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            selknn::mask_mode_en mask_mode,
            selknn::mask_logic_en mask_logic
            ) {


        //for too low n, d_indices might need to be initialised with some number the
        // rest of the code understands.. maybe -1?

        //just loop over n_rs, in a realistic setting these shouldn't be more than a handful entries

        grid_and_block gb(n_vert,256,n_neigh,4);

        gpu::set_defaults<<<gb.grid(),gb.block()>>>(
                d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);

        cudaDeviceSynchronize();


        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),d_row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost);

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){ //n_rs-1 important!

            int nvert_rs = cpu_rowsplits.at(j_rs+1) - cpu_rowsplits.at(j_rs);
            grid_and_block gb(nvert_rs,1024);

            gpu::select_knn_kernel<<<gb.grid(),gb.block() >>>(
                    d_coord,
                    d_coordmod,
                    d_row_splits,
                    d_mask,

                    d_indices,
                    d_dist,

                    n_vert,
                    n_neigh,
                    n_coords,

                    j_rs,
                    tf_compat,
                    max_radius,
                    mask_mode,
                    mask_logic);

            cudaDeviceSynchronize();

        }
    }

};



template struct SelectModKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
