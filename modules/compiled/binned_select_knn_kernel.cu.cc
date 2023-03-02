
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "helpers.h"
#include "binned_select_knn_kernel.h"
#include "binstepper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__device__
float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


__device__
int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

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
void set_defaults(
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



template<int N_bin_dims>
__global__
static void select_knn_kernel(

        const float * d_coord,
        const int * d_bin_idx,
        const int * d_direction,
        const int * d_dim_bin_idx,

        const int * d_bin_boundaries,
        const int * d_n_bins,

        const float* d_bin_width,

        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_bin_dim,

        const int n_bboundaries,
        bool use_direction) {

    //bin boundaries [i] [i+1] describe the scan ranges


    //really no buffering at all here


    int i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>=n_vert)
        return;//safe guard

    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
    if(use_direction &&
            (d_direction[i_v] == 0 || d_direction[i_v] == 2))
        return;

    //continue;//do nothing


    size_t nfilled=1;//self-reference from defaults
    size_t maxidx_local=0;
    float maxdistsq=0;

    int total_subbins = 1;
    for(int sbi=0;sbi<n_bin_dim;sbi++)
        total_subbins *= d_n_bins[sbi];

    int iv_bin = d_bin_idx[i_v];
    int gbin_offset = total_subbins*(iv_bin / total_subbins);
    int sb_flat_offset = iv_bin - gbin_offset;

    // printf("considering vertex %d, bin %d, flat offset %d, global bin offset %d\n",i_v,iv_bin,sb_flat_offset,gbin_offset);


    binstepper<N_bin_dims> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);

    bool continue_search = true;
    int distance = 0;
    while(continue_search){

        stepper.set_d(distance);

        continue_search=false;

        while(true){
            int idx = stepper.step();
            if(idx<0){//not valid
                if(!continue_search && !distance){//this should not happen
                    printf("stopping search for vtx %d at distance %d\n",i_v,distance);
                }
                break;

            }

            idx+=gbin_offset;

            if(idx>=n_bboundaries-1){
                printf("idx %d out of range, gb offset %d, distance %d, sb_flat_offset %d, nbb %d\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);
                continue;
            }

            int start_vertex = d_bin_boundaries[idx];
            int end_vertex = d_bin_boundaries[idx+1];

            for(size_t j_v=start_vertex;j_v<end_vertex;j_v++){
                if(i_v == j_v)
                    continue;

                // 0: can only be neighbour, 1: can only have neighbour, 2: neither
                if(use_direction &&
                        (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                    continue;

                //fill up
                float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                if(nfilled< n_neigh){
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

            continue_search=true;//at least one was valid

        }
        // debug: never stop unless all bins exhausted DEBUG FIXME
        if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
            break;//done

        distance++;
    }

}

//specify  dimensions
template __global__ void select_knn_kernel<2>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries,bool use_direction);
template __global__ void select_knn_kernel<3>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries,bool use_direction);
template __global__ void select_knn_kernel<4>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries,bool use_direction);
template __global__ void select_knn_kernel<5>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries,bool use_direction);

template<typename dummy>
struct BinnedSelectKnnOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,

            const float * d_coord,
            const int * d_bin_idx,
            const int * d_direction,
            const int * d_dim_bin_idx,

            const int * d_bin_boundaries,
            const int * d_n_bins,

            const float* d_bin_width,

            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_bin_dim,

            const int n_bboundaries,
            bool tf_compat,
            bool use_direction
            ){
        //GPU implementation
        grid_and_block gbdef(n_vert,256,n_neigh,4);
        set_defaults<<<gbdef.grid(),gbdef.block()>>>(d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);

        //really no buffering at all here

        if(n_bin_dim<2 || n_bin_dim>5){//just a sfe guard, also checked earlier
            throw std::out_of_range("BinnedSelectKnnOpFunctor: GPU implementation is restricted to 2-8 dimensions (inclusive).");
        }

        grid_and_block gb(n_vert,512);
        if(n_bin_dim==2)
            select_knn_kernel<2><<<gb.grid(),gb.block()>>>(d_coord, d_bin_idx,d_direction,d_dim_bin_idx,d_bin_boundaries,d_n_bins,d_bin_width,
                    d_indices,d_dist,n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);
        if(n_bin_dim==3)
            select_knn_kernel<3><<<gb.grid(),gb.block()>>>(d_coord, d_bin_idx,d_direction,d_dim_bin_idx,d_bin_boundaries,d_n_bins,d_bin_width,
                    d_indices,d_dist,n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);
        if(n_bin_dim==4)
            select_knn_kernel<4><<<gb.grid(),gb.block()>>>(d_coord, d_bin_idx,d_direction,d_dim_bin_idx,d_bin_boundaries,d_n_bins,d_bin_width,
                    d_indices,d_dist,n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);
        if(n_bin_dim==5)
            select_knn_kernel<5><<<gb.grid(),gb.block()>>>(d_coord, d_bin_idx,d_direction,d_dim_bin_idx,d_bin_boundaries,d_n_bins,d_bin_width,
                    d_indices,d_dist,n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);

    }
};


template struct BinnedSelectKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

