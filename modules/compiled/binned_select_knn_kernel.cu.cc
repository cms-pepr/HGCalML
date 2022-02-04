
#if GOOGLE_CUDASS
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

__global__
static void select_knn_kernel(

        const float * d_coord,
        const int * d_bin_idx,

        const int * d_bin_boundaries,
        const int * d_n_bins,

        const float* d_bin_width,

        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int n_bboundaries) {

    //bin boundaries [i] [i+1] describe the scan ranges


    //really no buffering at all here

    int i_v =  blockIdx.x * blockDim.x + threadIdx.x;

    if(i_v>=n_vert)
        return;//safe guard

    size_t nfilled=1;//self-reference from defaults
    size_t maxidx_local=0;
    float maxdistsq=0;

    int total_subbins = 1;
    for(int sbi=0;sbi<n_coords;sbi++)
        total_subbins *= d_n_bins[sbi];

    int iv_bin = d_bin_idx[i_v];
    int gbin_offset = total_subbins*(iv_bin / total_subbins);
    int sb_flat_offset = iv_bin - gbin_offset;

    // printf("considering vertex %d, bin %d, flat offset %d, global bin offset %d\n",i_v,iv_bin,sb_flat_offset,gbin_offset);

    binstepper_base * bs = NULL;
    if(n_coords==2)
        bs=new binstepper(2)({d_n_bins[0],d_n_bins[1]},sb_flat_offset);
    else if(n_coords==3)
        bs=new binstepper(3)({d_n_bins[0],d_n_bins[1],d_n_bins[2]},sb_flat_offset);
    else if(n_coords==4)
        bs=new binstepper(4)({d_n_bins[0],d_n_bins[1],d_n_bins[2],d_n_bins[3]},sb_flat_offset);
    else
        bs=NULL;//not supported

    bool continue_search = true;
    int distance = 0;
    while(continue_search){
        int b_idx = 0;
        bs->set_distance(distance);
        continue_search=false;

        while(true){
            bool valid=false;
            int idx = bs->step(valid);
            if(!valid){
                break;
            }

            idx+=gbin_offset;

            if(idx>=n_bboundaries-1){

                printf("idx %d out of range, gb offset %d, distance %d, sb_flat_offset %d, nbb %d\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);

                break;
            }
            int start_vertex = d_bin_boundaries[idx];
            int end_vertex = d_bin_boundaries[idx+1];


            for(size_t j_v=start_vertex;j_v<end_vertex;j_v++){
                if(i_v == j_v)
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

        if(maxdistsq && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
            break;//done

        distance++;
    }

    delete bs;

}


template<typename dummy>
struct BinnedSelectKnnOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,
            const float * d_coord,
            const int * d_bin_idx,

            const int * d_bin_boundaries,
            const int * d_n_bins,

            const float* d_bin_width,

            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_bboundaries,
            bool tf_compat
            ){
        //GPU implementation
        int N=10;//needs to be passed

        grid_and_block gbdef(n_vert,256,n_neigh,4);
        set_defaults<<<gbdef.grid(),gbdef.block()>>>(d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);
        //really no buffering at all here

        grid_and_block gb(n_vert,512);
        select_knn_kernel<<<gb.grid(),gb.block()>>>(
                d_coord,
                d_bin_idx,

                d_bin_boundaries,
                d_n_bins,

                d_bin_width,

                d_indices,
                d_dist,

                n_vert,
                n_neigh,
                n_coords,

                n_bboundaries);
    }
};

template struct BinnedSelectKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

