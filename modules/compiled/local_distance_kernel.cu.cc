//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "local_distance_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {


typedef Eigen::GpuDevice GPUDevice;

namespace gpu{


__device__
static float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}

__global__
static void set_defaults(
        float *d_dist,
        const int n_vert,
        const int n_neigh
){
    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t n= blockIdx.y * blockDim.y + threadIdx.y;
    if(n >= n_neigh)
        return;

    d_dist[I2D(i_v,n,n_neigh)] = 0;
}

__global__
static void calc_distances(const int *d_neigh_idxs,
        const float *d_coords,

        float * d_distances,
        const int n_coords,
        const int n_in_vert,
        const int n_out_vert,
        const int n_neigh
){
    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_out_vert)
        return;

    size_t j_n= blockIdx.y * blockDim.y + threadIdx.y;
    if(j_n >= n_neigh)
        return;

    int j_v = d_neigh_idxs[I2D(i_v,j_n,n_neigh)];
    if(j_v < 0)
        return;

    float distsq = calculateDistance(i_v,j_v,d_coords,n_coords);
    d_distances[I2D(i_v, j_n, n_neigh)] = distsq;

}

}

template<typename dummy>
struct LocalDistanceOpFunctor<GPUDevice,dummy> {
    void operator()(
            const GPUDevice &d,

            const int *d_neigh_idxs,
            const float *d_coords,

            float * d_distances,

            const int n_coords,
            const int n_in_vert,
            const int n_out_vert,
            const int n_neigh
    ){

        grid_and_block gb(n_out_vert,256,n_neigh,4);

        gpu::set_defaults<<<gb.grid(),gb.block()>>>(d_distances,n_out_vert,n_neigh);

        cudaDeviceSynchronize();

        gpu::calc_distances<<<gb.grid(),gb.block()>>>(
                d_neigh_idxs,
                d_coords,

                d_distances,
                n_coords,
                n_in_vert,
                n_out_vert,
                n_neigh
        );

        cudaDeviceSynchronize();

    }



};





template struct LocalDistanceOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

