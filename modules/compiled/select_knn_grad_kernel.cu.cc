//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "select_knn_grad_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

namespace gpu{

/*
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
*/

__global__
static void select_knn_grad_selfloop_kernel(
        const float *d_grad_dist, // V x N
        const int *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const int n_vert,
        const int n_neigh,
        const int n_coords) {

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    float self_contrib=0;
    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int k = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(k<0 || k>= n_vert){
            if( k>= n_vert)
                printf("select_knn_grad_kernel: k out of range\n");
            continue;
        }

        const float gik = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        const float xknu = d_coord[I2D(k,nu_c,n_coords)];


        self_contrib -= 2. * gik * (xknu - xinu);

    }
    d_grad_coord[I2D(i_v,nu_c,n_coords)] = self_contrib;
}

__global__
static void select_knn_grad_neighloop_kernel(
        const float *d_grad_dist, // V x N
        const int *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const int n_vert,
        const int n_neigh,
        const int n_coords){


    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m<0 || m>= n_vert){
            if(m>= n_vert)
                printf("select_knn_grad_kernel: m out of range\n");
            continue;
        }

        const float gim = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        const float xmnu = d_coord[I2D(m,nu_c,n_coords)];

        float add = 2. * gim * (xmnu - xinu);
        atomicAdd( &d_grad_coord[I2D(m, nu_c, n_coords)], add);

    }
}


}//gpu

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SelectKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,

            float * d_grad_coord,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {


        //for too low n, d_indices might need to be initialised with some number the
        // rest of the code understands.. maybe -1?

        //just loop over n_rs, in a realistic setting these shouldn't be more than a handful entries

        grid_and_block gb(n_vert,256,n_coords,4);


        gpu::select_knn_grad_selfloop_kernel<<<gb.grid(),gb.block(), 0, d.stream()>>>(
                d_grad_dist,
                d_indices,
                d_dist,
                d_coord,
                d_grad_coord,
                n_vert,
                n_neigh,
                n_coords
        );

        cudaDeviceSynchronize();

        gpu::select_knn_grad_neighloop_kernel<<<gb.grid(),gb.block(), 0, d.stream()>>>(
                d_grad_dist,
                d_indices,
                d_dist,
                d_coord,
                d_grad_coord,
                n_vert,
                n_neigh,
                n_coords
        );

        cudaDeviceSynchronize();

    }

};



template struct SelectKnnGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
