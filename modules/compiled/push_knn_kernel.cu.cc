//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "push_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

__global__
static void set_zero(
        float * tensor,
        size_t n_vert,
        size_t n_feat
){

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;

    tensor[I2D(i_v, i_f, n_feat)] = 0;

}

__global__
void push_knn_kernel(
        const float *d_weights,
        const float *d_feat,
        const int *d_idxs,

        float *d_out_feat,

        int n_vert,
        int n_neigh,
        int n_feat) {

    //parallelise over neighbours and features - no race conditions

    size_t i_f =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_v =  blockIdx.y * blockDim.y + threadIdx.y;
    size_t i_n =  blockIdx.z * blockDim.z + threadIdx.z;
    if(i_n >= n_neigh || i_f >= n_feat || i_v >= n_vert)
        return;

    int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
    if(nidx<0) return;


    float f = d_feat[I2D(i_v,i_f,n_feat)];
    float w = d_weights[I2D(i_v,i_n,n_neigh)];

    atomicAdd(&d_out_feat[I2D(nidx,i_f,n_feat)] , f*w);

}


typedef Eigen::GpuDevice GPUDevice;

template <typename dummy>
struct PushKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_weights,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,

            int n_vert,
            int n_neigh,
            int n_feat) {

        //parallelise over neighbours and features; would need to be atomic in nvert

        grid_and_block par0(n_vert, 256,n_feat, 2);

        set_zero<<<par0.grid(), par0.block(), 0, d.stream()>>>(d_out_feat, n_vert, n_feat);

        cudaDeviceSynchronize();

        //this should keep the atomic reasonably ok
        grid_and_block par(
                        n_feat, 32,
                        n_vert, 2,
                        n_neigh, 8);//explicit one in n_vert to avoid race conditions

        if(n_neigh <= 6){//this is kind of a standard setting so worth covering
            if(n_feat >= 128){ //32 and 64 are also rather standard
                par = grid_and_block(n_feat, 128,
                                     n_vert, 1,
                                     n_neigh, 1);//no atomic *within* one block, still can be globally!
            }
            else if(n_feat >= 64){ //32 and 64 are also rather standard
                par = grid_and_block(
                        n_feat, 64,
                        n_vert, 2,
                        n_neigh, 2);
            }
            else if(n_feat >= 32){ //32 and 64 are also rather standard
                par = grid_and_block(
                        n_feat, 32,
                        n_vert, 4,
                        n_neigh, 2);
            }
            if(n_feat < 2){ //this is for energy push, also standard
                par = grid_and_block(n_feat, 1,
                                     n_vert, 128,
                                     n_neigh, 2);
            }
            if(n_feat <32){ //this is for energy push, also standard
                par = grid_and_block(n_feat, 8,
                                     n_vert, 12,
                                     n_neigh, 2);
            }
        }

        push_knn_kernel<<<par.grid(), par.block(), 0, d.stream()>>>(
                d_weights,
                d_feat,
                d_idxs,

                d_out_feat,

                n_vert,
                n_neigh,
                n_feat);

        cudaDeviceSynchronize();
    }
};



template struct PushKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

/*
     * hints: atomicAdd(number, number)
     *  https://stackoverflow.com/questions/52793423/multi-thread-for-loop-by-cuda
     *
     *  Threads in the same block can communicate with each other via shared memory, barrier synchronization or other synchronization primitives such as atomic operations
     *
     * https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
     *
     *  https://www.diehlpk.de/blog/cuda-7-forall/
     *
     *  https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
     *
     *  grid_stride_range
     */


    //CUDA implementation
