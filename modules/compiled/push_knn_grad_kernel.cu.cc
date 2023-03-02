//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "push_knn_grad_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {


__global__
void push_knn_grad_weight_kernel(
        const float *d_grad,

        const float *d_weights,
        const float *d_feat,
        const int *d_idxs,

        float *d_feat_grad,
        float *d_w_grad,

        int n_vert,
        int n_neigh,
        int n_feat) {


    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_n = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_n >= n_neigh)
        return;

    int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
    if(nidx < 0){
        d_w_grad[I2D(i_v,i_n,n_neigh)] = 0;
        return;
    }

    float wgrad = 0;

    for (size_t i_f = 0; i_f < n_feat; i_f++) {

        float f = d_feat[I2D(i_v, i_f, n_feat)];
        wgrad += d_grad[I2D(nidx, i_f, n_feat)] * f;

    }
    d_w_grad[I2D(i_v,i_n,n_neigh)] = wgrad;

}

__global__
void push_knn_grad_feat_kernel(
        const float *d_grad,

        const float *d_weights,
        const float *d_feat,
        const int *d_idxs,

        float *d_feat_grad,
        float *d_w_grad,

        int n_vert,
        int n_neigh,
        int n_feat) {

    //feature gradient

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;

    float fgrad = 0;

    for (size_t i_n = 0; i_n < n_neigh; i_n++){

        int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
        if(nidx < 0)
            continue;

        float w = d_weights[I2D(i_v, i_n, n_neigh)];

        fgrad += d_grad[I2D(nidx, i_f, n_feat)] * w;

    }
    d_feat_grad[I2D(i_v, i_f, n_feat)] = fgrad;
}


typedef Eigen::GpuDevice GPUDevice;

template <typename dummy>
struct PushKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,
            const float *d_grad,

            const float *d_weights,
            const float *d_feat,
            const int *d_idxs,

            float *d_feat_grad,
            float *d_w_grad,

            int n_vert,
            int n_neigh,
            int n_feat) {

        grid_and_block par1(n_vert, 64, n_feat, 8);
        if(n_feat<8)
            par1 = grid_and_block(n_vert, 256, n_feat, 2);

        push_knn_grad_feat_kernel<<<par1.grid(), par1.block(), 0, d.stream()>>>(
                d_grad,d_weights,d_feat,d_idxs,d_feat_grad,d_w_grad,n_vert,n_neigh,n_feat);

        cudaDeviceSynchronize();

        grid_and_block par2(n_vert, 64, n_neigh, 8);
        push_knn_grad_weight_kernel<<<par2.grid(), par2.block(), 0, d.stream()>>>(
                d_grad,d_weights,d_feat,d_idxs,d_feat_grad,d_w_grad,n_vert,n_neigh,n_feat);

        cudaDeviceSynchronize();

    }

};



template struct PushKnnGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA


    //CUDA implementation
