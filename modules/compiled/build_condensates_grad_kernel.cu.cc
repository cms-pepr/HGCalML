//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "build_condensates_grad_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

namespace gpu{
//kernels
}

typedef Eigen::GpuDevice GPUDevice;

__global__
void kernel(
        const float *sum_features_grad,
        const int *asso_idx,
        float *features_grad,

        const int n_vert,
        const int n_feat){


    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;

    int asso = asso_idx[i_v];

    if(asso<0)
        features_grad[I2D(i_v,i_f,n_feat)] = 0;
    else
        features_grad[I2D(i_v,i_f,n_feat)] = sum_features_grad[I2D(asso,i_f,n_feat)];

}

template <typename dummy>
struct BuildCondensatesGradOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice& d,

            const float *sum_features_grad,
            const int *asso_idx,
            float *features_grad,

            const int n_vert,
            const int n_feat

    ) {

        grid_and_block gb_def_f(n_vert, 64, n_feat, 8);
        kernel<<<gb_def_f.grid(), gb_def_f.block(), 0, d.stream()>>>(sum_features_grad,asso_idx,features_grad,n_vert,n_feat);

        cudaDeviceSynchronize();

    }

};



template struct BuildCondensatesGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
