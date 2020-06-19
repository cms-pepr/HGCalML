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



    }

};



template struct BuildCondensatesGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
