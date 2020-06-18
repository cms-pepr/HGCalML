//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "build_condensates_kernel.h"
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
struct BuildCondensatesOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice& d,

            const float *d_ccoords,
            const float *d_betas,
            const int *beta_sorting,
            const float *features,
            const int *row_splits,


            float *summed_features,
            int *asso_idx,

            const int n_vert,
            const int n_feat,
            const int n_ccoords,

            const int n_rs,

            const float radius,
            const float min_beta

) {



        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){ //n_rs-1 important!


            //do stuff

            cudaDeviceSynchronize();

        }
    }

};



template struct BuildCondensatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
