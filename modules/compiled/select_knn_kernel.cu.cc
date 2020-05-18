//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "select_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {





typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SelectKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
            int *d_indices,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {


    }

};



template struct SelectKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
