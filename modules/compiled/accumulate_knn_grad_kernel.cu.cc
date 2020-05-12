//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "accumulate_knn_grad_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct AccumulateKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_from_out_features,
            const float *d_coord,
            const float *d_feat, // sum(V) x F
            const int *d_max_feat_indices,
            const int * d_neigh_indices,

            float *d_out_grad_coords,
            float *d_out_grad_features,

            int n_vert,
            int n_neigh,
            int n_coords,
            int n_feat,

            int n_grad_from_out_feat,

            int n_moments) {


        //CUDA implementation


  }
};



template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;

}
}


#endif  // GOOGLE_CUDA

