//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "assign_to_condensates_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template<typename dummy>
struct AssignToCondensatesOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

            const float *d_ccoords,
            const float *d_dist,
            const int *c_point_idx,

            const int *row_splits,

            int *asso_idx,

            const float radius,
            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_condensates
    ) {

        throw std::runtime_error("AssignToCondensatesOpFunctor: no GPU implementation.");

    }
};




template struct AssignToCondensatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
