//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "compare_knn_outputs_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

namespace gpu{

// Define the CUDA kernel.
__global__ void CompareKnnOpCudaKernel(size_t nvertices,
                                       size_t nneighbours,
                                       const int *input1, 
                                       const int *input2, 
                                       int *output){

    // TODO
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
    //         i += blockDim.x * gridDim.x) {
    //     output1D[i] = scaleFactor * ldg(input1D + i);
    // }
}

}//gpu

typedef Eigen::GpuDevice GPUDevice;

// Define the GPU implementation that launches the CUDA kernel.
template <typename dummy>
struct CompareKnnOpFunctor<GPUDevice, dummy>{
    void operator()(const GPUDevice& d,
                    size_t nvertices,
                    size_t nneighbours,
                    const int *input1, 
                    const int *input2, 
                    int *output){
        // Launch the cuda kernel.
        //
        // See core/util/gpu_kernel_helper.h for example of computing
        // block count and thread_per_block count.
        int block_count = 1024;
        int thread_per_block = 20;

        gpu::CompareKnnOpCudaKernel<<<block_count, thread_per_block, 0, d.stream()>>>(nvertices,nneighbours,input1,input2,output);
    }
};



template struct CompareKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
