
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "index_replacer_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__global__
static void calc(
        const int32 * to_be_replaced,
        const int32 * replacements,
        int32 * replaced,

        const int n_to_be_replaced,
        const int n_replacements){

    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_to_be_replaced)
        return;

    const int ridx = to_be_replaced[i];
    if(ridx<0){
        replaced[i] = ridx;
        return;
    }
    if(ridx>=n_replacements){
        replaced[i] = to_be_replaced[i]; //security measure but already screwed here
        printf("IndexReplacerOpFunctor: Fatal error: index out of range %d of %d at %d of %d\n", ridx, n_replacements, i, n_to_be_replaced);
        return;
    }
    replaced[i] = replacements[ridx];

}



template<typename dtype>
struct IndexReplacerOpFunctor<GPUDevice, dtype> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,
            const dtype * to_be_replaced,
            const dtype * replacements,
            dtype * replaced,

            const dtype n_to_be_replaced,
            const dtype n_replacements
            ){
            

            grid_and_block gb(n_to_be_replaced,512);

            calc<<<gb.grid(),gb.block()>>>(to_be_replaced,replacements,replaced,n_to_be_replaced,n_replacements);
     
            cudaDeviceSynchronize();
    }
};

template struct IndexReplacerOpFunctor<GPUDevice, int32>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

