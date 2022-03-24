
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
        const int * to_be_replaced,
        const int * replacements,
        int * replaced,

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
        printf("IndexReplacerOpFunctor: index out of range\n");
        return;
    }
    replaced[i] = replacements[ridx];

}



template<typename dummy>
struct IndexReplacerOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,
            const int * to_be_replaced,
            const int * replacements,
            int * replaced,

            const int n_to_be_replaced,
            const int n_replacements
            ){
            

            grid_and_block gb(n_to_be_replaced,1024);

            calc<<<gb.grid(),gb.block()>>>(to_be_replaced,replacements,replaced,n_to_be_replaced,n_replacements);
    }
};

template struct IndexReplacerOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

