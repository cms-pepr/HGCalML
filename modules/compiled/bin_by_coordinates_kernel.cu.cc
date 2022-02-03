
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "bin_by_coordinates_kernel.h"


namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


__global__
static void calc(const float *input){

}


template<typename dummy>
struct BinByCoordinatesOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d
            ){
            //GPU implementation
            int N=10;//needs to be passed
            grid_and_block gb(N,768);
            
            calc<<<gb.grid(),gb.block()>>>(NULL);
    }
};

template struct BinByCoordinatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

