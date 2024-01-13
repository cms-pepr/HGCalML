
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "rs_offset_adder_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__global__
static void calc(
        const int * t_dx,
            const int * rs,
            int * new_t_idx,

            const int n_vert,
            const int n_rs){

    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_vert)
        return;
    int offset = 0;
    for (int j = 0; j < n_rs; j++){
        if (i >= rs[j]){
            offset = rs[j];
        }
        else{
            break;
        }
    }
    int current =  t_dx[i];
    if (t_dx[i] >= 0)
        new_t_idx[i] = current + offset;
    else
        new_t_idx[i] = current;
}



template<typename dtype>
struct RSOffsetAdderOpFunctor<GPUDevice, dtype> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,
            const int * t_dx,
            const int * rs,
            int * new_t_idx,

            const int n_vert,
            const int n_rs
            ){
            

            grid_and_block gb(n_vert,512);

            calc<<<gb.grid(),gb.block()>>>(
                t_dx,
                rs,
                new_t_idx,
                n_vert,
                n_rs
            );
            cudaDeviceSynchronize();
    }
};

template struct RSOffsetAdderOpFunctor<GPUDevice, int32>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

