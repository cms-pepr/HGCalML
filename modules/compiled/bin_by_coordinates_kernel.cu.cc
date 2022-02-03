
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "helpers.h"
#include "bin_by_coordinates_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__global__
static void calc(
        const float * d_coords,
        const int * d_rs,

        const float * d_binswidth, //singleton
        const int* n_bins,//singleton

        int * d_assigned_bin,

        int n_vert,
        int n_coords,
        int n_rs){

    int iv=blockIdx.x * blockDim.x + threadIdx.x;
    if(iv>=n_vert)
        return;

    int mul = 1;
    int idx = 0;
    for (int ic = n_coords-1; ic != -1; ic--) {

        int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

        idx += cidx * mul;
        mul *= n_bins[ic];

    }

    //get row split index last
    int rsidx=0;
    for(int irs=1 ; irs < n_rs ; irs++){
        if(d_rs[irs] > iv){
            break;
        }
        rsidx++;
    }

    idx += rsidx * mul;

    d_assigned_bin[iv]=idx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]
}


template<typename dummy>
struct BinByCoordinatesOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,

            const float * d_coords,
            const int * d_rs,

            const float * d_binswidth, //singleton
            const int* n_bins,//singleton

            int * d_assigned_bin,

            int n_vert,
            int n_coords,
            int n_rs
            ){

            //GPU implementation
            grid_and_block gb(n_vert,512);
            
            calc<<<gb.grid(),gb.block()>>>(
                    d_coords,
                    d_rs,
                    d_binswidth,
                    n_bins,
                    d_assigned_bin,
                    n_vert,
                    n_coords,
                    n_rs
            );
    }
};

template struct BinByCoordinatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

