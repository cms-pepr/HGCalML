//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "oc_helper_m_indices_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


__global__
static void calc(
        const int *d_truthidx,
        const int *d_unique_idx,

        int * out_idx,
        float * m_not,

        const int n_vert,
        const int n_unique,
        const int n_max_per_unique,
        bool calc_m_not){

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k>=n_unique)
        return;


    //

    int uqidx = d_unique_idx[k];
    int puqcounter=0;
    for(int i_v = 0; i_v < n_vert; i_v++ ){
        if(uqidx>=0 && d_truthidx[i_v] == uqidx){
            out_idx[I2D(k, puqcounter, n_max_per_unique)] = i_v;
            puqcounter++;
        }
    }
    for(int prem = puqcounter; prem < n_max_per_unique; prem++){
        out_idx[I2D(k, prem, n_max_per_unique)] = -1;
    }

    //
    if(calc_m_not ){
        //m_not
        for(int i_v = 0; i_v < n_vert; i_v++ ){
            if(uqidx>=0 && d_truthidx[i_v] == uqidx)
                m_not [I2D(k, i_v, n_vert)] = 0.;
            else
                m_not [I2D(k, i_v, n_vert)] = 1.;
        }
    }


}


template<typename dummy>
struct MIndicesMaxUqOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(

            const GPUDevice &d,
            const int *d_maxunique,
            int * n_max_per_unique
            ){

        cudaMemcpy(n_max_per_unique, d_maxunique, sizeof(int), cudaMemcpyDeviceToHost);
    }
};

template<typename dummy>
struct MIndicesOpFunctor<GPUDevice, dummy> {
    void operator()(

            const GPUDevice &d,

            const int *d_truthidx,
            const int *d_unique_idx,

            int * out_idx,
            float * m_not,

            const int n_vert,
            const int n_unique,
            const int n_max_per_unique,

            bool calc_m_not

            ){

        grid_and_block gb(n_unique,768);
        calc<<<gb.grid(),gb.block()>>>(
                d_truthidx,
                d_unique_idx,

                out_idx,
                m_not,

                n_vert,
                n_unique,
                n_max_per_unique,

                calc_m_not
        );

    }
};



template struct MIndicesMaxUqOpFunctor<GPUDevice, int>;
template struct MIndicesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
