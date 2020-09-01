//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "pseudo_row_split_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// CPU specialization
template<typename dummy>
struct PseudoRowSplitOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

            const int *d_asso_idx,
            const int *d_n_per_rs,
            const int *row_splits,

            int *asso_idx,
            int *pseudo_rs,

            const int n_vert,
            const int n_rs
    ) {




    }
};

template<typename dummy>
struct PseudoRowSplitCountOpFunctor<GPUDevice, dummy> {
    void operator()(

            const GPUDevice &d,

            const int *d_n_per_rs,

            int& ntotal,
            const int n_per_rs

            ){

        //consider just doing this on CPU
        ntotal=0;
        std::vector<int> cpu_d_n_per_rs(n_per_rs);
        cudaMemcpy(&cpu_d_n_per_rs.at(0), d_n_per_rs, n_per_rs*sizeof(int),cudaMemcpyDeviceToHost);
        //includes a '+1'
        for(int i=0;i<n_per_rs;i++){
            ntotal+=cpu_d_n_per_rs[i];
        }

    }
};


template struct PseudoRowSplitOpFunctor<GPUDevice, int>;
template struct PseudoRowSplitCountOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
