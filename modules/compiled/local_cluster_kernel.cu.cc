//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "local_cluster_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

///__global__
///void local_cluster_kernel(/*TBI*/) {
/////parallelise over neighbours and row splits
///    size_t i_n =  blockIdx.x * blockDim.x + threadIdx.x;
///    size_t i_rs =  blockIdx.y * blockDim.y + threadIdx.y;
///
///}





typedef Eigen::GpuDevice GPUDevice;


template<typename dummy>
struct LocalClusterOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice &d,

            const int *d_neigh_idxs,
            const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op
            //the above will require an argsort on ragged (only with eager workaround so far)

            const int * d_global_idxs, //global index of each vertex: V x 1, not global dimension!
            const int * d_row_splits,  //keeps dimensions: N_rs x 1

            int * mask,
            int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
            int * n_sel_vtx,
            int * d_out_row_splits,

            const int n_in_vert,
            const int n_neigh,
            const int n_row_splits,


            //globals for bookkeeping. dimension n_global_vert_g!
            int *d_out_backscatter, //which global index each vertex is associated to V x 1
            int n_global_vert_g
    ){
        //TBI
    }

};



template struct LocalClusterOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

