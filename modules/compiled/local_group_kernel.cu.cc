//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "local_group_kernel.h"
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
struct LocalGroupOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice &d,

            const int *d_neigh_idxs,
            const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op
            //the above will require an argsort on ragged (only with eager workaround so far)

            const float * d_hierarchy_score,

            const float score_threshold,

            const int * d_row_splits,  //keeps dimensions: N_rs x 1

            int * mask,
            int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
            int * d_out_dir_neighbours, // V x K
            int * n_sel_vtx,
            int * d_out_row_splits,

            const int n_in_vert,
            const int n_neigh,
            const int n_row_splits,

            //globals for bookkeeping. dimension n_global_vert_g!
            int *d_out_backgather //which global index each vertex is associated to V x 1
    ){
        //TBI
    }

};



template struct LocalGroupOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

