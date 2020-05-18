//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "accumulate_knn_nd_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

__device__
float gpu_nd_distanceWeight(float distsq){
  // keep in sync  if(!distsq)return 1;
    return expf(-1.*ACCUMULATE_KNN_EXPONENT* distsq); //uses cuda built in exp
}

__global__
void acc_knn_nd_kernel(const float *d_coord,
        const float *d_feat,
        const int *d_idxs,

        float *d_out_feat,
        int *d_out_maxidxs,

        int n_vert,
        int n_neigh,
        int n_coords,
        int n_feat,

        int n_out_feat,

        int n_moments) {

    /*
     * hints: atomicAdd(number, number)
     *  https://stackoverflow.com/questions/52793423/multi-thread-for-loop-by-cuda
     *
     *  Threads in the same block can communicate with each other via shared memory, barrier synchronization or other synchronization primitives such as atomic operations
     *
     * https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
     *
     *  https://www.diehlpk.de/blog/cuda-7-forall/
     *
     *  https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
     *
     *  grid_stride_range
     */


    //CUDA implementation

    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    const size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_f >= n_feat)
        return;

    const size_t i_c =  blockIdx.z * blockDim.z + threadIdx.z;
    if(i_c >= n_coords)
        return;

    int max_i_n_gidx = 0;
    float t_mean = 0;
    float t_max =  -1e3;//self

    float vic = d_coord[I2D(i_v,i_c,n_coords)]; //buffer?


    for(size_t i_n=0;i_n<n_neigh;i_n++){
        size_t nidx = d_idxs[I2D(i_v,i_n,n_neigh)];

        float vnf = d_feat[I2D(nidx,i_f,n_feat)];
        float vnc = d_coord[I2D(nidx,i_c,n_coords)];
        float distsq = (vic-vnc)*(vic-vnc);

        float wfeat = vnf * gpu_nd_distanceWeight(distsq);
        t_mean += wfeat;
        if(wfeat > t_max){
            max_i_n_gidx = nidx;
            t_max = wfeat;
        }
    }

    t_mean /= (float)n_neigh;

    d_out_maxidxs[I3D(i_v,i_f,i_c,n_feat,n_coords)] = max_i_n_gidx; //just used for gradient
    d_out_feat[I3D(i_v,i_f,i_c,n_out_feat,n_coords)] = t_mean;
    d_out_feat[I3D(i_v,i_f+n_feat,i_c,n_out_feat,n_coords)] = t_max;


    __syncthreads(); //might not be needed

}





typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct AccumulateKnnNdOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,
            int *d_out_maxidxs,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,

            const int n_out_feat,

            const int n_moments) {

       // int gridsize=56;
      //  int blocksize=768;
      //  int numSMs = d.getNumCudaMultiProcessors();

        //just simple 1 thread per vertex


        //for GTX1080, also make some opt for V100
        dim3 numblocks(n_vert/32+1, n_feat/4+1, n_coords/4+1);
        dim3 threadsperblock(32,4,4);//32,4,4
        //just some default optimisation for now
      //  cudaOccupancyMaxPotentialBlockSize(&gridsize,&blocksize,acc_knn_nd_kernel);

     //   std::cout << "opt grid" << gridsize << " opt block " << blocksize << " numSM " << numSMs << std::endl;

        acc_knn_nd_kernel<<<numblocks, threadsperblock>>>(d_coord,d_feat,d_idxs,d_out_feat,d_out_maxidxs,
                n_vert,n_neigh,n_coords,n_feat,n_out_feat,n_moments);

    }

};



template struct AccumulateKnnNdOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
