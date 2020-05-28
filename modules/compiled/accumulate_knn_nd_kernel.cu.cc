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
void acc_knn_nd_featsum_kernel(
        const float *d_feat,
        const int *d_idxs,

        float *d_out_feat_sum,

        int n_vert,
        int n_neigh,
        int n_feat) {

    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    const size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_f >= n_feat)
        return;

    float m_featsum = 0;

    for(size_t i_n=0;i_n<n_neigh;i_n++){
        int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
        if(nidx<0) continue; //parallel for all coords and feats.

        float vnf = d_feat[I2D(nidx,i_f,n_feat)];
        m_featsum += vnf;

    }

    d_out_feat_sum[I2D(i_v,i_f,n_feat)] = m_featsum;

}

__global__
void acc_knn_nd_kernel(const float *d_coord,
        const float *d_feat,
        const int *d_idxs,

        float *d_out_feat,
        int *d_out_maxidxs,
        const float *d_out_feat_sum,

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

    float m_mean = 0;

    for(size_t i_n=0;i_n<n_neigh;i_n++){
        int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
        if(nidx<0) continue; //parallel for all coords and feats.

        float vnf = d_feat[I2D(nidx,i_f,n_feat)];
        float vnc = d_coord[I2D(nidx,i_c,n_coords)];
        float dist = (vnc-vic);

        float wfeat = vnf * gpu_nd_distanceWeight(dist*dist);
        t_mean += wfeat;
        if(wfeat > t_max){
            max_i_n_gidx = nidx;
            t_max = wfeat;
        }

        m_mean += vnf * dist;

        __syncthreads();
    }

    t_mean /= (float)n_neigh;

    // __syncthreads(); ?
    d_out_maxidxs[I3D(i_v,i_f,i_c,n_feat,n_coords)] = max_i_n_gidx; //just used for gradient
    d_out_feat[I3D(i_v,i_f,i_c,n_out_feat,n_coords)] = t_mean;
    d_out_feat[I3D(i_v,i_f+n_feat,i_c,n_out_feat,n_coords)] = t_max;

    if(! n_moments)
        return;

    float featsum = d_out_feat_sum[I2D(i_v,i_f,n_feat)];

    if(!featsum) //arrays are zero initialized
        return;

    m_mean /= featsum;

    d_out_feat[I3D(i_v,i_f+2*n_feat,i_c,n_out_feat,n_coords)] = m_mean;

    if(n_moments<2)
        return;
    float m_var = 0;
    float m_skew = 0;

    for(size_t i_n=0;i_n<n_neigh;i_n++){
        int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
        if(nidx<0) continue; //parallel for all coords and feats.

        float vnf = d_feat[I2D(nidx,i_f,n_feat)];
        float vnc = d_coord[I2D(nidx,i_c,n_coords)];
        float dist = (vnc-vic) - m_mean;

        m_var += vnf * dist*dist;
        m_skew += vnf * dist*dist*dist;

        __syncthreads();
    }

    m_var /= featsum;
    m_skew /= featsum;

    d_out_feat[I3D(i_v,i_f+3*n_feat,i_c,n_out_feat,n_coords)] = m_var;
    if(n_moments>2)
        d_out_feat[I3D(i_v,i_f+4*n_feat,i_c,n_out_feat,n_coords)] = m_skew;


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
            float *d_out_feat_sum,

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
        dim3 numblocks_f(n_vert/32+1, n_feat/16+1);
        dim3 threadsperblock_f(32,16);

        acc_knn_nd_featsum_kernel<<<numblocks_f, threadsperblock_f, 0, d.stream()>>>(
                d_feat,
                d_idxs,
                d_out_feat_sum,
                n_vert,
                n_neigh,
                n_feat);

        //for GTX1080, also make some opt for V100
        dim3 numblocks(n_vert/32+1, n_feat/4+1, n_coords/4+1);
        dim3 threadsperblock(32,4,4);//32,4,4
        //just some default optimisation for now
      //  cudaOccupancyMaxPotentialBlockSize(&gridsize,&blocksize,acc_knn_nd_kernel);

     //   std::cout << "opt grid" << gridsize << " opt block " << blocksize << " numSM " << numSMs << std::endl;

        acc_knn_nd_kernel<<<numblocks, threadsperblock, 0, d.stream()>>>(d_coord,d_feat,d_idxs,d_out_feat,d_out_maxidxs,d_out_feat_sum,
                n_vert,n_neigh,n_coords,n_feat,n_out_feat,n_moments);

        cudaDeviceSynchronize();
    }

};



template struct AccumulateKnnNdOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
