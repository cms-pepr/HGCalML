//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "accumulate_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

__device__
float gpu_distanceWeight(float distsq){
  // keep in sync  if(!distsq)return 1;
    return expf(-1.*ACCUMULATE_KNN_EXPONENT* distsq); //uses cuda built in exp
}

__global__
void acc_knn_kernel(const float *d_coord,
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

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;


    size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_f >= n_feat)
        return;
    //fully independent per i_v, no races

    //scales mildly up to about 10k vertices, and then linearly

//    for(const auto i_v: grid_stride_range(0,n_vert)){
    //  for (size_t i_v = 0; i_v < n_vert; i_v++) {

  //      for(size_t i_f=0;i_f<n_feat;i_f++){
//          for(const auto i_f: grid_stride_range_y(0,n_feat)){
            float t_mean = 0;
            float t_max = 0;
            int max_i_n_gidx = 0;

            for(size_t i_n=0;i_n<n_neigh;i_n++){
                size_t nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
                float vnf = d_feat[I2D(nidx,i_f,n_feat)];
                float distsq = 0;

                for(size_t i_c=0;i_c<n_coords;i_c++){//this is about a factor of 2 w.r.t. TF
                    float vic = d_coord[I2D(i_v,i_c,n_coords)];
                    float vnc = d_coord[I2D(nidx,i_c,n_coords)];
                    distsq += (vic-vnc)*(vic-vnc);
                }
                float wfeat = vnf * gpu_distanceWeight(distsq);
                t_mean += wfeat;
                if(wfeat > t_max){
                    max_i_n_gidx = nidx;
                    t_max = wfeat;
                }
            }
            t_mean /= (float)n_neigh;

            d_out_maxidxs[I2D(i_v,i_f,n_feat)] = max_i_n_gidx; //just used for gradient
            d_out_feat[I2D(i_v,i_f,n_out_feat)] = t_mean;
            d_out_feat[I2D(i_v,i_f+n_feat,n_out_feat)] = t_max;

            //moments in n_coords x n_neigh loop here {}

   //     }

 //  }

  //  __syncthreads(); //might not be needed

}





typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct AccumulateKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
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

       // int gridsize=56;
      //  int blocksize=768;
      //  int numSMs = d.getNumCudaMultiProcessors();

        //just simple 1 thread per vertex

        //for GTX1080, also make some opt for V100
        dim3 grid(n_vert/64+1,n_feat/12+1);
        dim3 block(64,12);
        //just some default optimisation for now
      //  cudaOccupancyMaxPotentialBlockSize(&gridsize,&blocksize,acc_knn_kernel);

     //   std::cout << "opt grid" << gridsize << " opt block " << blocksize << " numSM " << numSMs << std::endl;

        acc_knn_kernel<<<grid, block, 0, d.stream()>>>(d_coord,d_feat,d_idxs,d_out_feat,d_out_maxidxs,
                n_vert,n_neigh,n_coords,n_feat,n_out_feat,n_moments);
        cudaDeviceSynchronize();
    }

};



template struct AccumulateKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
