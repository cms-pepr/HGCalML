//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "select_threshold_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {



//just call once per rs, just to leave data on gpu

static int fill_empty_and_rs(

        int * n_scatter_idxs,

        int *d_new_rowsplits,
        const int j_rs,

        int old_n_scatter_idx, //needed to build row splits

        int *d_scatter_idxs,
        int startvert,
        int endvert){


    int new_scatter_idx;
    cudaMemcpy(&new_scatter_idx,n_scatter_idxs,sizeof(int),cudaMemcpyDeviceToHost);
    if(new_scatter_idx == old_n_scatter_idx){

        new_scatter_idx += 1;
        cudaMemcpy(n_scatter_idxs,&new_scatter_idx,sizeof(int),cudaMemcpyHostToDevice);

        cudaMemcpy(d_scatter_idxs+new_scatter_idx,&startvert,sizeof(int),cudaMemcpyHostToDevice);

    }
    //row splits
    cudaMemcpy(d_new_rowsplits+j_rs+1,&new_scatter_idx,sizeof(int),cudaMemcpyHostToDevice);

    return new_scatter_idx;
    //can be done on cpu
}


__global__
static void select_kernel(
        int*  idx_lock_mutex,
        const float *d_th_values,
        const int* d_row_splits,

        int *d_scatter_idxs,

        int *d_new_rowsplits,

        int * n_scatter_idxs,

        const int n_vert,

        const int n_rs,
        const int jrs,

        const float threshold){

    lock idx_lock(idx_lock_mutex);

    int startvert=d_row_splits[jrs];
    int endvert=d_row_splits[jrs+1];


    for (auto i_v : grid_stride_range(startvert, endvert)){ //process in blocks

        float th_val = d_th_values[i_v];

        if(th_val >= threshold){

            idx_lock.lock();
            int writeat = *n_scatter_idxs;
            d_scatter_idxs[writeat] = i_v;
            *n_scatter_idxs = writeat + 1;
            idx_lock.unlock();

        }

    }

}

typedef Eigen::GpuDevice GPUDevice;

template<typename dummy>
struct SelectThresholdOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

            const float *d_th_values,
            const int* d_row_splits,

            int *d_scatter_idxs,
            int *d_new_rowsplits,

            int * n_scatter_idxs_cpu, //cpu

            const int n_vert,

            const int n_rs,

            const float threshold) {


        printf("SelectThresholdOpFunctor<GPU> \n");
        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),d_row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost); //Async if needed, but these are just a few kB


        int *n_scatter_idxs=0; //on gpu
        cudaMalloc((void**)&n_scatter_idxs, sizeof(int));

        lock_mutex dlock;


        int oldnsel=0;
        for(size_t jrs=0;jrs<n_rs-1;jrs++){

            cpu_rowsplits.at(jrs);



            int startvert=cpu_rowsplits.at(jrs);
            int endvert=cpu_rowsplits.at(jrs+1);


            int * mutex = dlock.mutex();
            //gpu code
            select_kernel<<<768,1>>>(
                    mutex,
                    d_th_values,
                    d_row_splits,
                    d_scatter_idxs,
                    d_new_rowsplits,
                    n_scatter_idxs,
                    n_vert,
                    n_rs,
                    jrs,
                    threshold);

            cudaDeviceSynchronize();
            //host code
            //continue; //DEBUG
            printf("sel and rs\n");
            oldnsel = fill_empty_and_rs(n_scatter_idxs,d_new_rowsplits,jrs,oldnsel,
                    d_scatter_idxs,startvert,endvert);

        }
        cudaDeviceSynchronize();
        printf("done\n");

        cudaFree(n_scatter_idxs);

        *n_scatter_idxs_cpu = oldnsel;
    }
};

__global__
static void copy_all_kernel(
        int *d_scatter_idxs,

        int *d_tmp_scatter_idxs,

        int  n_scatter_idxs){


    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_scatter_idxs)
        return;
    d_scatter_idxs[i_v] = d_tmp_scatter_idxs[i_v];

}

template<typename dummy>
struct CopyOutputSelectThresholdOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

            int *d_scatter_idxs,

            int *d_tmp_scatter_idxs,

            int  n_scatter_idxs) {

        grid_and_block gb(n_scatter_idxs, 512);

        copy_all_kernel<<<gb.grid(),gb.block(), 0, d.stream()>>>(d_scatter_idxs,d_tmp_scatter_idxs,n_scatter_idxs);


    }
};

template struct SelectThresholdOpFunctor<GPUDevice, int>;
template struct CopyOutputSelectThresholdOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
