
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

template<typename dummy>
struct BinByCoordinatesNbinsHelperOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,
            const int * n_bins,
            int * out_tot_bins,
            int n_nbins,
            int nrs){
    int n=1;

    std::vector<int> cpu_n_bins(n_nbins);
    cudaMemcpy(&cpu_n_bins.at(0),n_bins,n_nbins*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<n_nbins;i++)
        n*=cpu_n_bins[i];

    *out_tot_bins=n*(nrs-1);
    }
};

__global__
static void set_defaults(
        int * d_n_per_bin,
        const int n_total_bins
       ){
    for(int i=0;i<n_total_bins;i++)
        d_n_per_bin[i]=0;
}

__global__
static void calc(
        const float * d_coords,
        const int * d_rs,

        const float * d_binswidth, //singleton
        const int * n_bins,//singleton

        int * d_assigned_bin,
        int * d_flat_assigned_bin,
        int * d_n_per_bin,

        const int n_vert,
        const int n_coords,
        const int n_rs,
        const int n_total_bins,
        const bool calc_n_per_bin
){
    int iv=blockIdx.x * blockDim.x + threadIdx.x;
    if(iv>=n_vert)
        return;

    ///same for cu

    int mul = 1;
    int idx = 0;

    for (int ic = n_coords-1; ic != -1; ic--) {

        int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

        if(cidx >= n_bins[ic]){
            printf("Fatal error: index %d of coordinate %d exceeds n bins %d\n",cidx,ic,n_bins[ic]);
            cidx = 0;
        }
        d_assigned_bin[I2D(iv,ic+1,n_coords+1)]=cidx;

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

    if(idx>=n_total_bins){
        printf("global index larger than total bins\n");//DEBUG if you see this you're screwed
        return;
    }

    d_assigned_bin[I2D(iv,0,n_coords+1)]=rsidx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]
    d_flat_assigned_bin[iv]=idx;


    if(calc_n_per_bin){
        //atomic in parallel!
        atomicAdd(&d_n_per_bin[idx] , 1);

    }
    //end same for cu

}




template<typename dummy>
struct BinByCoordinatesOpFunctor<GPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const GPUDevice &d,

            const float * d_coords,
            const int * d_rs,

            const float * d_binswidth, //singleton
            const int * n_bins,//singleton

            int * d_assigned_bin,
            int * d_flat_assigned_bin,
            int * d_n_per_bin,

            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_total_bins,
            const bool calc_n_per_bin
            ){

            //GPU implementation

            grid_and_block gb_def(n_total_bins,1024);

            set_defaults<<<gb_def.grid(),gb_def.block()>>>(d_n_per_bin,n_total_bins);
            cudaDeviceSynchronize();

            grid_and_block gb(n_vert,512);
            
            calc<<<gb.grid(),gb.block()>>>(d_coords, d_rs, d_binswidth,n_bins,

                    d_assigned_bin,
                    d_flat_assigned_bin,
                    d_n_per_bin,

                    n_vert,
                    n_coords,
                    n_rs,
                    n_total_bins,
                    calc_n_per_bin);

    }
};


template struct BinByCoordinatesNbinsHelperOpFunctor<GPUDevice, int>;
template struct BinByCoordinatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow

#endif //GOOGLE_CUDA

