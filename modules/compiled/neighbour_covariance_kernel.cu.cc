//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "neighbour_covariance_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

__global__
static void calcmeans( // <<< vout,nfeat,(ncoords)
        const float *d_coords,
        const float *d_feats,
        const int* d_n_dixs,

        float * d_covariance, //just for same interface Vout x F x C
        float * d_means,

        const int n_vert,
        const int n_coords,
        const int n_feat,
        const int n_neigh,
        const int n_vert_out){

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert_out)
        return;

    size_t i_f= blockIdx.y * blockDim.y + threadIdx.y;
    if(i_f >= n_feat)
        return;

    float sumw=0;
    for(int i_n = 0; i_n < n_neigh; i_n++){
        int nidx = d_n_dixs[I2D(i_v,i_n,n_neigh)];
        if(nidx<0)
            continue;
        float feat = d_feats[I2D(nidx,i_f,n_feat)];
        sumw += feat;
    }
    for (int i_c = 0; i_c < n_coords; i_c++) {

        float sum=0;
        for(int i_n = 0; i_n < n_neigh; i_n++){
            int nidx = d_n_dixs[I2D(i_v,i_n,n_neigh)];
            if(nidx<0)
                continue;
            float feat = d_feats[I2D(nidx,i_f,n_feat)];
            float coord = d_coords[I2D(nidx,i_c,n_coords)];
            sum += feat * coord;
        }

        float entry = sum/(sumw+1e-4);
        if(!sumw)
            entry=0;
        d_means[I3D(i_v,i_f,i_c,n_feat,n_coords)]=entry;
    }

}

__global__
static void calccov( // <<< vout,nfeat,(ncoords)
        const float *d_coords,
        const float *d_feats,
        const int* d_n_dixs,

        float * d_covariance, //just for same interface Vout x F x C
        float * d_means,

        const int n_vert,
        const int n_coords,
        const int n_feat,
        const int n_neigh,
        const int n_vert_out){

    int n_covs = n_coords*(n_coords+1)/2;

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert_out)
        return;

    size_t i_f= blockIdx.y * blockDim.y + threadIdx.y;
    if(i_f >= n_feat)
        return;

    float sumw=0;
    for(int i_n = 0; i_n < n_neigh; i_n++){
        int nidx = d_n_dixs[I2D(i_v,i_n,n_neigh)];
        if(nidx<0)
            continue;
        float feat = d_feats[I2D(nidx,i_f,n_feat)];
        sumw += feat;
    }
    for (int i_c = 0; i_c < n_coords; i_c++) {
        for (int j_c = 0; j_c <= i_c; j_c++) {


            float sum=0;
            float meancoordi = d_means[I3D(i_v,i_f,i_c,n_feat,n_coords)];
            float meancoordj = d_means[I3D(i_v,i_f,j_c,n_feat,n_coords)];
            for(int i_n = 0; i_n < n_neigh; i_n++){
                int nidx = d_n_dixs[I2D(i_v,i_n,n_neigh)];
                if(nidx<0)
                    continue;
                float feat = d_feats[I2D(nidx,i_f,n_feat)];
                float coordi = d_coords[I2D(nidx,i_c,n_coords)];
                float coordj = d_coords[I2D(nidx,j_c,n_coords)];
                sum += feat * (coordi - meancoordi)*(coordj - meancoordj);
            }
            //j<=i
            int covidx = j_c + (i_c+1)*i_c/2 ;
            float entry = sum/(sumw+1e-4);
            if(!sumw)
                entry=0;
            d_covariance[I3D(i_v,i_f,covidx,n_feat,n_covs)]=entry;
        }
    }

}

//just call once per rs, just to leave data on gpu


typedef Eigen::GpuDevice GPUDevice;

template<typename dummy>
struct NeighbourCovarianceOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice &d,

            const float *d_coords,
            const float *d_feats,
            const int* d_n_dixs,

            float * d_covariance,
            float * d_means,


            const int n_vert,
            const int n_coords,
            const int n_feat,
            const int n_neigh,
            const int n_vert_out) {

        grid_and_block gb(n_vert_out,512,n_feat,2);

        calcmeans<<<gb.grid(),gb.block()>>>( d_coords,d_feats,d_n_dixs,d_covariance, d_means,
                n_vert,n_coords,n_feat,n_neigh,n_vert_out);

        cudaDeviceSynchronize();

        calccov<<<gb.grid(),gb.block()>>>(d_coords,d_feats,d_n_dixs,d_covariance,d_means,
                n_vert,n_coords,n_feat,n_neigh,n_vert_out);

        cudaDeviceSynchronize();


    }
};

template struct NeighbourCovarianceOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
