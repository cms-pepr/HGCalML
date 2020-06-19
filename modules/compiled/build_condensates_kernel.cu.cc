//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "build_condensates_kernel.h"
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
static void set_defaults(
        int *asso_idx,

        const int n_vert){

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    asso_idx[i_v] = -1;
}

__global__
static void set_defaults_feat(
        float *summed_features,

        const int n_vert,
        const int n_feat){

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;

    summed_features[I2D(i_v,i_f,n_feat)]=0;

}

__device__
static float distancesq(
        const int v_a,
        const int v_b,
        const float *d_ccoords,
        const int n_ccoords){
    float distsq=0;
    for(size_t i=0;i<n_ccoords;i++){
        float xa = d_ccoords[I2D(v_a,i,n_ccoords)];
        float xb = d_ccoords[I2D(v_b,i,n_ccoords)];
        distsq += (xa-xb)*(xa-xb);
    }
    return distsq;
}

__global__
static void check_and_collect(

        const int ref_vertex,
        const float *d_ccoords,

        int *asso_idx,

        const int n_vert,
        const int n_feat,
        const int n_ccoords,

        const int start_vertex,
        const int end_vertex,
        const float radius){

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x + start_vertex;
    if(i_v >= end_vertex)
        return;
    if(i_v == ref_vertex){
        asso_idx[i_v] = ref_vertex;
        return;
    }
    if(asso_idx[i_v] < 0){
        if(distancesq(ref_vertex,i_v,d_ccoords,n_ccoords) <= radius){ //sum features in parallel?
            asso_idx[i_v] = ref_vertex;
        }
    }
}

__global__
static void accumulate_features(
        const float *features,
        float *summed_features,
        const int *asso_idx,
        const int n_vert,
        const int n_feat
){
    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_f =  blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;
    //make atomic

    float toadd=features[I2D(i_v,i_f,n_feat)];
    int assoto = asso_idx[i_v];
    if(assoto>=0)
        atomicAdd(&summed_features[I2D(assoto,i_f,n_feat)] , toadd);



}

template <typename dummy>
struct BuildCondensatesOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice& d,

            const float *d_ccoords,
            const float *d_betas,
            const int *beta_sorting,
            const float *features,
            const int *row_splits,


            float *summed_features,
            int *asso_idx,

            const int n_vert,
            const int n_feat,
            const int n_ccoords,

            const int n_rs,

            const float radius,
            const float min_beta

) {


        grid_and_block gb_def(n_vert, 512);
        set_defaults<<<gb_def.grid(), gb_def.block(), 0, d.stream()>>>(asso_idx,n_vert);

        grid_and_block gb_def_f(n_vert, 64, n_feat, 8);
        set_defaults_feat<<<gb_def_f.grid(), gb_def_f.block(), 0, d.stream()>>>(summed_features,n_vert,n_feat);

        //copy row splits
        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost); //Async if needed, but these are just a few kB

        std::vector<float> cpu_d_betas(n_vert);
        cudaMemcpy(&cpu_d_betas.at(0),d_betas,n_vert*sizeof(float),cudaMemcpyDeviceToHost);

        std::vector<int> cpu_beta_sorting(n_vert);
        cudaMemcpy(&cpu_beta_sorting.at(0),beta_sorting,n_vert*sizeof(int),cudaMemcpyDeviceToHost);

        std::vector<int> cpu_asso_idx(n_vert, -1);

        cudaDeviceSynchronize();

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            const int start_vertex = cpu_rowsplits[j_rs];
            const int end_vertex = cpu_rowsplits[j_rs+1];
            const int range = end_vertex-start_vertex;

            for(size_t i_v = start_vertex; i_v < end_vertex; i_v++){
                size_t ref = cpu_beta_sorting[i_v] + start_vertex; //sorting is done per row split

                if(cpu_d_betas[ref] < min_beta)continue;
                if(cpu_asso_idx[ref] >=0) continue;
                //this converges actually quite quickly


                grid_and_block gb_cac(n_vert, 512);
                check_and_collect<<<gb_cac.grid(),gb_cac.block(),0,d.stream()>>>(
                        ref,
                        d_ccoords,
                        asso_idx,
                        n_vert,
                        n_feat,
                        n_ccoords,
                        start_vertex,
                        end_vertex,
                        radius);

                cudaDeviceSynchronize();
                cudaMemcpy(&cpu_asso_idx.at(start_vertex),
                        asso_idx+start_vertex,range*sizeof(int),
                        cudaMemcpyDeviceToHost);


            }


        }
        accumulate_features<<<gb_def_f.grid(),gb_def_f.block(),0,d.stream()>>>(features,summed_features,asso_idx,n_vert,n_feat);

        //cudaDeviceSynchronize();
    }


};



template struct BuildCondensatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
