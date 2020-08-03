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
        int * is_cpoint,
        const float * d_betas,
        float * temp_betas,
        const int n_vert){

    size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>n_vert)
        return;

    asso_idx[i_v] = -1;

    temp_betas[i_v] = d_betas[i_v];

    is_cpoint[i_v] = 0;//not needed on GPU?
}


__device__
static float distancesq(const int v_a,
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
static void get_max_beta(
        const float* temp_betas,
        int *asso_idx,
        int * is_cpoint,
        int * maxidx,

        const int n_vert,
        const int start_vertex,
        const int end_vertex,
        const float min_beta){

    // THIS NEEDS IMPROVEMENTS!

    //this needs to be some smart algo here
    //it will be called N_condensate times at least
    int ref=-1;
    float max = min_beta;
    for(int i_v=start_vertex;i_v<end_vertex;i_v++){
        float biv = temp_betas[i_v];
        if(biv > max && asso_idx[i_v] < 0){
            max=biv;
            ref=i_v;
        }

    }

    //if none can be found set ref to -1
    *maxidx = ref;
    if(ref>=0){
        is_cpoint[ref]=1;
        asso_idx[ref]=ref;
    }
}

__global__
static void check_and_collect(

        const int ref_vertex,
        const float ref_beta,
        const float *d_ccoords,
        const float *d_betas,

        int *asso_idx,
        float * temp_betas,

        const int n_vert,
        const int n_ccoords,

        const int start_vertex,
        const int end_vertex,
        const float radius,
        const float min_beta,
        const bool soft){

    int i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>n_vert)
        return;

    if(asso_idx[i_v] < 0){
        float distsq = distancesq(ref_vertex,i_v,d_ccoords,n_ccoords);

        if(soft){
            //should the reduction in beta be using the original betas or the modified ones...?
            //go with original betas
            float moddist = 1 - sqrt(distsq / radius );
            if(moddist < 0)
                moddist = 0;
            float subtract =  moddist * ref_beta;
            temp_betas[i_v] -= subtract;
            if(temp_betas[i_v] <= min_beta && moddist)
                asso_idx[i_v] = ref_vertex;
        }
        else{
            if(distsq <= radius){ //sum features in parallel?
                asso_idx[i_v] = ref_vertex;
            }
        }
    }





}





template <typename dummy>
struct BuildCondensatesOpFunctor<GPUDevice, dummy> {
    void operator()(
            const GPUDevice& d,

            const float *d_ccoords,
            const float *d_betas,
            const int *row_splits,

            int *asso_idx,
            int *is_cpoint,
            float * temp_betas,

            const int n_vert,
            const int n_ccoords,

            const int n_rs,

            const float radius,
            const float min_beta,
            const bool soft

) {

        grid_and_block gb_vert(n_vert, 512);

        set_defaults<<<gb_vert.grid(),gb_vert.block()>>>(asso_idx,is_cpoint,d_betas,temp_betas,n_vert);

        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost); //Async if needed, but these are just a few kB

        cudaDeviceSynchronize();
        //copy RS to CPU

        int *gref=0;
        int ref=0;
        float ref_beta = 0;
        cudaMalloc((void**)&gref, sizeof(int));
        cudaMemcpy(gref, &ref, sizeof(int), cudaMemcpyHostToDevice);

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            const int start_vertex = cpu_rowsplits[j_rs];
            const int end_vertex = cpu_rowsplits[j_rs+1];


            get_max_beta<<<1,1>>>(temp_betas,asso_idx,is_cpoint,gref,n_vert,start_vertex,end_vertex,min_beta);
            //copy ref back

            cudaMemcpy(&ref, gref, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ref_beta, &d_betas[ref], sizeof(float), cudaMemcpyDeviceToHost);
            //copy ref and refBeta from GPU to CPU

            grid_and_block gb_rsvert(end_vertex-start_vertex, 512);

            while(ref>=0){

                // if(asso_idx[ref] >=0) continue; //
                // if(temp_betas[ref] < min_beta)continue;
                //probably better to copy here instead of accessing n_vert times in GPU mem


                check_and_collect<<<gb_rsvert.grid(),gb_rsvert.block()>>>(
                        ref,
                        ref_beta,
                        d_ccoords,
                        d_betas,
                        asso_idx,
                        temp_betas,
                        n_vert,
                        n_ccoords,
                        start_vertex,
                        end_vertex,
                        radius,
                        min_beta,
                        soft);

                get_max_beta<<<1,1>>>(temp_betas,asso_idx,is_cpoint,gref,n_vert,start_vertex,end_vertex,min_beta);
                //copy ref and refBeta from GPU to CPU
                cudaMemcpy(&ref, gref, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&ref_beta, &d_betas[ref], sizeof(float), cudaMemcpyDeviceToHost);

            }


        }


        cudaFree(gref);

    }


};



template struct BuildCondensatesOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
