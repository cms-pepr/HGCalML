//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "accumulate_knn_grad_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"


namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__device__
inline static float distanceWeight(const float& distsq){
    return distsq;
}

__global__
static void set_feature_grad_zero(
        float * d_out_grad_features,
        size_t n_vert,
        size_t n_feat
){

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || i_f >= n_feat)
        return;

    d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;

}

__global__
static void calc_feature_gradients(
        const float * d_grad_from_out_features,
        const int * d_max_feat_indices,
        const int * d_neigh_indices,
        const float * d_distances,

        const int n_vert,
        const int n_feat,
        const int n_neigh,

        const int n_grad_from_out_feat,

        float * d_out_grad_features,
        bool mean_and_max
){
    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nu_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert || nu_f >= n_feat)
        return;


    const float ginu = d_grad_from_out_features[I2D(i_v, nu_f, n_grad_from_out_feat)];
    float ginu_max = 0;
    int max_for_iv = -1;
    if(mean_and_max){
        ginu_max = d_grad_from_out_features[I2D(i_v, nu_f+n_feat, n_grad_from_out_feat)];
        max_for_iv = d_max_feat_indices[I2D(i_v,nu_f,n_feat)];
    }


    bool firstself=true;
    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

        if(m_v<0 || m_v >= n_vert) continue;//safe guard

        const float distsq_im = d_distances[I2D(i_v,i_i_n,n_neigh)];

        const float weight_im = distanceWeight(distsq_im);

        //if weight_im > some number?
        //     for (size_t nu_f = 0; nu_f < n_feat; nu_f++){

        float mean_contrib = ginu  / (float)n_neigh  * weight_im;
        float max_contrib = 0;
        if(m_v ==  max_for_iv){
            if(m_v == i_v){
                if(firstself){//count self just once
                    max_contrib = ginu_max * weight_im;
                    firstself=false;
                }
            }
            else{
                max_contrib = ginu_max * weight_im;
            }
        }

        //ATOMIC because of m_v which can occur in different threads. this is slow.. but needs to be atomic at least here...
        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)] , mean_contrib + max_contrib);


    }
}

__global__
static void calc_distance_gradients(
        const float * d_grad_from_out_features,
        const int *   d_max_feat_indices,
        const int *   d_neigh_indices,
        const float * d_distances,
        const float * d_feat,

        const int n_vert,
        const int n_feat,
        const int n_neigh,

        const int n_grad_from_out_feat,

        float * d_out_grad_distances,
        bool mean_and_max
){
    const size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t l = blockIdx.y * blockDim.y + threadIdx.y;

    if(m>=n_vert || l >= n_neigh)
        return;

    int l_g = d_neigh_indices[I2D(m,l,n_neigh)];
    if(l_g  < 0){
        if(l_g >= n_vert)//safe guard
            return;
        d_out_grad_distances[I2D(m,l,n_neigh)] = 0;
        return;
    }

    float mean_contrib=0;
    float max_contrib=0;

    //float dml = d_distances[I2D(m,l,n_neigh)]; //dlm == dml
    float expml = 1.;//distanceWeight(dml);

    for(size_t b_f=0;b_f<n_feat;b_f++){
        __syncthreads();

        bool firstself=true; ///To be checked!!! this needs to be per feature and stored!

        float gmb = d_grad_from_out_features[I2D(m, b_f, n_grad_from_out_feat)];
        float gmbmax = 0;
        if(mean_and_max)
            gmbmax  = d_grad_from_out_features[I2D(m, b_f+n_feat, n_grad_from_out_feat)];
        float flb = d_feat[I2D(l_g, b_f, n_feat)];

        mean_contrib += gmb * flb *expml;
        int maxform = -1;
        if(mean_and_max)
            maxform = d_max_feat_indices[I2D(m,b_f,n_feat)] ;
        if( l_g == maxform){
            if( l_g == m){
                if(firstself){
                    max_contrib += gmbmax * flb * expml;
                    firstself = false;
                }
            }
            else{
                max_contrib += gmbmax * flb * expml;
            }
        }

    }
    mean_contrib *= 1. / (float)n_neigh;
    max_contrib *= 1.;

    d_out_grad_distances[I2D(m,l,n_neigh)] = mean_contrib + max_contrib;

}


template <typename dummy>
struct AccumulateKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_from_out_features, // sum(V) x Fopout
            const float *d_distances, // sum(V) x N
            const float *d_feat, // sum(V) x S
            const int *d_max_feat_indices, // sum(V) x Fopin
            const int * d_neigh_indices, // sum(V) x N

            float *d_out_grad_distances, //sum(V) x S
            float *d_out_grad_features, //sum(V) x Fopin

            int n_vert,
            int n_neigh,
            int n_feat,

            int n_grad_from_out_feat,

            int n_moments,
            bool mean_and_max) {

        //try to minimise number of vertices per thread because of atomic add
        grid_and_block feat_par(n_vert, 64, n_feat, 8);
        if(n_feat >= 32)
            feat_par=grid_and_block(n_vert, 16, n_feat, 32);
        if(n_feat >= 64)
            feat_par=grid_and_block(n_vert, 8, n_feat, 64);
        if(n_feat >= 128)
            feat_par=grid_and_block(n_vert, 4, n_feat, 128);

        set_feature_grad_zero<<<feat_par.grid(), feat_par.block(), 0,  d.stream()>>>(d_out_grad_features, n_vert, n_feat);

        cudaDeviceSynchronize();

        calc_feature_gradients<<<feat_par.grid(), feat_par.block(), 0,  d.stream()>>>(
                d_grad_from_out_features,
                d_max_feat_indices,
                d_neigh_indices,
                d_distances,

                n_vert,
                n_feat,
                n_neigh,

                n_grad_from_out_feat,

                d_out_grad_features,
                mean_and_max);

        cudaDeviceSynchronize();


        grid_and_block neigh_par(n_vert, 128, n_neigh, 4);


        calc_distance_gradients<<<neigh_par.grid(), neigh_par.block(), 0,  d.stream()>>>(
                d_grad_from_out_features,
                d_max_feat_indices,
                d_neigh_indices,
                d_distances,
                d_feat,

                n_vert,
                n_feat,
                n_neigh,

                n_grad_from_out_feat,

                d_out_grad_distances,
                mean_and_max);

        cudaDeviceSynchronize();


    }
};



template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

