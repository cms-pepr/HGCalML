//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "accumulate_knn_nd_grad_kernel.h"
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
float gpu_grad_distanceWeight(float distsq){
    return expf(-1.*ACCUMULATE_KNN_EXPONENT* distsq);
}


__device__
float gpu_delta(int k, int m){
    if (k==m) return 1;
    return 0;
}
__global__
void acc_knn_nd_gradkernel_features(
        const float *d_grad_from_out_features,
        const float *d_coord,
        const float *d_feat, // sum(V) x F
        const int *d_max_feat_indices,
        const int * d_neigh_indices,
        float *d_out_grad_coords,
        float *d_out_grad_features,
        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_feat,
        const int n_grad_from_out_feat,
        const int n_moments){


    // for (size_t i_v = 0; i_v < n_vert; i_v++)
    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nu_f = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t i_c  = blockIdx.z * blockDim.z + threadIdx.z;

    if(i_v >= n_vert){
        return;
    }
    if(nu_f >= n_feat){
        return;
    }
    if(i_c >= n_coords){
        return;
    }
    float gradinuc =d_grad_from_out_features[I3D(i_v, nu_f, i_c, n_grad_from_out_feat, n_coords)];
    float gradinucmax = d_grad_from_out_features[I3D(i_v, nu_f+n_feat, i_c,n_grad_from_out_feat, n_coords)];

    float vic = d_coord[I2D(i_v,i_c,n_coords)];
    size_t max_for_iv = d_max_feat_indices[I3D(i_v,nu_f,i_c,n_feat,n_coords)];

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){
        __syncthreads();
        size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        float vnc = d_coord[I2D(m_v,i_c,n_coords)];


        float distsq_im = (vic-vnc)*(vic-vnc);

        float weight_imnu = gpu_grad_distanceWeight(distsq_im);

        float contrib = gradinuc / (float)n_neigh  * weight_imnu;
        //from max

        if(m_v ==  max_for_iv){
            contrib += gradinucmax * weight_imnu;
        }

        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)], contrib);
       // d_out_grad_features[I2D(m_v, nu_f, n_feat)] += contrib;

    }

    __syncthreads();

}

__global__
void acc_knn_nd_gradkernel_coordinates(const float *d_grad_from_out_features,
        const float *d_coord,
        const float *d_feat, // sum(V) x F
        const int *d_max_feat_indices,
        const int * d_neigh_indices,

        float *d_out_grad_coords,
        float *d_out_grad_features,

        int n_vert,
        int n_neigh,
        int n_coords,
        int n_feat,

        int n_grad_from_out_feat,

        int n_moments){


    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t b_f = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t nu_c  = blockIdx.z * blockDim.z + threadIdx.z;

    if(i_v >= n_vert){
        return;
    }
    if(b_f >= n_feat){
        return;
    }
    if(nu_c >= n_coords){
        return;
    }

    float gibnu = d_grad_from_out_features[I3D(i_v, b_f, nu_c, n_grad_from_out_feat,n_coords)];
    float gilnu = d_grad_from_out_features[I3D(i_v, b_f+n_feat, nu_c, n_grad_from_out_feat,n_coords)];
    size_t max_for_iv = d_max_feat_indices[I3D(i_v, b_f, nu_c, n_feat,n_coords)];
    float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

        float mean_contrib = 0;
        float maxcontr = 0;

        for(size_t ii_k =0; ii_k< n_neigh ; ii_k++){
            __syncthreads();

            size_t k = d_neigh_indices[I2D(i_v, ii_k, n_neigh)];

            float diknu= xinu - d_coord[I2D(k,  nu_c,n_coords)];
            float fbk = d_feat[I2D(k, b_f, n_feat)];

            //get them out of sync here, all memory access done

            float ddelta = gpu_delta(m_v,k) - gpu_delta(m_v,i_v);
            if(!ddelta)
                continue;
            float wiknu = gpu_grad_distanceWeight(diknu*diknu);

            mean_contrib += gibnu * wiknu * fbk * diknu * ddelta ;
            if(k == max_for_iv){//or k ??? also wrong.. something with this index
                maxcontr += gilnu * wiknu * fbk * diknu * ddelta ;
            }

        }


        float add = 2. * ACCUMULATE_KNN_EXPONENT/(float) n_neigh * mean_contrib +
                2 * ACCUMULATE_KNN_EXPONENT * maxcontr;
        //ATOMIC this is slow.. but better if some are out of sync
        atomicAdd( &d_out_grad_coords[I2D(m_v, nu_c, n_coords)], add);
    }

    __syncthreads();

}

__global__
void acc_knn_nd_zero_coordinates(
        float *d_out_grad_coords,
        int n_vert,
        int n_coords){

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;

    if(i_v >= n_vert){
        return;
    }
    for(auto i_c: grid_stride_range_y(0, n_coords)){
        d_out_grad_coords[I2D(i_v, i_c, n_coords)] = 0;
    }

}
__global__
void acc_knn_nd_zero_features(
        float *d_out_grad_features,
        const int n_vert,
        const int n_feat){

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;

    if(i_v >= n_vert){
        return;
    }
    for(auto b_f: grid_stride_range_y(0, n_feat)){
        d_out_grad_features[I2D(i_v, b_f, n_feat)] = 0;
    }
}

template <typename dummy>
struct AccumulateKnnNdGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_from_out_features,
            const float *d_coord,
            const float *d_feat, // sum(V) x F
            const int *d_max_feat_indices,
            const int * d_neigh_indices,

            float *d_out_grad_coords,
            float *d_out_grad_features,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,
            const int n_grad_from_out_feat,
            const int n_moments) {


        //Ti1080 has 768 blocks
        //zero out in threads
        dim3 fgridz(n_vert/32+1, n_feat/16+1);
        dim3 fblockz(32,4);

        acc_knn_nd_zero_features<<<fgridz,fblockz>>> (d_out_grad_features, n_vert, n_feat);


        dim3 fgridzc(n_vert/32+1, n_coords/4+1);
        dim3 fblockzc(32,2);

        acc_knn_nd_zero_coordinates<<<fgridzc , fblockzc>>> (d_out_grad_coords, n_vert, n_coords);


        cudaDeviceSynchronize();


        dim3 fgrid(n_vert/32+1, n_feat/4+1 ,n_coords/4+1);
        dim3 fblock(32,4,4);




        acc_knn_nd_gradkernel_features<<<fgrid, fblock, 0, d.stream()>>>(
                d_grad_from_out_features,
                d_coord,
                d_feat,
                d_max_feat_indices,
                d_neigh_indices,
                d_out_grad_coords,
                d_out_grad_features,
                n_vert,
                n_neigh,
                n_coords,
                n_feat,
                n_grad_from_out_feat,
                n_moments);

        cudaDeviceSynchronize();

        acc_knn_nd_gradkernel_coordinates<<<fgrid, fblock, 0, d.stream()>>>(
                d_grad_from_out_features,
                d_coord,
                d_feat,
                d_max_feat_indices,
                d_neigh_indices,
                d_out_grad_coords,
                d_out_grad_features,
                n_vert,
                n_neigh,
                n_coords,
                n_feat,
                n_grad_from_out_feat,
                n_moments);

        cudaDeviceSynchronize();
    }
};



template struct AccumulateKnnNdGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

