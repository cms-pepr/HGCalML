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
void acc_knn_nd_gradkernel_features_fromsum(
        const float *d_grad_from_sum_features,
        const int * d_neigh_indices,
        float *d_out_grad_features,
        const int n_vert,
        const int n_neigh,
        const int n_feat){

    const size_t i_v  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nu_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(i_v >= n_vert){
        return;
    }
    if(nu_f >= n_feat){
        return;
    }

    float gradinucfeatsum = d_grad_from_sum_features[I2D(i_v, nu_f, n_feat)];

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m_v<0) continue;

        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)], gradinucfeatsum);
    }
}



__global__
void acc_knn_nd_gradkernel_features(
        const float *d_grad_from_out_features,
        const float *d_grad_from_sum_features,

        const float *d_coord,
        const float *d_feat, // sum(V) x F
        const float *d_orig_out_feat, // sum(V) x F
        const float *d_orig_out_feat_sum,
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
    //float gradinucfeatsum = d_grad_from_sum_features[I2D(i_v, nu_f, n_feat)];


    float featsum = d_orig_out_feat_sum[I2D(i_v, nu_f,n_feat)];

    float grad_m_mean = 0;
    float grad_m_var  = 0;
    float grad_m_skew = 0;

    float m_mean = 0;
    float m_var  = 0;
    float m_skew = 0;

    if(n_moments > 0){
        grad_m_mean = d_grad_from_out_features[I3D(i_v, nu_f+ 2*n_feat, i_c,n_grad_from_out_feat, n_coords)];
        m_mean = d_orig_out_feat[I3D(i_v, nu_f+2*n_feat, i_c,n_grad_from_out_feat, n_coords)];
    }
    if(n_moments > 1){
        grad_m_var  = d_grad_from_out_features[I3D(i_v, nu_f+ 3*n_feat, i_c,n_grad_from_out_feat, n_coords)];
        m_var = d_orig_out_feat[I3D(i_v, nu_f+3*n_feat, i_c,n_grad_from_out_feat, n_coords)];
    }
    if(n_moments > 2){
        grad_m_skew = d_grad_from_out_features[I3D(i_v, nu_f+ 4*n_feat, i_c,n_grad_from_out_feat, n_coords)];
        m_skew = d_orig_out_feat[I3D(i_v, nu_f+4*n_feat, i_c,n_grad_from_out_feat, n_coords)];
    }

    float vic = d_coord[I2D(i_v,i_c,n_coords)];
    size_t max_for_iv = d_max_feat_indices[I3D(i_v,nu_f,i_c,n_feat,n_coords)];

    bool first_self=true;
    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m_v<0) continue;


        float vnc = d_coord[I2D(m_v,i_c,n_coords)];


        float dist_im = (vnc-vic);

        float weight_imnu = gpu_grad_distanceWeight(dist_im*dist_im);

        float contrib = gradinuc / (float)n_neigh  * weight_imnu;
        //from max

        if(m_v ==  max_for_iv){
            if(m_v == i_v){
                if(first_self){
                    first_self=false;
                    contrib += gradinucmax * weight_imnu;
                }
            }
            else{
                contrib += gradinucmax * weight_imnu;
            }
        }

        if(n_moments>0 && featsum){
            //just gradient w.r.t. features. in addition, there is also gradients w.r.t. mean directly

            //gradient w.r.t. 1/sum (gives -delta(m,k)/fsum
            contrib -=  1./featsum * (grad_m_mean*m_mean);// + grad_m_var*m_var + grad_m_skew*m_skew);

            //grad w.r.t sum gives 1/sum delta(m,k) D^N_ki
            float disttomean = dist_im-m_mean;
            contrib += 1./featsum * dist_im * grad_m_mean;
          //  contrib += 1./featsum * disttomean*disttomean * grad_m_var;
          //  contrib += 1./featsum * disttomean*disttomean*disttomean * grad_m_skew;
        }

        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)], contrib);
       // d_out_grad_features[I2D(m_v, nu_f, n_feat)] += contrib;

    }

    __syncthreads();

}

__global__
void acc_knn_nd_gradkernel_coordinates(
        const float *d_grad_from_out_features,
        const float *d_grad_from_sum_features,

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

    float self_mean_contrib = 0;
    float self_max_contrib = 0;
    for(size_t ii_k =0; ii_k< n_neigh ; ii_k++){

        int k = d_neigh_indices[I2D(i_v, ii_k, n_neigh)];
        if(k<0) continue;
        if(k == i_v)
            continue;

        float diknu= d_coord[I2D(k,  nu_c,n_coords)] - xinu ;
        float fbk = d_feat[I2D(k, b_f, n_feat)];

        //get them out of sync here, all memory access done

        float ddelta =  (gpu_delta(i_v,i_v) - gpu_delta(i_v,k));
        if(!ddelta) // m == k (see below)
            continue;
        float wiknu = gpu_grad_distanceWeight(diknu*diknu);

        self_mean_contrib +=  wiknu * fbk * diknu * ddelta ;
        if(k == max_for_iv){//or k ??? also wrong.. something with this index
            self_max_contrib +=  wiknu * fbk * diknu * ddelta ;
        }
        //possible sync?
    }

    float add = 2. * gibnu *ACCUMULATE_KNN_EXPONENT/(float) n_neigh *self_mean_contrib
            + 2 * gilnu *ACCUMULATE_KNN_EXPONENT * self_max_contrib;

    atomicAdd( &d_out_grad_coords[I2D(i_v, nu_c, n_coords)], add);


    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){//can integrate the above loop???

        int m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m_v<0) continue;

        float mean_contrib = 0;
        float maxcontr = 0;

        if(m_v != i_v){ // m != i, therefore m must be k
            size_t k = m_v;

            float diknu= d_coord[I2D(k,  nu_c,n_coords)] - xinu ;
            float fbk = d_feat[I2D(k, b_f, n_feat)];

            //get them out of sync here, all memory access done

            float ddelta = (gpu_delta(m_v,i_v) - gpu_delta(m_v,k));
            if(!ddelta)
                continue;
            float wiknu = gpu_grad_distanceWeight(diknu*diknu);

            mean_contrib +=  wiknu * fbk * diknu * ddelta ;
            if(k == max_for_iv){//or k ??? also wrong.. something with this index
                maxcontr +=  wiknu * fbk * diknu * ddelta ;
            }
        }



        float add = 2. * gibnu *ACCUMULATE_KNN_EXPONENT/(float) n_neigh * mean_contrib +
                2 * gilnu *ACCUMULATE_KNN_EXPONENT * maxcontr;
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
            const float *d_grad_from_sum_features,

            const float *d_coord,
            const float *d_feat, // sum(V) x F
            const float *d_orig_out_feat,
            const float *d_orig_out_feat_sum,
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
        dim3 fblockz(32,16);

        acc_knn_nd_zero_features<<<fgridz,fblockz, 0, d.stream()>>> (d_out_grad_features, n_vert, n_feat);


        dim3 fgridzc(n_vert/32+1, n_coords/4+1);
        dim3 fblockzc(32,4);

        acc_knn_nd_zero_coordinates<<<fgridzc , fblockzc, 0, d.stream()>>> (d_out_grad_coords, n_vert, n_coords);


        cudaDeviceSynchronize();

        acc_knn_nd_gradkernel_features_fromsum<<<dim3(n_vert/32+1,n_feat/16+1) , dim3(32,16), 0, d.stream()>>> (
                d_grad_from_sum_features,
                 d_neigh_indices,
                d_out_grad_features,
                n_vert,
                n_neigh,
                n_feat);

        cudaDeviceSynchronize();

        dim3 fgrid(n_vert/4+1, n_feat/32+1 ,n_coords/4+1);
        dim3 fblock(4,32,4);




        acc_knn_nd_gradkernel_features<<<fgrid, fblock, 0, d.stream()>>>(
                d_grad_from_out_features,
                d_grad_from_sum_features,
                d_coord,
                d_feat,
                d_orig_out_feat,
                d_orig_out_feat_sum,
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
                d_grad_from_sum_features,
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

