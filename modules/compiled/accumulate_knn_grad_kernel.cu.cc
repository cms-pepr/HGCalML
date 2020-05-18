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
float gpu_grad_distanceWeight(float distsq){
   // if(!distsq)return 1;
    //  const float cutoffsq = -log(0.01)/ACCUMULATE_KNN_EXPONENT;
    //  if(distsq>cutoff)
    //      return 0;
    return expf(-1.*ACCUMULATE_KNN_EXPONENT* distsq);
}
__device__
inline float gpu_distWeightD(const float *d_coord, size_t i, size_t j, size_t n_coords){

    const float cutoff = sqrt(-log(0.01)/ACCUMULATE_KNN_EXPONENT);//cut off at 1% contribution

    float distsq=0;
    for(size_t i_c=0;i_c<n_coords;i_c++){
        float xic = d_coord[I2D(i,i_c,n_coords)];
        float xkc = d_coord[I2D(j,  i_c,n_coords)];

        //  if(fabs(xic-xkc) > cutoff){ //opt
        //      distsq = cutoff*cutoff;
        //      break;
        //  }

        distsq += (xic-xkc)*(xic-xkc);
    }
    return gpu_grad_distanceWeight(distsq);
}
__device__
float gpu_delta(int k, int m){
    if (k==m) return 1;
    return 0;
}
__global__
void acc_knn_gradkernel_features(
        const float *d_grad_from_out_features,
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

    // for (size_t i_v = 0; i_v < n_vert; i_v++)
    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert){
        return;
    }
    size_t nu_f = blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_f >= n_feat){
        return;
    }
    //  for (size_t nu_f = 0; nu_f < n_feat; nu_f++)
  //  d_out_grad_features[I2D(i_v, nu_f, n_feat)] = 0;

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

        float distsq_im = 0;
        for(size_t i_c=0;i_c<n_coords;i_c++){
            float vic = d_coord[I2D(i_v,i_c,n_coords)];
            float vnc = d_coord[I2D(m_v,i_c,n_coords)];
            distsq_im += (vic-vnc)*(vic-vnc);
        }
        float weight_im = gpu_grad_distanceWeight(distsq_im);

        //if weight_im > some number?
        //     for (size_t nu_f = 0; nu_f < n_feat; nu_f++){

        float contrib=0;
        //from mean
        contrib +=d_grad_from_out_features[I2D(i_v, nu_f, n_grad_from_out_feat)]  / (float)n_neigh  * weight_im;
        float maxgrad = d_grad_from_out_features[I2D(i_v, nu_f+n_feat, n_grad_from_out_feat)];
        //from max
        if(m_v == d_max_feat_indices[I2D(i_v,nu_f,n_feat)] ){
            contrib += maxgrad * weight_im;
        }

        //ATOMIC because of m_v which can occur in different threads. this is slow.. but needs to be atomic at least here...
        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)], contrib);

        //     }
    }

}

__global__
void acc_knn_gradkernel_coordinates(const float *d_grad_from_out_features,
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

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;
    //set zero
    // for (size_t i_v = 0; i_v < n_vert; i_v++){
    //  for(size_t nu_c=0;nu_c<n_coords;nu_c++)
   // d_out_grad_coords[I2D(i_v,nu_c,n_coords)] = 0;
    //  }



    //look for neighbours, add gradient term to every *neighbour*

    //for (size_t i_v = 0; i_v < n_vert; i_v++) {

    //these should be all vertices that have m as neighbour, not the other way around - here is the problem
    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

        //     for(size_t nu_c = 0; nu_c < n_coords; nu_c++){

        float mean_contrib = 0;
        float maxcontr = 0;

        //  for (size_t b_f = 0; b_f < n_feat; b_f++){
        //  float thisfeat_mean_contr = 0;
        // float thisfeat_max_contr = 0;

        // m_v == k && m_v != i_v
        // m_v != k && m_v == i_v (*= -1)
        //

        for(size_t ii_k =0; ii_k< n_neigh ; ii_k++){
            size_t k = d_neigh_indices[I2D(i_v, ii_k, n_neigh)];
            float ddelta = gpu_delta(m_v,k) - gpu_delta(m_v,i_v);
            if(!ddelta)
                continue;

            float diknu= d_coord[I2D(i_v,nu_c,n_coords)] - d_coord[I2D(k,  nu_c,n_coords)];
            //if(fabs(diknu)<0.01)continue;

            //possible improvement: if fabs(diknu) <  0.05 or > 0.5 continue (will be absorbed by diknu * exp(-10 *...))
            // less than 5% contribution
            // the whole contribution can never exceed about 0.15 * in_feature (for s=10)..

            float wik = gpu_distWeightD(d_coord,i_v,k,n_coords);
            //if(fabs(wik * diknu)< 0.002) continue;

            for (size_t b_f = 0; b_f < n_feat; b_f++){

                mean_contrib +=  wik * d_feat[I2D(k, b_f, n_feat)] * diknu
                        * ddelta * d_grad_from_out_features[I2D(i_v, b_f, n_grad_from_out_feat)];

                if(k == d_max_feat_indices[I2D(i_v,b_f,n_feat)] ){
                    maxcontr += wik * d_feat[I2D(k, b_f, n_feat)] * diknu
                            * ddelta * d_grad_from_out_features[I2D(i_v, b_f, n_grad_from_out_feat)];
                }

            }


        }
        float add = 2. * ACCUMULATE_KNN_EXPONENT/(float) n_neigh * mean_contrib +
                2 * ACCUMULATE_KNN_EXPONENT * maxcontr;
        //ATOMIC this is slow..
        atomicAdd( &d_out_grad_coords[I2D(m_v, nu_c, n_coords)], add);
    }
    //  }
    //}
}

template <typename dummy>
struct AccumulateKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_from_out_features,
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

            int n_moments) {

        //Ti1080 has 768 blocks

        dim3 fgrid(n_vert/96+1, n_feat/8 +1);
        dim3 fblock(96,8);


        acc_knn_gradkernel_features<<<fgrid, fblock, 0,  d.stream()>>>(
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


        dim3 cgrid(n_vert/192+1, n_coords/4+1);
        dim3 cblock(192,4);

        acc_knn_gradkernel_coordinates<<<cgrid, cblock, 0,  d.stream()>>>(d_grad_from_out_features,d_coord,d_feat,d_max_feat_indices,
                d_neigh_indices,d_out_grad_coords,d_out_grad_features,
                n_vert,n_neigh,n_coords,n_feat,n_grad_from_out_feat,n_moments);

    }
};



template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA

