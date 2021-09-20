//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "select_mod_knn_grad_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"

namespace tensorflow {
namespace functor {

namespace gpu{



__global__
static void select_mod_knn_grad_selfloop_kernel(
        const float *d_grad_dist, // V x N
        const int *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,
        const float *d_coord_mod,

        float * d_grad_coord,

        const int n_vert,
        const int n_neigh,
        const int n_coords) {

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    //const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    float self_contrib=0;
    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int k = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(k<0) break;



        float modifier=0;
        for(size_t la=0;la<n_coords;la++){
            const float Rinula = d_coord_mod[I3D(i_v,nu_c,la,n_coords,n_coords)];
            for(size_t a=0;a<n_coords;a++){
                float xia = d_coord[I2D(i_v,a,n_coords)];
                float xma = d_coord[I2D(k,a,n_coords)];
                modifier += d_coord_mod[I3D(i_v,a,la,n_coords,n_coords)] * (xia - xma) * Rinula;
            }
        }
        const float gik = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        //???????
        self_contrib += 2 * modifier * gik; //2. * gik * (xinu - xknu);

    }
    //add mod

    d_grad_coord[I2D(i_v,nu_c,n_coords)] = self_contrib;
}

__global__
static void select_mod_knn_grad_neighloop_kernel(
        const float *d_grad_dist, // V x N
        const int *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,
        const float *d_coord_mod,

        float * d_grad_coord,

        const int n_vert,
        const int n_neigh,
        const int n_coords){


    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    //const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m<0) break;//padded with -1

        const float gim = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        //const float xmnu = d_coord[I2D(m,nu_c,n_coords)];

        float modifier=0;
        for(size_t la=0;la<n_coords;la++){
            const float Rinula = d_coord_mod[I3D(i_v,nu_c,la,n_coords,n_coords)];
            for(size_t a=0;a<n_coords;a++){
                float xia = d_coord[I2D(i_v,a,n_coords)];
                float xma = d_coord[I2D(m,a,n_coords)];
                modifier += d_coord_mod[I3D(i_v,a,la,n_coords,n_coords)] * (xia - xma) * Rinula;
            }
        }

        float add = - 2. * gim * modifier;// (xinu - xmnu);

        atomicAdd( &d_grad_coord[I2D(m, nu_c, n_coords)], add);

    }
}

__global__
static void select_mod_knn_grad_mod_kernel(
        const float *d_grad_dist, // V x N
        const int *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,
        const float *d_coord_mod,

        float * d_grad_coord_mod,

        const int n_vert,
        const int n_neigh,
        const int n_coords){


    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    size_t tau= blockIdx.z * blockDim.z + threadIdx.z;
    if(tau >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];
    //const float Rtaunui = d_coord_mod[I3D(i_v,tau,nu_c,n_coords,n_coords)];

    float contrib=0;

    for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        int m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m<0) break;//padded with -1

        const float gim = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        const float xmnu = d_coord[I2D(m,nu_c,n_coords)];
        const float dimnu = (xinu - xmnu);

        float coorddist=0;
        for(size_t a=0;a<n_coords;a++){ //contract over axis 1
            float Rivatau = d_coord_mod[I3D(i_v, a, tau, n_coords, n_coords)];
            coorddist += Rivatau * (d_coord[I2D(i_v,a,n_coords)]-d_coord[I2D(m,a,n_coords)]);
        }
        contrib += gim *  coorddist * dimnu ;// * Rtaunui;
    }
    d_grad_coord_mod[I3D(i_v,nu_c,tau,n_coords,n_coords)] = 2. * contrib;
}



}//gpu

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SelectModKnnGradOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,
            const float *d_coord_mod,

            float * d_grad_coord,
            float * d_grad_coord_mod,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {


        //for too low n, d_indices might need to be initialised with some number the
        // rest of the code understands.. maybe -1?

        //just loop over n_rs, in a realistic setting these shouldn't be more than a handful entries

        grid_and_block gb(n_vert,256,n_coords,4);


        gpu::select_mod_knn_grad_selfloop_kernel<<<gb.grid(),gb.block(), 0, d.stream()>>>(
                d_grad_dist,
                d_indices,
                d_dist,
                d_coord,
                d_coord_mod,
                d_grad_coord,
                n_vert,
                n_neigh,
                n_coords
        );

        cudaDeviceSynchronize();

        gpu::select_mod_knn_grad_neighloop_kernel<<<gb.grid(),gb.block(), 0, d.stream()>>>(
                d_grad_dist,
                d_indices,
                d_dist,
                d_coord,
                d_coord_mod,
                d_grad_coord,
                n_vert,
                n_neigh,
                n_coords
        );

        cudaDeviceSynchronize();

        grid_and_block gbr(n_vert,512,n_coords,1,n_coords,1);

        gpu::select_mod_knn_grad_mod_kernel<<<gbr.grid(),gbr.block(), 0, d.stream()>>>(
                d_grad_dist, // V x N
                d_indices,
                d_dist,
                d_coord,
                d_coord_mod,

                d_grad_coord_mod,

                n_vert,
                n_neigh,
                n_coords);


    }

};



template struct SelectModKnnGradOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
