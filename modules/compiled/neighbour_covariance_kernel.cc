
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "neighbour_covariance_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>


namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


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

    for (int i_v = 0; i_v < n_vert_out; i_v++) {
        for (int i_f = 0; i_f < n_feat; i_f++) {
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
        }//i_f
    }//i_v
}

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

    for (int i_v = 0; i_v < n_vert_out; i_v++) {
        for (int i_f = 0; i_f < n_feat; i_f++) {
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
        }//i_f
    }//i_v
}

// CPU specialization
template<typename dummy>
struct NeighbourCovarianceOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

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

        //vert and feat
        calcmeans( d_coords,d_feats,d_n_dixs,d_covariance, d_means,
                n_vert,n_coords,n_feat,n_neigh,n_vert_out);


        calccov(d_coords,d_feats,d_n_dixs,d_covariance,d_means,
                n_vert,n_coords,n_feat,n_neigh,n_vert_out);


    }
};


template<typename Device>
class NeighbourCovarianceOp : public OpKernel {
public:
    explicit NeighbourCovarianceOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {

        /*
         * .Input("coordinates: float32")
         * .Input("features: float32")
         * .Input("n_idxs: int32")
         * .Output("covariance: float32")
         * .Output("means float32");
         */

        const Tensor &t_coords = context->input(0);
        const Tensor &t_feats  = context->input(1);
        const Tensor &t_n_idxs = context->input(2);

        const int n_vert = t_coords.dim_size(0);
        const int n_coords = t_coords.dim_size(1);
        const int n_feat = t_feats.dim_size(1);
        const int n_vert_out = t_n_idxs.dim_size(0);//can be different!
        const int n_neigh = t_n_idxs.dim_size(1);

        //covariance has N_f x N(N+1)/2


        Tensor * t_covariance=NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            (long long int)n_vert_out,
                    (long long int)n_feat,
                    (long long int)(n_coords*(n_coords+1)/2)
        }), &t_covariance));

        Tensor * t_means=NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({
            (long long int)n_vert_out,
                    (long long int)n_feat,
                    (long long int)n_coords
        }), &t_means));

        NeighbourCovarianceOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_coords.flat<float>().data(),
                t_feats.flat<float>().data(),
                t_n_idxs.flat<int>().data(),

                t_covariance->flat<float>().data(),
                t_means->flat<float>().data(),


                n_vert,
                n_coords,
                n_feat,
                n_neigh,
                n_vert_out
        );


    }

};

REGISTER_KERNEL_BUILDER(Name("NeighbourCovariance").Device(DEVICE_CPU), NeighbourCovarianceOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct NeighbourCovarianceOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("NeighbourCovariance").Device(DEVICE_GPU), NeighbourCovarianceOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
