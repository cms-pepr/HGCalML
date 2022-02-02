#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_grad_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
namespace functor {


inline static float distanceWeight(const float& distsq){
    return distsq;
}

static void set_feature_grad_zero(
        float * d_out_grad_features,
        size_t n_vert,
        size_t n_feat
){

    for (size_t i_v = 0; i_v < n_vert; i_v++){
        for (size_t i_f = 0; i_f < n_feat; i_f++)
            d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;
    }
}

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
    for (size_t i_v = 0; i_v < n_vert; i_v++){
        for(size_t nu_f=0;nu_f<n_feat;nu_f++){

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
                if(m_v<0) continue;

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
                d_out_grad_features[I2D(m_v, nu_f, n_feat)] += mean_contrib + max_contrib;

                //     }
            }
        }
    }
}

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
    for (size_t m = 0; m < n_vert; m++){

        for (size_t l = 0; l < n_neigh; l++){


            int l_g = d_neigh_indices[I2D(m,l,n_neigh)];
            if(l_g  < 0 ){
                d_out_grad_distances[I2D(m,l,n_neigh)] = 0;
                continue;
            }

            float mean_contrib=0;
            float max_contrib=0;

            float dml = d_distances[I2D(m,l,n_neigh)]; //dlm == dml
            float expml = 1.; //linear scaling so grad 1 distanceWeight(dml);

            for(size_t b_f=0;b_f<n_feat;b_f++){

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
            max_contrib *= 1;

            d_out_grad_distances[I2D(m,l,n_neigh)] = mean_contrib + max_contrib;
        }
    }
}

// CPU specialization
template<typename dummy>
struct AccumulateKnnGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

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

        //CPU implementation

        //set zero
        set_feature_grad_zero(d_out_grad_features, n_vert, n_feat);

        calc_feature_gradients(
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

        calc_distance_gradients(
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

    }
};

template<typename Device>
class AccumulateKnnGradOp : public OpKernel {
public:
    explicit AccumulateKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &t_grad_from_out_features = context->input(0);
        const Tensor &t_distances = context->input(1);
        const Tensor &t_feat = context->input(2);
        const Tensor &t_neigh_indices = context->input(3);
        const Tensor &t_max_feat_indices = context->input(4);

        int n_in_grad_feat = t_grad_from_out_features.dim_size(1);


        int n_vert = t_grad_from_out_features.dim_size(0);

        int n_neigh = t_neigh_indices.dim_size(1);
        int n_feat = t_feat.dim_size(1);
        int n_moments = 0;

        //auto detect
        const bool mean_and_max = n_in_grad_feat > n_feat;

        TensorShape outputShapeDist;
        outputShapeDist.AddDim(n_vert);
        outputShapeDist.AddDim(n_neigh);


        TensorShape outputShapeFeat;
        outputShapeFeat.AddDim(n_vert);
        outputShapeFeat.AddDim(n_feat);


        Tensor *t_out_grad_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShapeDist, &t_out_grad_distances));

        Tensor *t_out_grad_features = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShapeFeat, &t_out_grad_features));


        AccumulateKnnGradOpFunctor<Device, int>()(

                context->eigen_device<Device>(),

                t_grad_from_out_features.flat<float>().data(),
                t_distances.flat<float>().data(),
                t_feat.flat<float>().data(),
                t_max_feat_indices.flat<int>().data(),
                t_neigh_indices.flat<int>().data(),

                t_out_grad_distances->flat<float>().data(),
                t_out_grad_features->flat<float>().data(),

                n_vert,
                n_neigh,
                n_feat,

                n_in_grad_feat,
                n_moments,
                mean_and_max
        );



    }

};

REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_CPU), AccumulateKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_GPU), AccumulateKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow



