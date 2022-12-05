
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "push_knn_grad_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template<typename dummy>
struct PushKnnGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_grad,

            const float *d_weights,
            const float *d_feat,
            const int *d_idxs,

            float *d_feat_grad,
            float *d_w_grad,

            int n_vert,
            int n_neigh,
            int n_feat) {

        printf("calling push knn grad cpu kernel for %d vertices.\n", n_vert);
        //feature gradient

        for (size_t i_v = 0; i_v < n_vert; i_v++) {
            for (size_t i_f = 0; i_f < n_feat; i_f++) {

                float fgrad = 0;

                for (size_t i_n = 0; i_n < n_neigh; i_n++){

                    int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
                    if(nidx < 0)
                        continue;

                    float w = d_weights[I2D(i_v, i_n, n_neigh)];

                    fgrad += d_grad[I2D(nidx, i_f, n_feat)] * w;

                }
                if(fgrad != fgrad)
                    printf("feature gradient is non for vertex %d, feature %d \n", i_v, i_f);
                d_feat_grad[I2D(i_v, i_f, n_feat)] = fgrad;
            }
        }

        // weight gradient
        for (size_t i_v = 0; i_v < n_vert; i_v++) {

            for (size_t i_n = 0; i_n < n_neigh; i_n++){

                float wgrad = 0;
                int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];
                if(nidx < 0){
                    d_w_grad[I2D(i_v,i_n,n_neigh)] = 0;
                    continue;
                }

                for (size_t i_f = 0; i_f < n_feat; i_f++) {

                    float f = d_feat[I2D(i_v, i_f, n_feat)];
                    wgrad += d_grad[I2D(nidx, i_f, n_feat)] * f;

                }

                if(wgrad != wgrad)
                    printf("weight gradient is non for vertex %d, neighbour %d \n", i_v, i_n);
                d_w_grad[I2D(i_v,i_n,n_neigh)] = wgrad;

            }

        }

}
};


template<typename Device>
class PushKnnGradOp : public OpKernel {
public:
    explicit PushKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        /*
         *  .Input("grad: float32") //change to distances!!
            .Input("weights: float32") //change to distances!!
            .Input("features: float32")
            .Input("indices: int32")
            .Output("feature_grad: float32")
            .Output("weight_grad: float32");
         *
         *
         */

        const Tensor &d_grad_tensor = context->input(0);
        const Tensor &d_weight_tensor = context->input(1);
        const Tensor &d_feat_tensor = context->input(2);
        const Tensor &d_idxs_tensor = context->input(3);


        int n_vert = d_weight_tensor.dim_size(0);
        int n_neigh = d_idxs_tensor.dim_size(1);
        int n_feat = d_feat_tensor.dim_size(1);


        OP_REQUIRES(context, n_vert == d_grad_tensor.dim_size(0) && d_grad_tensor.dim_size(1) == d_feat_tensor.dim_size(1),
                    errors::InvalidArgument("PushKnnGradOp expects feat tensor and grad tensor second dim to match."));

        OP_REQUIRES(context, n_vert == d_idxs_tensor.dim_size(0) && n_vert == d_feat_tensor.dim_size(0),
                    errors::InvalidArgument("PushKnnGradOp expects first dimensions of all inputs to match."));

        OP_REQUIRES(context, n_neigh == d_weight_tensor.dim_size(1),
                    errors::InvalidArgument("PushKnnGradOp expects second dimension of distance and neighbour index tensor to match"));


        // after testing basic functionality!
        // n_out_feat += n_moments * n_feat * n_coords;

        Tensor *out_weight_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {n_vert, n_neigh}, &out_weight_grad));


        Tensor *out_feat_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, {n_vert, n_feat}, &out_feat_grad));


        PushKnnGradOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                d_grad_tensor.flat<float>().data(),
                d_weight_tensor.flat<float>().data(),
                d_feat_tensor.flat<float>().data(),
                d_idxs_tensor.flat<int>().data(),
                out_feat_grad->flat<float>().data(),
                out_weight_grad->flat<float>().data(),
                n_vert,
                n_neigh,
                n_feat
        );

    }

};

REGISTER_KERNEL_BUILDER(Name("PushKnnGrad").Device(DEVICE_CPU), PushKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct PushKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("PushKnnGrad").Device(DEVICE_GPU), PushKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
