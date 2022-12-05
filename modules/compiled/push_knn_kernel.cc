
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "push_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

static void set_zero(
        float * tensor,
        size_t n_vert,
        size_t n_feat
){

    for (size_t i_v = 0; i_v < n_vert; i_v++) {
        for(size_t i_f=0;i_f<n_feat;i_f++){
            tensor[I2D(i_v, i_f, n_feat)] = 0;
        }
    }

}

// CPU specialization
template<typename dummy>
struct PushKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_weights,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,

            int n_vert,
            int n_neigh,
            int n_feat) {

        printf("calling push knn cpu kernel for %d vertices.\n", n_vert);

        set_zero(d_out_feat,n_vert, n_feat);

        //simple loop here, no need to use atomics
        for (size_t i_v = 0; i_v < n_vert; i_v++) {

            for(size_t i_f=0;i_f<n_feat;i_f++){

                float f = d_feat[I2D(i_v,i_f,n_feat)];

                for(size_t i_n=0;i_n<n_neigh;i_n++){
                    int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];

                    if(nidx<0) continue;

                    float w = d_weights[I2D(i_v,i_n,n_neigh)];

                    d_out_feat[I2D(nidx,i_f,n_feat)] += f*w;
                }
            }
        }
    }
};

template<typename Device>
class PushKnnOp : public OpKernel {
public:
    explicit PushKnnOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_weight_tensor = context->input(0);
        const Tensor &d_feat_tensor = context->input(1);
        const Tensor &d_idxs_tensor = context->input(2);


        int n_vert = d_weight_tensor.dim_size(0);
        int n_neigh = d_idxs_tensor.dim_size(1);
        int n_feat = d_feat_tensor.dim_size(1);

        OP_REQUIRES(context, n_vert == d_idxs_tensor.dim_size(0) && n_vert == d_feat_tensor.dim_size(0),
                    errors::InvalidArgument("PushKnnOp expects first dimensions of all inputs to match."));

        OP_REQUIRES(context, n_neigh == d_weight_tensor.dim_size(1),
                    errors::InvalidArgument("PushKnnOp expects second dimension of distance and neighbour index tensor to match"));


        // after testing basic functionality!
        // n_out_feat += n_moments * n_feat * n_coords;


        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_feat);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));


        PushKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                d_weight_tensor.flat<float>().data(),
                d_feat_tensor.flat<float>().data(),
                d_idxs_tensor.flat<int>().data(),
                output_tensor->flat<float>().data(),
                n_vert,
                n_neigh,
                n_feat
        );

    }

};

REGISTER_KERNEL_BUILDER(Name("PushKnn").Device(DEVICE_CPU), PushKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct PushKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("PushKnn").Device(DEVICE_GPU), PushKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
