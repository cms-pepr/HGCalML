
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "build_condensates_grad_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

//helpers here



// CPU specialization
template<typename dummy>
struct BuildCondensatesGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *sum_features_grad,
            const int *asso_idx,
            float *features_grad,

            const int n_vert,
            const int n_feat) {

        for(size_t i_v=0;i_v<n_vert;i_v++){

            int asso = asso_idx[i_v];

            for(size_t i_f=0;i_f<n_feat;i_f++){
                if(asso<0)
                    features_grad[I2D(i_v,i_f,n_feat)] = 0;
                else
                    features_grad[I2D(i_v,i_f,n_feat)] = sum_features_grad[I2D(asso,i_f,n_feat)];
            }
        }


    }
};

template<typename Device>
class BuildCondensatesGradOp : public OpKernel {
public:
    explicit BuildCondensatesGradOp(OpKernelConstruction *context) : OpKernel(context) {

    }


    void Compute(OpKernelContext *context) override {

        /*
         *
         * .Attr("radius: float")
    .Attr("min_beta: float")
    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("beta_sorting: int32")
    .Input("features: float32")
    .Input("row_splits: int32")
    .Output("summed_features: float32")
    .Output("asso_idx: int32"

    const float *sum_features_grad,
            const int *asso_idx,
            float *features_grad,

            const int n_vert,
            const int n_feat
         */

        const Tensor &t_sum_features_grad = context->input(0);
        const Tensor &t_asso_idx = context->input(1);

        const int n_vert = t_sum_features_grad.dim_size(0);
        const int n_feat = t_sum_features_grad.dim_size(1);


        TensorShape outputShape_feat;
        outputShape_feat.AddDim(n_vert);
        outputShape_feat.AddDim(n_feat);
       // outputShape.AddDim(K_);

        Tensor *output_features_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_feat, &output_features_grad));


        BuildCondensatesGradOpFunctor<Device, int>()
                (
                context->eigen_device<Device>(),                                //
                t_sum_features_grad.flat<float>().data(),
                t_asso_idx.flat<int>().data(),
                output_features_grad->flat<float>().data(),
                n_vert,
                n_feat//            const float min_beta                 / const float min_beta
        );



    }

};

REGISTER_KERNEL_BUILDER(Name("BuildCondensatesGrad").Device(DEVICE_CPU), BuildCondensatesGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BuildCondensatesGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BuildCondensatesGrad").Device(DEVICE_GPU), BuildCondensatesGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
