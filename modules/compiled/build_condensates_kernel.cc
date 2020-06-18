
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "build_condensates_kernel.h"
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
struct BuildCondensatesOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_ccoords,
            const float *d_betas,
            const int *beta_sorting,
            const float *features,
            const int *row_splits,


            float *summed_features,
            int *asso_idx,

            const int n_vert,
            const int n_feat,
            const int n_ccoords,

            const int n_rs,

            const float radius,
            const float min_beta) {



        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            //do something
        }
    }
};

template<typename Device>
class BuildCondensatesOp : public OpKernel {
public:
    explicit BuildCondensatesOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                context->GetAttr("min_beta", &min_beta_));
        OP_REQUIRES_OK(context,
                context->GetAttr("radius", &radius_));
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
         */

        const Tensor &t_ccoords = context->input(0);
        const Tensor &t_betas = context->input(1);
        const Tensor &t_beta_sorting = context->input(2);
        const Tensor &t_features = context->input(3);
        const Tensor &t_row_splits = context->input(4);




        int n_vert = t_ccoords.dim_size(0);
        int n_ccoords = t_ccoords.dim_size(1);

        int n_feat = t_features.dim_size(1);

        int n_rs = t_row_splits.dim_size(0);



        TensorShape outputShape_feat;
        outputShape_feat.AddDim(n_vert);
        outputShape_feat.AddDim(n_feat);
       // outputShape.AddDim(K_);

        Tensor *output_features = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_feat, &output_features));

        TensorShape outputShape_idx;
        outputShape_feat.AddDim(n_vert);

        Tensor *output_indices = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape_idx, &output_indices));


        BuildCondensatesOpFunctor<Device, int>()
                (
                context->eigen_device<Device>(),                                //
                //                                                                                                     /
                t_ccoords.flat<float>().data(),                     //            const float *d_ccoords,              / const float *d_ccoords,
                t_betas.flat<float>().data(),                       //            const float *d_betas,                / const float *d_betas,
                t_beta_sorting.flat<int>().data(),                  //            const float *beta_sorting,           / const float *beta_sorting,
                t_features.flat<float>().data(),                    //            const float *features,               / const float *features,
                t_row_splits.flat<int>().data(),                    //            const int *row_splits,               / const int *row_splits,
                //                                                                                                     /
                //                                                                                                     /
                output_features->flat<float>().data(),              //            float *summed_features,              / float *summed_features,
                output_indices->flat<int>().data(),                 //            int *asso_idx,                       / int *asso_idx,
                //                                                                                                     /
                n_vert,                                             //            const int n_vert,                    / const int n_vert,
                n_feat,                                             //            const int n_feat,                    / const int n_feat,
                n_ccoords,                                          //            const int n_ccoords,                 / const int n_ccoords,
                //                                                                                                     /
                n_rs,                                               //            const int n_rs,                      / const int n_rs,
                //                                                                                                     /
                radius_,                                            //            const float radius,                  / const float radius,
                min_beta_                                           //            const float min_beta                 / const float min_beta
        );



    }

private:
    float min_beta_;
    float radius_;
};

REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_CPU), BuildCondensatesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BuildCondensatesOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_GPU), BuildCondensatesOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
