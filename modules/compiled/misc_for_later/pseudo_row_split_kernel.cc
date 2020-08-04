
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "pseudo_row_split_kernel.h"
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
struct PseudoRowSplitOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const int *in_asso_idx,
            const int *d_n_per_rs,
            const int *row_splits,

            int *asso_idx,
            int *pseudo_rs,

            const int n_vert,
            const int n_pseudo_rs,
            const int n_rs
    ) {

        // missing info here: which index goes where..





    }
};

template<typename dummy>
struct PseudoRowSplitCountOpFunctor<CPUDevice, dummy> {
    void operator()(
            const CPUDevice &d,
            const int *d_n_per_rs,
            int& ntotal,
            const int n_prs
            ){

        //consider just doing this on CPU
        ntotal=0;
        //includes a '+1'
        for(int i=0;i<n_prs;i++){
            ntotal+=d_n_per_rs[i];
        }

    }
};

template<typename Device>
class PseudoRowSplitOp : public OpKernel {
public:
    explicit PseudoRowSplitOp(OpKernelConstruction *context) : OpKernel(context) {

    }


    void Compute(OpKernelContext *context) override {

        /*
         *

    .Input("asso_idx: int32")
    .Input("n_per_rs: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("pseudo_rs: int32")



         */

        const Tensor &t_in_asso_idx = context->input(0);
        const Tensor &t_unique_assos = context->input(1);
        const Tensor &t_n_per_rs = context->input(2);
        const Tensor &t_row_splits = context->input(3);

        const int n_vert = t_in_asso_idx.dim_size(0);
        const int n_rs = t_row_splits.dim_size(0);
        const int n_unique = t_unique_assos.dim_size(0);


        TensorShape outputShape_idx;
        outputShape_idx.AddDim(n_vert);

        Tensor *t_asso_idx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_idx, &t_asso_idx));

        int npsrtotal=0;
        PseudoRowSplitCountOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                t_n_per_rs.flat<int>().data(),
                npsrtotal,
                n_rs - 1);

        TensorShape outputShape_psrs;
        outputShape_psrs.AddDim(npsrtotal+1);

        Tensor *t_pseudo_rs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_psrs, &t_pseudo_rs));


        PseudoRowSplitOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                t_in_asso_idx.flat<int>().data(),           // const int *d_asso_idx,
                t_n_per_rs.flat<int>().data(),            // const int *d_n_per_rs,
                t_row_splits.flat<int>().data(),           // const int *row_splits,
                           //
                t_asso_idx->flat<int>().data(),            // int *asso_idx,
                t_pseudo_rs->flat<int>().data(),           // int *pseudo_rs,
                           //
                n_vert,           // const int n_vert,
                npsrtotal+1,
                n_rs           // const int n_rs
        );



    }

private:

};

REGISTER_KERNEL_BUILDER(Name("PseudoRowSplit").Device(DEVICE_CPU), PseudoRowSplitOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct PseudoRowSplitOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("PseudoRowSplit").Device(DEVICE_GPU), PseudoRowSplitOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
