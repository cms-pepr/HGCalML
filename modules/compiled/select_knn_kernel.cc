
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_knn_kernel.h"
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
struct SelectKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coord,
            int *d_indices,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {



    }
};

template<typename Device>
class SelectKnnOp : public OpKernel {
public:
    explicit SelectKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_neighbours", &K_));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_feat_tensor = context->input(1);
        const Tensor &d_idxs_tensor = context->input(2);


        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));



        SelectKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                d_coord_tensor.flat<float>().data(),
                output_tensor->flat<int>().data(),
                n_vert,
                K_,
                n_coords
        );



    }

private:
    int K_;
};

REGISTER_KERNEL_BUILDER(Name("SelectKnn").Device(DEVICE_CPU), SelectKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SelectKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SelectKnn").Device(DEVICE_GPU), SelectKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
