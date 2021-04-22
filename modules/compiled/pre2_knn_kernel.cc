
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "pre2_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template<typename dummy>
struct Pre2KnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coords, // accessible only on GPU!!!
            float *d_coords_sorted,
            int *auxaliry_knn_arrays,
            const int V, // # of vertices
            const int n_coords,
            const int n_bins_x,
            const int n_bins_y) {

        // EMPTY

    }
};

template<typename Device>
class Pre2KnnOp : public OpKernel {
public:
    explicit Pre2KnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_bins_x", &n_bins_x_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_bins_y", &n_bins_y_));

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);

        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_coords);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        TensorShape outputShape2;
        outputShape2.AddDim(1+1+(n_bins_x_*n_bins_y_+1)+(9*n_bins_x_*n_bins_y_)+n_vert);
        outputShape2.AddDim(1);
        Tensor *output_tensor2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape2, &output_tensor2));

        Pre2KnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                output_tensor->flat<float>().data(),
                output_tensor2->flat<int>().data(),

                n_vert,
                n_coords,
                n_bins_x_,
                n_bins_y_
        );


    }

private:
    int n_bins_x_;
    int n_bins_y_;
};

REGISTER_KERNEL_BUILDER(Name("Pre2Knn").Device(DEVICE_CPU), Pre2KnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct Pre2KnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("Pre2Knn").Device(DEVICE_GPU), Pre2KnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
