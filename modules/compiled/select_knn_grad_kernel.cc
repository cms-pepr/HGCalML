
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_knn_grad_kernel.h"
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
struct SelectKnnGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,

            float * d_grad_coord,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {



    }
};

template<typename Device>
class SelectKnnGradOp : public OpKernel {
public:
    explicit SelectKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {


    }

    void Compute(OpKernelContext *context) override {


        const Tensor &t_grad_dist = context->input(0);
        const Tensor &t_indices = context->input(1);
        const Tensor &t_distances = context->input(2);
        const Tensor &t_coord = context->input(3);


        int n_vert = t_coord.dim_size(0);
        int n_coords = t_coord.dim_size(1);
        int n_neigh = t_distances.dim_size(1);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_coords);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));


        SelectKnnGradOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_grad_dist.flat<float>().data(),
                t_indices.flat<int>().data(),
                t_distances.flat<float>().data(),
                t_coord.flat<float>().data(),

                output_tensor->flat<float>().data(),

                n_vert,
                n_neigh,
                n_coords
        );



    }

private:

};

REGISTER_KERNEL_BUILDER(Name("SelectKnnGrad").Device(DEVICE_CPU), SelectKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SelectKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SelectKnnGrad").Device(DEVICE_GPU), SelectKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
