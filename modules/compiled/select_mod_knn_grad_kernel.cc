
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_mod_knn_grad_kernel.h"
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
struct SelectModKnnGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_grad_dist,
            const int *d_indices,
            const float *d_dist,
            const float *d_coord,
            const float *d_coord_mod,

            float * d_grad_coord,
            float * d_grad_coord_mod,

            const int n_vert,
            const int n_neigh,
            const int n_coords) {

        throw std::runtime_error("SelectModKnnGradOpFunctor: no CPU implementation available");

    }
};

template<typename Device>
class SelectModKnnGradOp : public OpKernel {
public:
    explicit SelectModKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {


    }

    void Compute(OpKernelContext *context) override {


        const Tensor &t_grad_dist = context->input(0);
        const Tensor &t_indices = context->input(1);
        const Tensor &t_distances = context->input(2);
        const Tensor &t_coord = context->input(3);
        const Tensor &t_coord_mod = context->input(4);


        int n_vert = t_coord.dim_size(0);
        int n_coords = t_coord.dim_size(1);
        int n_neigh = t_distances.dim_size(1);

        auto coorddimsok = t_coord_mod.dims() == 3
                && t_coord_mod.dim_size(0) == n_vert
                && t_coord_mod.dim_size(1) == n_coords
                && t_coord_mod.dim_size(2) == n_coords
                ? tensorflow::Status(): tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                        "Coordinate modifier tensor needs to have 3 dimensions (V x C x C)");
        OP_REQUIRES_OK(context,coorddimsok);


        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_coords);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        outputShape.AddDim(n_coords);

        Tensor *output_mod_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_mod_tensor));


        SelectModKnnGradOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_grad_dist.flat<float>().data(),
                t_indices.flat<int>().data(),
                t_distances.flat<float>().data(),
                t_coord.flat<float>().data(),
                t_coord_mod.flat<float>().data(),

                output_tensor->flat<float>().data(),
                output_mod_tensor->flat<float>().data(),

                n_vert,
                n_neigh,
                n_coords
        );



    }

private:

};

REGISTER_KERNEL_BUILDER(Name("SelectModKnnGrad").Device(DEVICE_CPU), SelectModKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SelectModKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SelectModKnnGrad").Device(DEVICE_GPU), SelectModKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
