

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "rs_offset_adder_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template<typename dtype>
struct RSOffsetAdderOpFunctor<CPUDevice, dtype> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,
            const int * t_dx,
            const int * rs,
            int * new_t_idx,

            const int n_vert,
            const int n_rs
            ){

    }
};



template<typename Device>
class RSOffsetAdderOp : public OpKernel {
public:
    explicit RSOffsetAdderOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }


    void Compute(OpKernelContext *context) override {

        const Tensor &t_t_idx = context->input(0);
        const Tensor &t_rs = context->input(1);

        const int n_vert = t_t_idx.dim_size(0);
        const int n_rs = t_rs.dim_size(0);
        
        Tensor *out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                t_t_idx.shape(), &out));

        RSOffsetAdderOpFunctor<Device, int32>() (
                context->eigen_device<Device>(),

                t_t_idx.flat<int32>().data(),
                t_rs.flat<int32>().data(),
                out->flat<int32>().data(),
                n_vert,
                n_rs
        );



    }

};

REGISTER_KERNEL_BUILDER(Name("RSOffsetAdder").Device(DEVICE_CPU), RSOffsetAdderOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct RSOffsetAdderOpFunctor<GPUDevice, int32>;
REGISTER_KERNEL_BUILDER(Name("RSOffsetAdder").Device(DEVICE_GPU), RSOffsetAdderOp<GPUDevice>);
#endif  

}//functor
}//tensorflow
