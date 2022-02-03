

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "bin_by_coordinates_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template<typename dummy>
struct BinByCoordinatesOpFunctor<CPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d
            ){
            //CPU implementation
    }
};



template<typename Device>
class BinByCoordinatesOp : public OpKernel {
public:
    explicit BinByCoordinatesOp(OpKernelConstruction *context) : OpKernel(context) {
    
        //replace with actual configuration attributes
        
        OP_REQUIRES_OK(context,
                context->GetAttr("attr", &attr_));
    }


    void Compute(OpKernelContext *context) override {


        const Tensor &t_in = context->input(0);
        int dim0 = t_in.dim_size(0);
        
        Tensor *out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,
                {dim0, 1}, //shape
                &out));

        BinByCoordinatesOpFunctor<Device, int>() (
                context->eigen_device<Device>()

        );



    }
private:
    float attr_;

};

REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_CPU), BinByCoordinatesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BinByCoordinatesOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_GPU), BinByCoordinatesOp<GPUDevice>);
#endif  

}//functor
}//tensorflow
