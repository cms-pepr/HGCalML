

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "index_replacer_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template<typename dtype>
struct IndexReplacerOpFunctor<CPUDevice, dtype> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,
            const dtype * to_be_replaced,
            const dtype * replacements,
            dtype * replaced,

            const int n_to_be_replaced,
            const int n_replacements
            ){

        for(int i=0;i<n_to_be_replaced;i++){
            const int ridx = to_be_replaced[i];
            if(ridx<0){
                replaced[i] = ridx;
                continue;
            }
            if(ridx>=n_replacements){
                printf("IndexReplacerOpFunctor: index out of range\n");
                continue;
            }
            replaced[i] = replacements[ridx];
        }
    }
};



template<typename Device>
class IndexReplacerOp : public OpKernel {
public:
    explicit IndexReplacerOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }


    void Compute(OpKernelContext *context) override {

        const Tensor &t_to_be_replaced = context->input(0);
        const Tensor &t_replacements = context->input(1);

        const int n_to_be_replaced = t_to_be_replaced.NumElements();
        const int n_replacements = t_replacements.dim_size(0);
        
        Tensor *out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                t_to_be_replaced.shape(), &out));//same shape

        IndexReplacerOpFunctor<Device, int32>() (
                context->eigen_device<Device>(),

                t_to_be_replaced.flat<int32>().data(),
                t_replacements.flat<int32>().data(),
                out->flat<int32>().data(),
                n_to_be_replaced,
                n_replacements
        );



    }

};

REGISTER_KERNEL_BUILDER(Name("IndexReplacer").Device(DEVICE_CPU), IndexReplacerOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct IndexReplacerOpFunctor<GPUDevice, int32>;
REGISTER_KERNEL_BUILDER(Name("IndexReplacer").Device(DEVICE_GPU), IndexReplacerOp<GPUDevice>);
#endif  

}//functor
}//tensorflow
