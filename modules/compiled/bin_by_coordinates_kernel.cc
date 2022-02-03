

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
            const CPUDevice &d,

            const float * d_coords,
            const int * d_rs,

            const float * d_binswidth, //singleton
            const int* n_bins,//singleton

            int * d_assigned_bin,

            int n_vert,
            int n_coords,
            int n_rs){

        int nbins = n_bins[0];

        //this will be parallelisation dimension
        for(int iv=0; iv<n_vert; iv++){

            int mul = 1;
            int idx = 0;
            for (int ic = n_coords-1; ic != -1; ic--) {

                int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[ic];

                idx += cidx * mul;
                mul *= nbins;

            }

            //get row split index last
            int rsidx=0;
            for(int irs=1 ; irs < n_rs ; irs++){
                if(d_rs[irs] > iv){
                    break;
                }
                rsidx++;
            }

            idx += rsidx * mul;

            d_assigned_bin[iv]=idx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]

        }//iv loop

    }
};



template<typename Device>
class BinByCoordinatesOp : public OpKernel {
public:
    explicit BinByCoordinatesOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }


    void Compute(OpKernelContext *context) override {


        const Tensor &t_coords = context->input(0);
        const Tensor &t_rs = context->input(1);
        const Tensor &t_binwdith = context->input(2);
        const Tensor &t_nbins = context->input(3);

        int n_vert = t_coords.dim_size(0);
        int n_coords = t_coords.dim_size(1);
        int n_rs = t_rs.dim_size(0);

        ///size checks

        OP_REQUIRES(context, n_coords == t_binwdith.dim_size(0),
                    errors::InvalidArgument("BinByCoordinatesOp expects coordinate dimensions for bin width."));
        OP_REQUIRES(context, 1 == t_nbins.dim_size(0),
                    errors::InvalidArgument("BinByCoordinatesOp expects singleton for number of bins."));

        
        Tensor *t_assigned_bin = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,{n_vert},&t_assigned_bin));

        BinByCoordinatesOpFunctor<Device, int>() (
                context->eigen_device<Device>(),

                t_coords.flat<float>().data(),
                t_rs.flat<int>().data(),

                t_binwdith.flat<float>().data(), //singleton
                t_nbins.flat<int>().data(),

                t_assigned_bin->flat<int>().data() ,

                n_vert,
                n_coords,
                n_rs
        );



    }


};

REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_CPU), BinByCoordinatesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BinByCoordinatesOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_GPU), BinByCoordinatesOp<GPUDevice>);
#endif  

}//functor
}//tensorflow
