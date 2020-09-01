
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_threshold_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


/*
 * copy th_values gpu->cpu
 * sequential 'push-back' index mask
 *
 *
 *
 */

// CPU specialization
template<typename dummy>
struct SelectThresholdOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_th_values,
            const int* d_row_splits,

            int *d_scatter_idxs,
            int *d_new_rowsplits,

            int * n_scatter_idxs,

            const int n_vert,

            const int n_rs,

            const float threshold) {

        *n_scatter_idxs = 0;

        d_new_rowsplits[0]=0;

        //same in "gpu" implementation, copies to cpu
        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){



            size_t startvert = d_row_splits[j_rs];
            size_t endvert = d_row_splits[j_rs+1];

            size_t n_over_th=0;
            size_t largest_below_thresh=0;
            float largest_below_th_f=-1000;
            for(size_t i_v=startvert;i_v<endvert ;i_v++){


                float th_val = d_th_values[i_v];

                if(th_val >= threshold){


                    d_scatter_idxs[*n_scatter_idxs] = i_v;
                    *n_scatter_idxs = *n_scatter_idxs + 1;
                    n_over_th++;
                }
                else if(!n_over_th){
                    if(largest_below_th_f < th_val){
                        largest_below_th_f=th_val;
                        largest_below_thresh = i_v;
                    }
                }

            }
            //if none of them is above threshold, select the highest value one to keep data structure intact
            if(!n_over_th){
                d_scatter_idxs[*n_scatter_idxs] = largest_below_thresh; //just put first one to avoid empty vector
                *n_scatter_idxs = *n_scatter_idxs + 1;
            }
            d_new_rowsplits[j_rs+1] = *n_scatter_idxs;
        }
        //put last n_nsidxs as las new rs
    }
};

template<typename dummy>
struct CopyOutputSelectThresholdOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            int *d_scatter_idxs,

            int *d_tmp_scatter_idxs,

            int  n_scatter_idxs) {

        for(size_t i=0;i<n_scatter_idxs;i++)
            d_scatter_idxs[i] = d_tmp_scatter_idxs[i];
    }
};

template<typename Device>
class SelectThresholdOp : public OpKernel {
public:
    explicit SelectThresholdOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("threshold", &threshold_));

    }

    void Compute(OpKernelContext *context) override {

        /*
         *   .Attr("threshold: float")
         *
    .Input("th_value: float32")
    .Input("rowsplits: int32")

    .Output("scatter_idxs: int32")
    .Output("new_rowsplits: int32");

         *
         */

        const Tensor &t_th_value = context->input(0);
        const Tensor &t_rowsplits = context->input(1);


        int n_vert = t_th_value.dim_size(0);
        int n_rs = t_rowsplits.dim_size(0);


        TensorShape tempshape;
        tempshape.AddDim(n_vert);
        Tensor *temp_output_idxs = new Tensor();
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,tempshape,temp_output_idxs));

        //row split dimensions don't change
        Tensor *output_rs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output( 1, t_rowsplits.shape() ,&output_rs));

        int new_nidx=0;
        //context->allocate_temp()//full size
        //work with temp, return the sizes,


        SelectThresholdOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_th_value.flat<float>().data(), // const float *d_th_values,
                t_rowsplits.flat<int>().data(), //const int* d_row_splits,

                temp_output_idxs->flat<int>().data(), //int *d_scatter_idxs,
                output_rs->flat<int>().data(), //int *d_new_rowsplits,

                &new_nidx, //int * n_scatter_idxs,

                n_vert,
                n_rs,
                threshold_
        );



        //use sizes to allocate actual output, run a simple copy kernel
        TensorShape outshape_idxs;
        outshape_idxs.AddDim(new_nidx);
        outshape_idxs.AddDim(1); //gather_nd compatible
        Tensor *output_idxs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output( 0, outshape_idxs,&output_idxs));



        CopyOutputSelectThresholdOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                output_idxs->flat<int>().data() , //int *d_scatter_idxs,

                temp_output_idxs->flat<int>().data(), //int *d_scatter_idxs,

                new_nidx
       );

        //free temp resources (auto TF)??

        delete temp_output_idxs;
    }

private:
    float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("SelectThreshold").Device(DEVICE_CPU), SelectThresholdOp<CPUDevice>);

//#ifdef GOOGLE_CUDA
//extern template struct SelectThresholdOpFunctor<GPUDevice, int>;
//REGISTER_KERNEL_BUILDER(Name("SelectThreshold").Device(DEVICE_GPU), SelectThresholdOp<GPUDevice>);
//#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
