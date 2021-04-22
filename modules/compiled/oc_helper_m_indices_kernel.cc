
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "oc_helper_m_indices_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

//helpers here


template<typename dummy>
struct MIndicesMaxUqOpFunctor<CPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,
            const int *d_maxunique,
            int * n_max_per_unique
            ){
        *n_max_per_unique=d_maxunique[0];
    }
};

// CPU specialization
template<typename dummy>
struct MIndicesOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const int *d_truthidx,
            const int *d_unique_idx,

            int * out_idx,
            float * m_not,

            const int n_vert,
            const int n_unique,
            const int n_max_per_unique,
            bool calc_m_not) {

        //main axis: n_unique == K_obj
        //no other parallisation possible

        for (int k = 0; k < n_unique; k++) {

            //

            int uqidx = d_unique_idx[k];
            int puqcounter=0;
            for(int i_v = 0; i_v < n_vert; i_v++ ){
                if(uqidx>=0 && d_truthidx[i_v] == uqidx){
                    out_idx[I2D(k, puqcounter, n_max_per_unique)] = i_v;
                    puqcounter++;
                }
            }
            for(int prem = puqcounter; prem < n_max_per_unique; prem++){
                out_idx[I2D(k, prem, n_max_per_unique)] = -1;
            }

            //
            if(calc_m_not ){
                //m_not
                for(int i_v = 0; i_v < n_vert; i_v++ ){
                    if(uqidx>=0 && d_truthidx[i_v] == uqidx)
                        m_not [I2D(k, i_v, n_vert)] = 0.;
                    else
                        m_not [I2D(k, i_v, n_vert)] = 1.;
                }
            }

        }

    }

};

template<typename Device>
class MIndicesOp : public OpKernel {
public:
    explicit MIndicesOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                context->GetAttr("calc_m_not", &calc_m_not_));
    }


    void Compute(OpKernelContext *context) override {

        /*

         */

        const Tensor &t_tidx = context->input(0);
        const Tensor &t_uqtixs = context->input(1);
        const Tensor &t_nmax_puq = context->input(2);//needs to be evaluated in kernel

        const int n_vert = t_tidx.dim_size(0);
        const int n_unique = t_uqtixs.dim_size(0);
        int n_max_per_unique=0;

        MIndicesMaxUqOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                t_nmax_puq.flat<int>().data(),
                &n_max_per_unique
        );

        Tensor *out_idx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                {n_unique, n_max_per_unique}, &out_idx));

        Tensor *m_not = NULL;
        TensorShape m_not_shape={1, 1};
        if(calc_m_not_)
            m_not_shape={n_unique, n_vert};

        OP_REQUIRES_OK(context, context->allocate_output(1,
                m_not_shape, &m_not));

        MIndicesOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                t_tidx.flat<int>().data(),
                t_uqtixs.flat<int>().data(),

                out_idx->flat<int>().data(),
                m_not->flat<float>().data(),

                n_vert,
                n_unique,
                n_max_per_unique,
                calc_m_not_

        );



    }
private:
    bool calc_m_not_;

};

REGISTER_KERNEL_BUILDER(Name("MIndices").Device(DEVICE_CPU), MIndicesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct MIndicesMaxUqOpFunctor<GPUDevice, int>;
extern template struct MIndicesOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("MIndices").Device(DEVICE_GPU), MIndicesOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
