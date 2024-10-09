
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
            const int *rs,

            int * out_idx,
            int * m_not,

            const int n_vert,
            const int n_unique,
            const int n_max_per_unique,
            const int n_max_in_rs,

            const int n_rs,
            bool calc_m_not) {

        //main axis: n_unique == K_obj
        //no other parallisation possible

        for (int k = 0; k < n_unique; k++) {

            

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
            //m_not
            if(calc_m_not){
                //find row for uqidx
                int rowForUqidx = 0;
                while(rowForUqidx+1<n_rs && uqidx >= rs[rowForUqidx+1]){
                    rowForUqidx++;
                }

                
                int mnot_index = 0;
                for(int i_v = 0; i_v < n_vert; i_v++ ){
                    //find row for i_v
                    int rowFori_v= 0;
                    while(rowFori_v+1<n_rs && i_v >= rs[rowFori_v+1]){
                        rowFori_v++;
                    }

                    //compare rows and index
                    if (rowFori_v == rowForUqidx && (uqidx < 0 || d_truthidx[i_v] != uqidx)){
                        m_not [I2D(k, mnot_index, n_max_in_rs)] = i_v;
                        mnot_index++;
                    }
                }
                //fill rest with -1
                while(mnot_index < n_max_in_rs){
                    m_not [I2D(k, mnot_index, n_max_in_rs)] = -1;
                    mnot_index++;
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
        const Tensor &t_rs = context->input(3);
        const Tensor &t_max_in_rs = context->input(4);

        const int n_vert = t_tidx.dim_size(0);
        const int n_unique = t_uqtixs.dim_size(0);
        const int n_rs = t_rs.dim_size(0);

        int n_max_per_unique=0;
        MIndicesMaxUqOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                t_nmax_puq.flat<int>().data(),
                &n_max_per_unique
        );

        int n_max_in_rs = 0;
        MIndicesMaxUqOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                t_max_in_rs.flat<int>().data(),
                &n_max_in_rs
        );

        Tensor *out_idx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                {n_unique, n_max_per_unique}, &out_idx));

        Tensor *m_not = NULL;
        TensorShape m_not_shape={1, 1};
        if(calc_m_not_)
            m_not_shape={n_unique, n_max_in_rs};
        OP_REQUIRES_OK(context, context->allocate_output(1,
                m_not_shape, &m_not));

        MIndicesOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                t_tidx.flat<int>().data(),
                t_uqtixs.flat<int>().data(),
                t_rs.flat<int>().data(),

                out_idx->flat<int>().data(),
                m_not->flat<int>().data(),

                n_vert,
                n_unique,
                n_max_per_unique,
                n_rs,
                n_max_in_rs,
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
