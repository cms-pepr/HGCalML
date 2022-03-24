
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


static inline float distanceWeight(const float& distsq){
    return distsq;
}

// CPU specialization
template<typename dummy>
struct AccumulateKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_distances,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,
            int *d_out_maxidxs,

            int n_vert,
            int n_neigh,
            int n_feat,

            int n_out_feat,

            int n_moments,
            bool mean_and_max) {


        for (size_t i_v = 0; i_v < n_vert; i_v++) {

            for(size_t i_f=0;i_f<n_feat;i_f++){
                float t_mean = 0;
                float t_max = 0;
                int max_i_n_gidx = 0;

                for(size_t i_n=0;i_n<n_neigh;i_n++){
                    int nidx = d_idxs[I2D(i_v,i_n,n_neigh)];

                    if(nidx<0) continue;

                    float vnf = d_feat[I2D(nidx,i_f,n_feat)];
                    float distsq = d_distances[I2D(i_v,i_n,n_neigh)];
                    float wfeat = vnf * distanceWeight(distsq);
                    //DEBUGCOUT(wfeat);
                    t_mean += wfeat;
                    if(mean_and_max && (wfeat >= t_max || !i_n)){
                        max_i_n_gidx = nidx;
                        t_max = wfeat;
                    }
                }
                t_mean /= (float)n_neigh;

                d_out_feat[I2D(i_v,i_f,n_out_feat)] = t_mean;
                if(mean_and_max){
                    d_out_maxidxs[I2D(i_v,i_f,n_feat)] = max_i_n_gidx; //just used for gradient
                    d_out_feat[I2D(i_v,i_f+n_feat,n_out_feat)] = t_max;
                }
                //moments in n_coords x n_neigh loop here {}

            }

        }
    }
};

template<typename Device>
class AccumulateKnnOp : public OpKernel {
public:
    explicit AccumulateKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                context->GetAttr("n_moments", &n_moments));
        OP_REQUIRES_OK(context,
                context->GetAttr("mean_and_max", &mean_and_max));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_dist_tensor = context->input(0);
        const Tensor &d_feat_tensor = context->input(1);
        const Tensor &d_idxs_tensor = context->input(2);


        int n_vert = d_dist_tensor.dim_size(0);
        int n_neigh = d_idxs_tensor.dim_size(1);
        int n_coords = d_dist_tensor.dim_size(1);
        int n_feat = d_feat_tensor.dim_size(1);

        OP_REQUIRES(context, n_vert == d_idxs_tensor.dim_size(0) && n_vert == d_feat_tensor.dim_size(0),
                    errors::InvalidArgument("AccumulateKnnOp expects first dimensions of all inputs to match."));

        OP_REQUIRES(context, n_neigh == d_dist_tensor.dim_size(1),
                    errors::InvalidArgument("AccumulateKnnOp expects second dimension of distance and neighbour index tensor to match"));

        int n_out_feat = n_feat; //mean and max
        if(mean_and_max)
            n_out_feat*=2;

        // after testing basic functionality!
        // n_out_feat += n_moments * n_feat * n_coords;


        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_out_feat);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        TensorShape outputShape_max_idxs;
        outputShape_max_idxs.AddDim(n_vert);
        outputShape_max_idxs.AddDim(n_feat);

        Tensor *output_max_idxs_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape_max_idxs, &output_max_idxs_tensor));


        AccumulateKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                d_dist_tensor.flat<float>().data(),
                d_feat_tensor.flat<float>().data(),
                d_idxs_tensor.flat<int>().data(),
                output_tensor->flat<float>().data(),
                output_max_idxs_tensor->flat<int>().data(),
                n_vert,
                n_neigh,
                n_feat,
                n_out_feat,
                n_moments,
                mean_and_max
        );



    }

private:
    int n_moments;
    bool mean_and_max;
};

REGISTER_KERNEL_BUILDER(Name("AccumulateKnn").Device(DEVICE_CPU), AccumulateKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnn").Device(DEVICE_GPU), AccumulateKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
