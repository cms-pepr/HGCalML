
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_nd_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


static float distanceWeight(float distsq){
    if(!distsq)return 1;
    return exp(-1.*ACCUMULATE_KNN_EXPONENT* distsq);
}

// CPU specialization
template<typename dummy>
struct AccumulateKnnNdOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coord,
            const float *d_feat,
            const int *d_idxs,

            float *d_out_feat,
            int *d_out_maxidxs,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,

            const int n_out_feat,

            const int n_moments) {

        for (size_t i_v = 0; i_v < n_vert; i_v++) {

            for(size_t i_f=0;i_f<n_feat;i_f++){

                for(size_t i_c=0;i_c<n_coords;i_c++){

                    int max_i_n_gidx = 0;
                    float t_max = -1e3;
                    float t_mean = 0;

                    float vic = d_coord[I2D(i_v,i_c,n_coords)]; //buffer?

                    for(size_t i_n=0;i_n<n_neigh;i_n++){
                        size_t nidx = d_idxs[I2D(i_v,i_n,n_neigh)];

                        float vnf = d_feat[I2D(nidx,i_f,n_feat)];
                        float vnc = d_coord[I2D(nidx,i_c,n_coords)];
                        float distsq = (vic-vnc)*(vic-vnc);

                        float wfeat = vnf * distanceWeight(distsq);
                        t_mean += wfeat;
                        if(wfeat > t_max){
                            max_i_n_gidx = nidx;
                            t_max = wfeat;
                        }
                    }

                    t_mean /= (float)n_neigh;

                    d_out_maxidxs[I3D(i_v,i_f,i_c,n_feat,n_coords)] = max_i_n_gidx; //just used for gradient
                    d_out_feat[I3D(i_v,i_f,i_c,n_out_feat,n_coords)] = t_mean;
                    d_out_feat[I3D(i_v,i_f+n_feat,i_c,n_out_feat,n_coords)] = t_max;

                    //moments in n_coords x n_neigh loop here {}
                }
            }

        }
    }
};

template<typename Device>
class AccumulateKnnNdOp : public OpKernel {
public:
    explicit AccumulateKnnNdOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                context->GetAttr("n_moments", &n_moments));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_feat_tensor = context->input(1);
        const Tensor &d_idxs_tensor = context->input(2);


        int n_vert = d_coord_tensor.dim_size(0);
        int n_neigh = d_idxs_tensor.dim_size(1);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_feat = d_feat_tensor.dim_size(1);


        int n_out_feat = 2 * n_feat; //mean and max

        // after testing basic functionality!
        // n_out_feat += n_moments * n_feat * n_coords;


        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(n_out_feat);
        outputShape.AddDim(n_coords);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        TensorShape outputShape_max_idxs;
        outputShape_max_idxs.AddDim(n_vert);
        outputShape_max_idxs.AddDim(n_feat);
        outputShape_max_idxs.AddDim(n_coords);

        Tensor *output_max_idxs_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape_max_idxs, &output_max_idxs_tensor));

        AccumulateKnnNdOpFunctor<Device, int>()(
                context->eigen_device<Device>(),
                d_coord_tensor.flat<float>().data(),
                d_feat_tensor.flat<float>().data(),
                d_idxs_tensor.flat<int>().data(),
                output_tensor->flat<float>().data(),
                output_max_idxs_tensor->flat<int>().data(),
                n_vert,
                n_neigh,
                n_coords,
                n_feat,
                n_out_feat,
                n_moments
        );



    }

private:
    int n_moments;
};

REGISTER_KERNEL_BUILDER(Name("AccumulateKnnNd").Device(DEVICE_CPU), AccumulateKnnNdOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnNdOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnnNd").Device(DEVICE_GPU), AccumulateKnnNdOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
