
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "slicing_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>
#include <vector>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template<typename dummy>
struct SlicingKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coords, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            std::vector<float> phase_space_bin_boundary,
            const int n_rs,

            std::vector<int> n_bins,
            std::vector<int> features_to_bin_on
            ) {

        printf("\n*** Slicing CPU KERNEL IS NOT IMPLEMENTED ***\n");
        // EMPTY

    }
};

template<typename Device>
class SlicingKnnOp : public OpKernel {

private:
    int K_;
    std::vector<int> n_bins;
    std::vector<int> features_to_bin_on;
    std::vector<float> phase_space_bin_boundary;

public:
    explicit SlicingKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_bins", &n_bins));
        OP_REQUIRES_OK(context,
                        context->GetAttr("features_to_bin_on", &features_to_bin_on));
        OP_REQUIRES_OK(context,
                        context->GetAttr("phase_space_bin_boundary", &phase_space_bin_boundary));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_rs_tensor = context->input(1);

        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_rs = d_rs_tensor.dim_size(0);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));

        SlicingKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                d_rs_tensor.flat<int>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,
                phase_space_bin_boundary,
                n_rs,

                n_bins,
                features_to_bin_on
        );
    }

};

REGISTER_KERNEL_BUILDER(Name("SlicingKnn").Device(DEVICE_CPU), SlicingKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SlicingKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SlicingKnn").Device(DEVICE_GPU), SlicingKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
