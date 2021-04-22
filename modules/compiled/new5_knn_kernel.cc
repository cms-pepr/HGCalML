
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "new5_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template<typename dummy>
struct New5KnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coords_sorted, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,

            const int n_bins,
            const int* n_vtx_per_bin_cumulative, // size: n_bins_x*n_bins_y+1
            const int* bin_neighbours, // size: 9*n_bins_x*n_bins_y, bin itself + up to 8 neighbour bins
            const int* vtx_bin_assoc// size: V
            ) {

        // EMPTY

    }
};

template<typename Device>
class New5KnnOp : public OpKernel {
public:
    explicit New5KnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_neighbours", &K_));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_n_vtx_per_bin_cumulative = context->input(1);
        const Tensor &d_bin_neighbours = context->input(2);
        const Tensor &d_vtx_bin_assoc = context->input(3);

        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_bins = d_n_vtx_per_bin_cumulative.dim_size(0)-1;

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));

        New5KnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,
                n_bins,

                d_n_vtx_per_bin_cumulative.flat<int>().data(),
                d_bin_neighbours.flat<int>().data(),
                d_vtx_bin_assoc.flat<int>().data()
        );
    }

private:
    int K_;
};

REGISTER_KERNEL_BUILDER(Name("New5Knn").Device(DEVICE_CPU), New5KnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct New5KnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("New5Knn").Device(DEVICE_GPU), New5KnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
