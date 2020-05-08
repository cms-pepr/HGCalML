
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_kernel.h"
#include <queue>


namespace tensorflow {
    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;


    namespace functor {

        // Redefinition
        struct combined {
            int index;
            float distance;
        };
        class combinedcomparator {
        public:
            int operator() (const combined& p1, const combined& p2)
            {
                return p1.distance < p2.distance;
            }
        };

        // CPU specialization null
        template<typename dummy>
        struct AccumulateKnnOpFunctor<CPUDevice, dummy> {
            void operator()(const CPUDevice &d,

                    const float *d_coord,
                    const float *d_feat,
                    const int *d_idxs,

                    float *d_out_feat,

                    int n_vert,
                    int n_neigh,
                    int n_coords,
                    int n_feat,

                    int n_out_feat,

                    int n_moments) {

                //CPU implementation

            }
        };

        template<typename Device>
        class AccumulateKnnOp : public OpKernel {
        public:
            explicit AccumulateKnnOp(OpKernelConstruction *context) : OpKernel(context) {
                OP_REQUIRES_OK(context,
                               context->GetAttr("n_moments", &n_moments));
            }

            void Compute(OpKernelContext *context) override {

                const Tensor &d_coord_tensor = context->input(0);
                const Tensor &d_feat_tensor = context->input(1);
                const Tensor &d_idxs_tensor = context->input(2);


                int n_vert = d_coord_tensor.dim_size(0);
                int n_neigh = d_idxs_tensor.dim_size(1);// CHECK!
                int n_coords = d_coord_tensor.dim_size(1);
                int n_feat = d_feat_tensor.dim_size(1);


                int n_out_feat = 2 * n_feat; //mean and max

                // after testing basic functionality!
                // n_out_feat += n_moments * n_feat * n_coords;


                TensorShape outputShape;
                outputShape.AddDim(n_vert);
                outputShape.AddDim(n_out_feat);

                Tensor *output_tensor = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

                AccumulateKnnOpFunctor<Device, int>()(
                        context->eigen_device<Device>(),
                        d_coord_tensor.flat<float>().data(),
                        d_feat_tensor.flat<float>().data(),
                        d_idxs_tensor.flat<int>().data(),
                        output_tensor->flat<float>().data(),
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

REGISTER_KERNEL_BUILDER(Name("AccumulateKnn").Device(DEVICE_CPU), AccumulateKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnn").Device(DEVICE_GPU), AccumulateKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

    }
}
