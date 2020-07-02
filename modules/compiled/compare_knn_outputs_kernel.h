
#ifndef COMPARE_KNN_OUTPUTS_KERNEL_H
#define COMPARE_KNN_OUTPUTS_KERNEL_H

namespace tensorflow {
namespace functor {

// Device - can be GPU or CPU
template<typename Device, typename dummy>
struct CompareKnnOpFunctor {
    void operator()(
            const Device &d, // GPU or CPU
            size_t nvertices,
            size_t nneighbours,
            const int *input1, 
            const int *input2, 
            int *output

            );
};
//
// // TODO why didn't original files have this specialization?
// #if GOOGLE_CUDA
// // Partially specialize functor for GpuDevice.
// template <typename Eigen::GpuDevice, typename dummy>
// struct DummyKnnOpFunctor {
//     void operator()(
//             const Eigen::GpuDevice& d,
//
//             size_t size,
//             const int *input1D,
//             int *output1D,
//
//             const int scaleFactor
//             );
// };
// #endif
//

}  // namespace functor
}  // namespace tensorflow

#endif //COMPARE_KNN_OUTPUTS_KERNEL_H

