
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "compare_knn_outputs_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream>
#include <set>
#include <algorithm>
#include <iterator>

// #include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

void printSet(std::set<int>& inList, string prefixLine=""){
    std::cout << prefixLine << std::endl;
    for (auto const& i: inList){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void printVector(std::vector<int>& inList, string prefixLine=""){
    std::cout << prefixLine << std::endl;
    for (auto const& i: inList){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int getNumberOfUnmatchedNeigbours(const int* list1, const int* list2, int vertex_index, int n_neighbour){
    int i = vertex_index;
    int low_index = i*n_neighbour;
    int high_index = (i+1)*n_neighbour;
    std::set<int> s1(list1 + low_index,list1 + high_index);
    std::set<int> s2(list2 + low_index,list2 + high_index);
    std::vector<int> v_intersection;

    std::set_intersection(s1.begin(), s1.end(),
                          s2.begin(), s2.end(),
                          std::back_inserter(v_intersection));

    // std::cout << "low_index: " << low_index << "; high_index: " << high_index << std::endl;
    // printSet(s1,"s1");
    // printSet(s2,"s2");
    // printVector(v_intersection,"v_intersection:");

    return (n_neighbour-v_intersection.size());
}

// CPU specialization
template<typename dummy>
struct CompareKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,
                    size_t nvertices,
                    size_t nneighbours,
                    const int *input1, 
                    const int *input2, 
                    int *output){
        for (int i = 0; i < nvertices; ++i) {
            output[i] = getNumberOfUnmatchedNeigbours(input1, input2, i, nneighbours);
        }
    }
};

// implementation of the OpKernel
// contains mandatory override of Compute() function
template<typename Device>
class CompareKnnOp : public OpKernel {

// private:
//     int scaleFactor_;

public:
    // constructor
    explicit CompareKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        // OP_REQUIRES_OK(context,
        //                 context->GetAttr("scale_factor", &scaleFactor_));
    }

    // mandatory override
    // usually this is a common part for GPU and CPU devices
    // grabs and transform input tensors and define outputs
    void Compute(OpKernelContext *context) override {

        // grab input
        // V x K (indices: vertices x neighbours) 
        const Tensor &input_tensor1 = context->input(0);
        const Tensor &input_tensor2 = context->input(1);

        // Create an output tensor
        // V x 1 (indices: vertices x number of unmatched indices) 
        Tensor* output_tensor = NULL;
        TensorShape outputShape;
        outputShape.AddDim(input_tensor1.shape().dim_size(0));
        outputShape.AddDim(1);
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                         &output_tensor));

        // Do the computation
        CompareKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                // inputs/outputs
                static_cast<size_t>(input_tensor1.shape().dim_size(0)),
                static_cast<size_t>(input_tensor1.shape().dim_size(1)),
                input_tensor1.flat<int>().data(), // transform 2D tensor into 1D
                input_tensor2.flat<int>().data(), // transform 2D tensor into 1D
                output_tensor->flat<int>().data() // transform 2D tensor into 1D
        );
    }

};

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("CompareKnnOutputs").Device(DEVICE_CPU), CompareKnnOp<CPUDevice>);

// TODO
// Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// extern template struct CompareKnnOpFunctor<GPUDevice, int>;
// REGISTER_KERNEL_BUILDER(Name("CompareKnnOutputs").Device(DEVICE_GPU), CompareKnnOp<GPUDevice>);
// #endif  // GOOGLE_CUDA

}//functor
}//tensorflow
