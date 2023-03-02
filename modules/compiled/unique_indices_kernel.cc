

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "unique_indices_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template<class T>
void coutVec(const std::vector<T> v){
    std::cout << "{ ";
    for(const auto& vv:v)
        std::cout << vv << ", ";
    std::cout << "}" <<std::endl;
}


class unique_wrapper {
public:
    unique_wrapper(
            const int check,
            const int idx):check_(check), idx_(idx){}

    bool operator == (const unique_wrapper& rhs){
        return check_ == rhs.check_;
    }
    bool operator > (const unique_wrapper& rhs){
        return check_ > rhs.check_;
    }
    bool operator < (const unique_wrapper& rhs){
        return check_ < rhs.check_;
    }

    int idx()const { return idx_;}
    int check()const {return check_;}
private:
    int check_;
    int idx_;
};

std::ostream &operator<<(std::ostream &os, unique_wrapper const &m) {
    return os << "(c: " << m.check() << ",i: " << m.idx() << ")";
}

static std::vector<unique_wrapper> calc(

        const int * input_labels,
        const int * rs,

        int * u_rs,

        const int n_x,
        const int n_rs

        ){

    std::vector<unique_wrapper> out;

    u_rs[0] = 0;

    for (int i_rs = 0; i_rs < n_rs - 1; i_rs++){
        if(rs[i_rs+1] > n_x) //error
            throw std::runtime_error("unique_indices_kernel: row splits wrong");

        int n_in_rs = rs[i_rs+1] - rs[i_rs];

        std::vector<unique_wrapper> wrapped;
        for(int i=0;i<n_in_rs;i++){
            const int il = i + rs[i_rs];
            wrapped.emplace_back(input_labels[il],il);
        }
        std::sort(wrapped.begin(),wrapped.end());

        auto last = std::unique(wrapped.begin(),wrapped.end());

        out.insert(out.end(),wrapped.begin(),last);
        u_rs[i_rs+1] = out.size();
    }
    return out;
}


template<typename Device>
class UniqueIndicesOp : public OpKernel {
public:
    explicit UniqueIndicesOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }


    void Compute(OpKernelContext *context) override {

        const Tensor &input_labels = context->input(0);
        const Tensor &row_splits = context->input(1);

        const int n_rs = row_splits.dim_size(0);
        const int n_x = input_labels.dim_size(0);

        Tensor *out_rs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,
                row_splits.shape(), &out_rs));//same shape

        
        auto uwraps = calc(
                input_labels.flat<int>().data(),
                row_splits.flat<int>().data(),

                out_rs->flat<int>().data(),

                n_x,
                n_rs);

        //fill to tensor


        Tensor *out_idxs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                {uwraps.size() ,1}, &out_idxs));//same shape

        int * p = out_idxs->flat<int>().data();
        for(int i = 0; i < uwraps.size() ; i++){
            p[i] = uwraps[i].idx();
        }

    }

};

REGISTER_KERNEL_BUILDER(Name("UniqueIndices").Device(DEVICE_CPU), UniqueIndicesOp<CPUDevice>);

}//functor
}//tensorflow
