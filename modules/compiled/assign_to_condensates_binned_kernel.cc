
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "assign_to_condensates_binned_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>
#include "binstepper.h"
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
    void assign(
            int shower_idx,
            int alpha_idx_h,
            const int n_vert_h,
            const int dimensions,
            float*max_search_dist_binning_h,
            int*condensates_assigned_h,
            int*condensates_dominant_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            const int* bins_flat_h,
            const int* bin_splits_h,
            const int* n_bins_h,
            const float* bin_widths_h,
            float* high_assigned_status_h,
            const int n_vert,
            const float* ccoords,
            const float* dist,
            const float* beta,
            int*assignment,
            const int* bins_flat,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int *indices_to_filtered
            ) {
        float bin_width = bin_widths[0];
        float bin_width_h = bin_widths_h[0];

        float radius = dist_h[alpha_idx_h];
        float min_[3];
        float max_[3];
        float radius_sq = radius*radius;

        std::cout<<"R "<<radius_sq<<std::endl;

        float my_ccoords[3];

        for(int id=0;id<dimensions;id++) {
            my_ccoords[id] = ccoords_h[alpha_idx_h*3+id];
            min_[id] = my_ccoords[id] - radius*4;
            max_[id] = my_ccoords[id] + radius*4;

        }
        high_assigned_status_h[alpha_idx_h]=1;

        binstepper_2 stepper(3);
        stepper.set(min_, max_, bin_width, n_bins);


        int num_assigned = 0;
        while(true) {
            int flat_bin_index = stepper.step();
            if(flat_bin_index==-1)
                break;

            for(int iv=bin_splits[flat_bin_index];iv<bin_splits[flat_bin_index+1];iv++) {
//            for(int iv=0;iv<n_vert;iv++) {
//                std::cout<<" "<<iv<<" "<<n_vert<<std::endl;
//                std::cout<<"Check B "<<flat_bin_index<<" "<<bin_splits[flat_bin_index]<<" "<<bin_splits[flat_bin_index+1]<<std::endl;
                if(assignment[iv]==-1) {
                    float dist = 0;
                    for(int id=0;id<dimensions;id++)
                        dist += (my_ccoords[id] - ccoords[iv*3+id]) * (my_ccoords[id] - ccoords[iv*3+id]);
                    if(dist < radius_sq) {
                        assignment[iv] = shower_idx;
                        num_assigned += 1;
                        if(indices_to_filtered[iv]!=-1) {
//                            std::cout<<"X "<<beta_h[alpha_idx_h]<<" "<<beta_h[indices_to_filtered[iv]]<<" "<<beta[iv]<<" "<<beta_h[alpha_idx_h]-beta_h[indices_to_filtered[iv]]<<" "<<indices_to_filtered[iv]<<" "<<iv<<std::endl;
                            high_assigned_status_h[indices_to_filtered[iv]]=1;
//                            std::cout<<"Check"<<std::endl;
                        }
                    }
                }
            }

//                break;
        }
        std::cout<<"This guy assigned "<<num_assigned<<std::endl;

    }


template<typename dummy>
struct BinnedCondensatesFinderOpFunctor<CPUDevice, dummy> {
    void operator()(
            const CPUDevice &d,
            const int n_vert_h,
            const int dimensions,
            float*max_search_dist_binning_h,
            int*condensates_assigned_h,
            int*condensates_dominant_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            const int* bins_flat_h,
            const int* bin_splits_h,
            const int* n_bins_h,
            const float* bin_widths_h,
            float* high_assigned_status_h,
            const int n_vert,
            const float* ccoords,
            const float* dist,
            const float* beta,
            const int* indices_to_filtered,
            int* assigned,
            const int* bins_flat,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths
            ) {
        /// CPU version here
        std::cout<<"Running the CPU version..."<<n_vert<<" "<<n_vert_h<<std::endl;

        for(int i=0;i<n_vert_h;i++) {
            high_assigned_status_h[i] = 0;
        }
        for(int i=0;i<n_vert;i++) {
            assigned[i] = -1;
        }
        int shower_idx=0;
        std::cout<<"Hello1"<<std::endl;
        for (int i=0;i<500;i++) {
            std::cout<<"Hello2"<<std::endl;
            float biggest_beta=-1;
            int biggest_beta_idx=-1;
            // Find the biggest beta vertex
            for(int iv_h=0;iv_h<n_vert_h;iv_h++) {
                if (high_assigned_status_h[iv_h]==0) {
                    if(beta_h[iv_h]>biggest_beta) {
                        biggest_beta = beta_h[iv_h];
                        biggest_beta_idx=iv_h;
                    }
                }
            }

            std::cout<<"Biggest beta "<<biggest_beta<< " "<<biggest_beta_idx<<" "<<shower_idx<<std::endl;
            if (biggest_beta==-1)
                break;

            assign(shower_idx,
                    biggest_beta_idx,
                    n_vert_h,
                    dimensions,
                    max_search_dist_binning_h,
                    condensates_assigned_h,
                    condensates_dominant_h,
                    ccoords_h,
                    dist_h,
                    beta_h,
                    bins_flat_h,
                    bin_splits_h,
                    n_bins_h,
                    bin_widths_h,
                    high_assigned_status_h,
                    n_vert,
                    ccoords,
                    dist,
                    beta,
                    assigned,
                    bins_flat,
                    bin_splits,
                    n_bins,
                    bin_widths,
                    indices_to_filtered);
            shower_idx += 1;
//        break;
        }
    }
};

//helpers here

template<typename Device>
class AssignToCondensatesBinnedOp : public OpKernel {
public:
    explicit AssignToCondensatesBinnedOp(OpKernelConstruction *context) : OpKernel(context) {
        std::cout<<"AssignToCondensatesBinnedOp context thing is being called...\n";
        OP_REQUIRES_OK(context, context->GetAttr("beta_threshold", &beta_threshold));
    }


    void Compute(OpKernelContext *context) override {
        std::cout<<"The main Compute function is getting called...\n";
        const Tensor &t_ccoords = context->input(0);
        const Tensor &t_dist = context->input(1);
        const Tensor &t_beta = context->input(2);
        const Tensor &t_bins_flat = context->input(3);
        const Tensor &t_bin_splits = context->input(4);
        const Tensor &t_n_bins = context->input(5);
        const Tensor &t_bin_widths = context->input(6);
        const Tensor &t_indices_to_filtered = context->input(7);
        const Tensor &t_ccoords_h = context->input(8);
        const Tensor &t_dist_h = context->input(9);
        const Tensor &t_beta_h = context->input(10);
        const Tensor &t_bins_flat_h = context->input(11);
        const Tensor &t_bin_splits_h = context->input(12);
        const Tensor &t_n_bins_h = context->input(13);
        const Tensor &t_bin_widths_h = context->input(14);

        int length = t_ccoords.dim_size(0);
        int length_h = t_beta_h.dim_size(0);
        int dimensions = t_n_bins_h.dim_size(0);
        std::cout<<"t_ccoords first dim size "<<t_ccoords.dim_size(0)<<std::endl;
        std::cout<<"t_dist first dim size "<<t_dist.dim_size(0)<<std::endl;
        std::cout<<"t_beta first dim size "<<t_beta.dim_size(0)<<std::endl;
        std::cout<<"t_bins_flat first dim size "<<t_bins_flat.dim_size(0)<<std::endl;
        std::cout<<"t_bin_splits first dim size "<<t_bin_splits.dim_size(0)<<std::endl;


//        std::cout<<"n_bins "<<n_bins.flat<int>().data()[0]<<std::endl;
//        std::cout<<"bin_widths "<<bin_widths.flat<float>().data()<<std::endl;

        std::cout<<"t_ccoords_h first dim size "<<t_ccoords_h.dim_size(0)<<std::endl;
        std::cout<<"t_dist_h first dim size "<<t_dist_h.dim_size(0)<<std::endl;
        std::cout<<"t_beta_h first dim size "<<t_beta_h.dim_size(0)<<std::endl;
        std::cout<<"t_bins_flat_h first dim size "<<t_bins_flat_h.dim_size(0)<<std::endl;
        std::cout<<"t_bin_splits_h first dim size "<<t_bin_splits_h.dim_size(0)<<std::endl;
//        std::cout<<"n_bins_h "<<n_bins_h<<std::endl;
//        std::cout<<"bin_widths_h "<<bin_widths_h<<std::endl;


//        BinnedCondensatesFinderOpFunctor<Device, int>() (
//                context->eigen_device<Device>()
////                t_ccoords_h.flat<float>().data(),
////                t_dist_h.flat<float>().data(),
////                t_beta_h.flat<float>().data(),
////                t_bins_flat_h.flat<int>().data(),
////                t_bin_splits_h.flat<int>().data(),
////                t_n_bins_h.flat<int>().data(),
////                t_bin_widths_h.flat<float>().data()
//        );

        Tensor *t_high_assigned_status_h = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,{length_h},&t_high_assigned_status_h));

        Tensor *t_assigned = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,{length},&t_assigned));

        Tensor t_max_search_dist_binning_h;
        Tensor t_condensates_assigned_h;
        Tensor t_condensates_dominant_h;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {640}, &t_max_search_dist_binning_h));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {length_h}, &t_condensates_assigned_h));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {length_h}, &t_condensates_dominant_h));

        std::cout<<"Alloc success"<<std::endl;
        BinnedCondensatesFinderOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                length_h,
                dimensions,
                t_max_search_dist_binning_h.flat<float>().data(),
                t_condensates_assigned_h.flat<int>().data(),
                t_condensates_dominant_h.flat<int>().data(),
                t_ccoords_h.flat<float>().data(),
                t_dist_h.flat<float>().data(),
                t_beta_h.flat<float>().data(),
                t_bins_flat_h.flat<int>().data(),
                t_bin_splits_h.flat<int>().data(),
                t_n_bins_h.flat<int>().data(),
                t_bin_widths_h.flat<float>().data(),
                t_high_assigned_status_h->flat<float>().data(),
                length,
                t_ccoords.flat<float>().data(),
                t_dist.flat<float>().data(),
                t_beta.flat<float>().data(),
                t_indices_to_filtered.flat<int>().data(),
                t_assigned->flat<int>().data(),
                t_bins_flat.flat<int>().data(),
                t_bin_splits.flat<int>().data(),
                t_n_bins.flat<int>().data(),
                t_bin_widths.flat<float>().data()
        );

    }

private:
    float beta_threshold;
};
//only CPU for now
//REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_CPU), AssignToCondensatesBinnedOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AssignToCondensatesBinnedOpFunctor<GPUDevice, int>;
extern template struct BinnedCondensatesFinderOpFunctor<GPUDevice, int>;

REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_GPU), AssignToCondensatesBinnedOp<GPUDevice>);
#endif

REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_CPU), AssignToCondensatesBinnedOp<CPUDevice>);



//#ifdef GOOGLE_CUDA
//extern template struct BuildCondensatesOpFunctor<GPUDevice, int>;
//REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_GPU), BuildCondensatesOp<GPUDevice>);
//#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
