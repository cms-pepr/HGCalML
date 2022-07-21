
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
            const int n_vert,
            const int n_vert_h,
            int shower_idx,
            int alpha_idx_h,
            const int dimensions,
            int*condensates_assigned_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            int* high_assigned_status_h,
            const float* ccoords,
            const float* dist,
            const float* beta,
            int*assignment,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int *indices_to_filtered,
            const int row_offset,
            const int row_offset_h, int r
            ) {
        float bin_width = bin_widths[0];

        float radius = dist_h[row_offset_h+alpha_idx_h];
        float *min_ = new float[dimensions];
        float *max_ = new float[dimensions];
        float radius_sq = radius*radius;

        float *my_ccoords = new float[dimensions];

        for(int id=0;id<dimensions;id++) {
            my_ccoords[id] = ccoords_h[row_offset_h*dimensions + alpha_idx_h*dimensions+id];
            min_[id] = my_ccoords[id] - radius;
            max_[id] = my_ccoords[id] + radius;

        }
        high_assigned_status_h[row_offset_h+alpha_idx_h]=1;
        condensates_assigned_h[row_offset_h+alpha_idx_h]=1;

        binstepper_2 stepper(dimensions);
        stepper.set(min_, max_, bin_width, n_bins);

        int num_assigned = 0;
        int min_flat_bin_index = 100000000;
        int max_flat_bin_index = 0;
        while(true) {
            int flat_bin_index = stepper.step();

            if(flat_bin_index==-1)
                break;
            min_flat_bin_index = flat_bin_index<min_flat_bin_index?flat_bin_index:min_flat_bin_index;
            max_flat_bin_index = flat_bin_index>max_flat_bin_index?flat_bin_index:max_flat_bin_index;


            for(int iv=bin_splits[flat_bin_index];iv<bin_splits[flat_bin_index+1];iv++) {
                if(assignment[iv]==-1) {
                    float dist = 0;
                    for(int id=0;id<dimensions;id++)
                        dist += (my_ccoords[id] - ccoords[iv*dimensions+id]) * (my_ccoords[id] - ccoords[iv*dimensions+id]);
                    if(dist < radius_sq) {
                        assignment[iv] = shower_idx;
                        num_assigned += 1;
                        if(indices_to_filtered[iv]!=-1) {
                            condensates_assigned_h[indices_to_filtered[iv]]=1;
                        }
                    }
                }
            }
        }
    }



template<typename dummy>
struct BinnedCondensatesFinderOpFunctor<CPUDevice, dummy> {
    void operator()(
            const CPUDevice &d,
            const int dimensions,
            float*max_search_dist_binning_h,
            int*condensates_assigned_h,
            int*condensates_dominant_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            const int* bin_splits_h,
            const int* n_bins_h,
            const float* bin_widths_h,
            int* high_assigned_status_h,
            const int*row_splits_h,
            const float* ccoords,
            const float* dist,
            const float* beta,
            const int* indices_to_filtered,
            int* assignment,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int*row_splits,
            const int num_rows
            ) {
        int total_bins = 1;
            for(int id=0;id<dimensions;id++) {
                total_bins*=n_bins[id];
        }

        for (int r = 0; r<num_rows;r++) {
            int n_vert = row_splits[r+1]-row_splits[r];
            int n_vert_h = row_splits_h[r+1]-row_splits_h[r];

            int row_offset=row_splits[r];
            int row_offset_h=row_splits_h[r];


            for(int i=0;i<n_vert_h;i++) {
                high_assigned_status_h[row_offset_h+i] = 0;
                condensates_assigned_h[row_offset_h+i] = 0;
            }
            for(int i=0;i<n_vert;i++) {
                assignment[row_offset+i] = -1;
            }
            int shower_idx=0;
            int last_idx=-1;


            while(true) {
                float biggest_beta=-1;
                int alpha_idx_h=-1;
                // Find the biggest beta vertex
                for(int iv_h=0;iv_h<n_vert_h;iv_h++) {
                    if (condensates_assigned_h[row_offset_h+iv_h]==0) {
                        if(beta_h[row_offset_h+iv_h]>biggest_beta) {
                            biggest_beta = beta_h[row_offset_h+iv_h];
                            alpha_idx_h=iv_h;
                        }
                    }
                }
                if (biggest_beta==-1)
                    break;
                if (biggest_beta==last_idx) {
                    throw std::runtime_error("Error progressing to next highest beta, something might be wrong in binning or in indices_to_filtered");
                }
                last_idx = alpha_idx_h;
                assign(n_vert, n_vert_h,shower_idx,
                        alpha_idx_h,
                        dimensions,
                        condensates_assigned_h,
                        ccoords_h,
                        dist_h,
                        beta_h,
                        high_assigned_status_h,
                        ccoords,
                        dist,
                        beta,
                        assignment,
                        bin_splits+(total_bins)*r,
                        n_bins,
                        bin_widths,
                        indices_to_filtered, row_offset, row_offset_h, r);
                shower_idx += 1;
    //        break;
            }
        }
    }
};

//helpers here

template<typename Device>
class AssignToCondensatesBinnedOp : public OpKernel {
public:
    explicit AssignToCondensatesBinnedOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("beta_threshold", &beta_threshold));
    }


    void Compute(OpKernelContext *context) override {
        const Tensor & t_ccoords = context->input(0);
        const Tensor & t_dist = context->input(1);
        const Tensor & t_beta = context->input(2);
        const Tensor & t_bins_flat = context->input(3);
        const Tensor & t_bin_splits = context->input(4);
        const Tensor & t_indices_to_filtered = context->input(5);
        const Tensor & t_row_splits = context->input(6);
        const Tensor & t_n_bins = context->input(7);
        const Tensor & t_bin_widths = context->input(8);
        const Tensor & t_ccoords_h = context->input(9);
        const Tensor & t_dist_h = context->input(10);
        const Tensor & t_beta_h = context->input(11);
        const Tensor & t_bins_flat_h = context->input(12);
        const Tensor & t_bin_splits_h = context->input(13);
        const Tensor & t_n_bins_h = context->input(14);
        const Tensor & t_bin_widths_h = context->input(15);
        const Tensor & t_row_splits_h = context->input(16);

        int length = t_ccoords.dim_size(0);
        int length_h = t_beta_h.dim_size(0);
        int dimensions = t_n_bins_h.dim_size(0);
        int num_rows = t_row_splits.dim_size(0)-1;

        Tensor *t_high_assigned_status_h = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,{length_h},&t_high_assigned_status_h));

        Tensor *t_assigned = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,{length},&t_assigned));

//        Tensor t_max_search_dist_binning_h;
        Tensor t_condensates_assigned_h;
//        Tensor t_condensates_dominant_h;
//        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {640}, &t_max_search_dist_binning_h));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {length_h}, &t_condensates_assigned_h));
//        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {length_h}, &t_condensates_dominant_h));

        /*
        (
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
            int* high_assigned_status_h,
            const int*row_splits_h,
            const int n_vert,
            const float* ccoords,
            const float* dist,
            const float* beta,
            const int* indices_to_filtered,
            int* assigned,
            const int* bins_flat,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int*row_splits,
            const int num_rows
            )
        */

        BinnedCondensatesFinderOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                dimensions,
                nullptr,//t_max_search_dist_binning_h.flat<float>().data(),
                t_condensates_assigned_h.flat<int>().data(),
                nullptr,//t_condensates_dominant_h.flat<int>().data(),
                t_ccoords_h.flat<float>().data(),
                t_dist_h.flat<float>().data(),
                t_beta_h.flat<float>().data(),
                t_bin_splits_h.flat<int>().data(),
                t_n_bins_h.flat<int>().data(),
                t_bin_widths_h.flat<float>().data(),
                t_high_assigned_status_h->flat<int>().data(),
                t_row_splits_h.flat<int>().data(),
                t_ccoords.flat<float>().data(),
                t_dist.flat<float>().data(),
                t_beta.flat<float>().data(),
                t_indices_to_filtered.flat<int>().data(),
                t_assigned->flat<int>().data(),
                t_bin_splits.flat<int>().data(),
                t_n_bins.flat<int>().data(),
                t_bin_widths.flat<float>().data(),
                t_row_splits.flat<int>().data(),
                num_rows
        );

    }

private:
    float beta_threshold;
};
//only CPU for now
//REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_CPU), AssignToCondensatesBinnedOp<CPUDevice>);

//#ifdef GOOGLE_CUDA
//extern template struct AssignToCondensatesBinnedOpFunctor<GPUDevice, int>;
//extern template struct BinnedCondensatesFinderOpFunctor<GPUDevice, int>;
//
//REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_GPU), AssignToCondensatesBinnedOp<GPUDevice>);
//#endif

REGISTER_KERNEL_BUILDER(Name("AssignToCondensatesBinned").Device(DEVICE_CPU), AssignToCondensatesBinnedOp<CPUDevice>);



//#ifdef GOOGLE_CUDA
//extern template struct BuildCondensatesOpFunctor<GPUDevice, int>;
//REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_GPU), BuildCondensatesOp<GPUDevice>);
//#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
