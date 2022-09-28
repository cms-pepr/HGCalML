
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "binned_assign_to_condensates_kernel.h"
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
            const int dimensions_binning,
            int*condensates_assigned_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            int* high_assigned_status_h,
            const int* original_indices_h,
            const float* ccoords,
            const float* dist,
            float*assignment_min_distance,
            int*assignment,
            int*association,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int *indices_to_filtered,
            const int row_offset,
            const int row_offset_h, int r, bool assign_by_max_beta
            ) {
        float bin_width = bin_widths[0];

        float radius = dist_h[row_offset_h+alpha_idx_h];
        float *min_ = new float[dimensions_binning];
        float *max_ = new float[dimensions_binning];
        float radius_sq = radius*radius;

        float *my_ccoords = new float[dimensions];

        for(int id=0;id<dimensions;id++) {
            my_ccoords[id] = ccoords_h[row_offset_h*dimensions + alpha_idx_h*dimensions+id];
        }

        for(int id=0;id<dimensions_binning;id++) {
            min_[id] = my_ccoords[id] - radius;
            max_[id] = my_ccoords[id] + radius;
        }

        high_assigned_status_h[row_offset_h+alpha_idx_h]=1;
        condensates_assigned_h[row_offset_h+alpha_idx_h]=1;

        ccoords2flat_binstepper stepper(dimensions_binning);
        stepper.set(min_, max_, bin_width, n_bins);

        while(true) {
            int flat_bin_index = stepper.step();
            if(flat_bin_index==-1)
                break;

            for(int iv=bin_splits[flat_bin_index];iv<bin_splits[flat_bin_index+1];iv++) {
                if (assign_by_max_beta) {
                    if(assignment[iv]==-1) {
                        float dist = 0;
                        for(int id=0;id<dimensions;id++)
                            dist += (my_ccoords[id] - ccoords[iv*dimensions+id]) * (my_ccoords[id] - ccoords[iv*dimensions+id]);
                        if(dist <= radius_sq) {
                            assignment[iv] = shower_idx;
                            association[iv] = original_indices_h[row_offset_h+alpha_idx_h];
                            if(indices_to_filtered[iv]!=-1) {
                                condensates_assigned_h[indices_to_filtered[iv]]=1;
                            }
                        }
                    }
                }
                else {

                    float dist = 0;
                    for(int id=0;id<dimensions;id++)
                        dist += (my_ccoords[id] - ccoords[iv*dimensions+id]) * (my_ccoords[id] - ccoords[iv*dimensions+id]);
                    if(dist <= radius_sq) {
                        float dist_by_radiussq = dist/radius_sq;
                        if (assignment_min_distance[iv] > dist_by_radiussq) {

                            assignment[iv] = shower_idx;
                            association[iv] = original_indices_h[row_offset_h+alpha_idx_h];
                            assignment_min_distance[iv] = dist_by_radiussq;
                            if(indices_to_filtered[iv]!=-1) {
                                condensates_assigned_h[indices_to_filtered[iv]]=1;
                            }
                        }
                    }
                }
            }
        }

        //clean up
        delete [] min_;
        delete [] max_;
        delete [] my_ccoords;
    }


template<typename dummy>
struct BinnedCondensatesFinderOpFunctor<CPUDevice, dummy> {
    void operator()(
            const CPUDevice &d,
            const int dimensions,
            const int dimensions_binning,
            int*condensates_assigned_h,
            const float* ccoords_h,
            const float* dist_h,
            const float* beta_h,
            int* high_assigned_status_h,
            const int* original_indices_h,
            const int*row_splits_h,
            const float* ccoords,
            const float* dist,
            const int* indices_to_filtered,
            float* assignment_min_distance,
            int* assignment,
            int* association,
            int* n_condensates,
            int* alpha_indices,
            const int* bin_splits,
            const int* n_bins,
            const float* bin_widths,
            const int*row_splits,
            const int num_rows,
            const bool assign_by_max_beta
            ) {
        int total_bins = 1;
        int shower_rs_offset=0;
        n_condensates[0] = 0;
        for(int id=0;id<dimensions_binning;id++) {
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
                alpha_indices[row_offset_h+i] = -1;
            }
            for(int i=0;i<n_vert;i++) {
                assignment[row_offset+i] = -1;
                association[row_offset+i] = -1;
                if (not assign_by_max_beta)
                    assignment_min_distance[row_offset+i] = 10e20;
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
                if (alpha_idx_h==last_idx) {
                    throw std::runtime_error("Error progressing to next highest beta, something might be wrong in binning or in indices_to_filtered");
                }
                last_idx = alpha_idx_h;

                assign(n_vert, n_vert_h,shower_idx,
                        alpha_idx_h,
                        dimensions,
                        dimensions_binning,
                        condensates_assigned_h,
                        ccoords_h,
                        dist_h,
                        beta_h,
                        high_assigned_status_h,
                        original_indices_h,
                        ccoords,
                        dist,
                        assignment_min_distance,
                        assignment,
                        association,
                        bin_splits+(total_bins)*r,
                        n_bins,
                        bin_widths,
                        indices_to_filtered, row_offset, row_offset_h, r, assign_by_max_beta);

                alpha_indices[row_offset_h+shower_idx] = original_indices_h[row_offset_h+alpha_idx_h];
                shower_idx += 1;
            }
            n_condensates[r+1] = shower_rs_offset+shower_idx;
            shower_rs_offset = n_condensates[r+1];
        }
    }
};

//helpers here

template<typename Device>
class BinnedAssignToCondensatesOp : public OpKernel {
public:
    explicit BinnedAssignToCondensatesOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("beta_threshold", &beta_threshold));
        OP_REQUIRES_OK(context, context->GetAttr("assign_by_max_beta", &assign_by_max_beta));
    }


    void Compute(OpKernelContext *context) override {
        const Tensor & t_ccoords = context->input(0);
        const Tensor & t_dist = context->input(1);
        const Tensor & t_beta = context->input(2);
        const Tensor & t_bins_flat = context->input(3);
        const Tensor & t_bin_splits = context->input(4);
        const Tensor & t_indices_to_filtered = context->input(5);
        const Tensor & t_original_indices = context->input(6);
        const Tensor & t_row_splits = context->input(7);
        const Tensor & t_n_bins = context->input(8);
        const Tensor & t_bin_widths = context->input(9);
        const Tensor & t_ccoords_h = context->input(10);
        const Tensor & t_dist_h = context->input(11);
        const Tensor & t_beta_h = context->input(12);
        const Tensor & t_row_splits_h = context->input(13);

        int length = t_ccoords.dim_size(0);
        int length_h = t_beta_h.dim_size(0);
        int dimensions = t_ccoords.dim_size(1);
        int dimensions_binning = t_n_bins.dim_size(0);
        int num_rows = t_row_splits.dim_size(0)-1;

        Tensor *t_high_assigned_status_h = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,{length_h},&t_high_assigned_status_h));

        Tensor *t_assigned = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,{length},&t_assigned));


        Tensor *t_alpha_indices = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2,{length_h},&t_alpha_indices));

        Tensor *t_association = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3,{length},&t_association));

        Tensor *t_n_condensates = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(4,{num_rows+1},&t_n_condensates));

        Tensor t_condensates_assigned_h;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, {length_h}, &t_condensates_assigned_h));


        Tensor t_max_distance;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {length}, &t_max_distance));

        BinnedCondensatesFinderOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                dimensions,
                dimensions_binning,
                t_condensates_assigned_h.flat<int>().data(),
                t_ccoords_h.flat<float>().data(),
                t_dist_h.flat<float>().data(),
                t_beta_h.flat<float>().data(),
                t_high_assigned_status_h->flat<int>().data(),
                t_original_indices.flat<int>().data(),
                t_row_splits_h.flat<int>().data(),
                t_ccoords.flat<float>().data(),
                t_dist.flat<float>().data(),
                t_indices_to_filtered.flat<int>().data(),
                assign_by_max_beta ? nullptr : t_max_distance.flat<float>().data(),
                t_assigned->flat<int>().data(),
                t_association->flat<int>().data(),
                t_n_condensates->flat<int>().data(),
                t_alpha_indices->flat<int>().data(),
                t_bin_splits.flat<int>().data(),
                t_n_bins.flat<int>().data(),
                t_bin_widths.flat<float>().data(),
                t_row_splits.flat<int>().data(),
                num_rows,
                assign_by_max_beta
        );

    }

private:
    float beta_threshold;
    bool assign_by_max_beta;
};


REGISTER_KERNEL_BUILDER(Name("BinnedAssignToCondensates").Device(DEVICE_CPU), BinnedAssignToCondensatesOp<CPUDevice>);


}//functor
}//tensorflow
