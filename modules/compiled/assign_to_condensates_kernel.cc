
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "assign_to_condensates_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

//helpers here

static float calc_distsq(
        const int idxa,
        const int idxb,
        const float * d_coords,
        const int n_coords
){
    float dist=0;
    for(int i_c=0;i_c<n_coords;i_c++){
        float d = d_coords[I2D(idxa,i_c,n_coords)]-d_coords[I2D(idxb,i_c,n_coords)];
        dist += d*d;
    }
    return dist;
}


static void assign_per_rs(
        const float *d_ccoords,
            const float *d_dist,
            const int *c_point_idx,

            const int *row_splits,

            int *asso_idx,

            const float radius,
            const int n_vert,
            const int n_coords,
            const int i_rs,
            const int n_condensates){

    const int i_v_start = row_splits[i_rs];
    const int i_v_end = row_splits[i_rs+1];

    for(int i_v=i_v_start; i_v<i_v_end;i_v++){
        int closest=-1;
        float closestdist=1e8;
        for(int i_c=0;i_c<n_condensates;i_c++){
            const int cidx = c_point_idx[i_c];//for gpu version do this once in advance
            if(cidx<0)
                continue;//protect against -1
            if(cidx<i_v_start || cidx>=i_v_end)
                continue; //only within row split

            float distscale = d_dist[cidx];
            float distsq = calc_distsq(i_v,cidx,d_ccoords,n_coords);
            distsq /= distscale*distscale;
            if(distsq<radius
                    && closestdist>distsq){
                closest = cidx;
                closestdist = distsq;
            }
        }
        asso_idx[i_v] = closest;
    }
}

// CPU specialization
template<typename dummy>
struct AssignToCondensatesOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_ccoords,
            const float *d_dist,
            const int *c_point_idx,

            const int *row_splits,

            int *asso_idx,

            const float radius,
            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_condensates
    ) {
        for(int i_rs=0;i_rs<n_rs-1;i_rs++){
            assign_per_rs(d_ccoords,d_dist,c_point_idx,row_splits,asso_idx,radius,n_vert,n_coords,i_rs,n_condensates);
        }
    }
};

template<typename Device>
class AssignToCondensatesOp : public OpKernel {
public:
    explicit AssignToCondensatesOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                context->GetAttr("radius", &radiussq_));
        radiussq_*=radiussq_;
    }


    void Compute(OpKernelContext *context) override {

        /*
         *
         *.Attr("radius: float")
          .Input("ccoords: float32")
          .Input("dist: float32")
          .Input("c_point_idx: int32")
          .Input("row_splits: int32")
          .Output("asso_idx: int32");
         */

        const Tensor &t_ccoords = context->input(0);
        const Tensor &t_dist = context->input(1);
        const Tensor &t_c_point_idx = context->input(2);
        const Tensor &t_row_splits = context->input(3);

        const int n_vert = t_ccoords.dim_size(0);
        const int n_ccoords = t_ccoords.dim_size(1);
        const int n_rs = t_row_splits.dim_size(0);
        const int n_cpoints = t_c_point_idx.dim_size(0);

        // size checks here

        OP_REQUIRES(context, n_vert == t_dist.dim_size(0),
                    errors::InvalidArgument("AssignToCondensatesOp expects first dimensions of coords and dist inputs to match."));

        TensorShape outputShape_idx;
        outputShape_idx.AddDim(n_vert);

        Tensor *t_asso_idx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_idx, &t_asso_idx));


        AssignToCondensatesOpFunctor<Device, int>()
                        (
                        context->eigen_device<Device>(),
                        //
                        t_ccoords.flat<float>().data(),
                        t_dist.flat<float>().data(),
                        t_c_point_idx.flat<int>().data(),

                        t_row_splits.flat<int>().data(),

                        t_asso_idx->flat<int>().data(),

                        radiussq_,
                        n_vert,
                        n_ccoords,
                        n_rs,
                        n_cpoints
                );


    }

private:
    float radiussq_;
};
//only CPU for now
REGISTER_KERNEL_BUILDER(Name("AssignToCondensates").Device(DEVICE_CPU), AssignToCondensatesOp<CPUDevice>);

//#ifdef GOOGLE_CUDA
//extern template struct BuildCondensatesOpFunctor<GPUDevice, int>;
//REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_GPU), BuildCondensatesOp<GPUDevice>);
//#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
