
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "local_group_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


// CPU specialization
template<typename dummy>
struct LocalGroupOpFunctor<CPUDevice, dummy> {
    void operator()(
            const CPUDevice &d,

            const int *d_neigh_idxs,
            const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op
            //the above will require an argsort on ragged (only with eager workaround so far)

            const float * d_hierarchy_score,

            const float score_threshold,

            const int * d_global_idxs, //global index of each vertex: V x 1, not global dimension!
            const int * d_row_splits,  //keeps dimensions: N_rs x 1

            int * mask,
            int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
            int * n_sel_vtx,
            int * d_out_row_splits,

            const int n_in_vert,
            const int n_neigh,
            const int n_row_splits,

            //globals for bookkeeping. dimension n_global_vert_g!
            int *d_out_backgather, //which global index each vertex is associated to V x 1
            int n_global_vert_g
    ){

        *n_sel_vtx=0;
        int nseltotal=0;
        for(int i_v=0;i_v<n_in_vert; i_v++)
            mask[i_v]=0;


        d_out_row_splits[0]=0;
        for(int i_rs=0; i_rs<n_row_splits-1;i_rs++){
            //row splits only relevant for parallelisation since they don't interact

            for(int _i_v=d_row_splits[i_rs];_i_v<d_row_splits[i_rs+1]; _i_v++){
                int i_v = d_hierarchy_idxs[_i_v];
                if(mask[i_v])
                    continue;
                bool below_threshold = d_hierarchy_score[i_v] < score_threshold;

                int v_gl_idx = d_global_idxs[i_v];
                d_out_backgather[v_gl_idx] = nseltotal; //global self-associate

                int curr_neigh=0;
                for(int i_n=0;i_n<n_neigh;i_n++){

                    int nidx = d_neigh_idxs[I2D(i_v,i_n,n_neigh)];

                    if(nidx<0)//not a neighbour
                        continue;
                    if(mask[nidx])
                        continue;//already used

                    d_out_selection_idxs[I2D(nseltotal,curr_neigh,n_neigh)] = nidx;
                    curr_neigh++;
                    //mask
                    mask[nidx]=1;
                    int ngl_idx = d_global_idxs[nidx];
                    d_out_backgather[ngl_idx] = nseltotal; //global associate

                    if(below_threshold)
                        break; //below threshold, only allow self-reference
                }
                nseltotal++;
            }

            d_out_row_splits[i_rs+1] = nseltotal;
        }

        *n_sel_vtx=nseltotal;

    }


};


static void set_defaults(
        int *d_out_selection_idxs,
        int n_vert,
        int n_neigh
        ){
    for(int i_v=0;i_v<n_vert;i_v++){
        for(int i_n=0;i_n<n_neigh;i_n++){
            d_out_selection_idxs[I2D(i_v,i_n,n_neigh)]=-1;
        }
    }
}


template<typename dummy>
struct LocalGroupTruncateOpFunctor<CPUDevice,dummy> {
    void operator()(
            const CPUDevice &d,

            const int *d_in_selection_idxs, //which ones to keep
            int *d_out_selection_idxs,
            int *d_out_ningroup,
            int n_new_vert,
            int n_neigh
    ){
        for(int i_v=0;i_v<n_new_vert;i_v++){
            int ingr=0;
            for(int i_n=0;i_n<n_neigh;i_n++){
                int idx = d_in_selection_idxs[I2D(i_v,i_n, n_neigh)];
                d_out_selection_idxs[I2D(i_v,i_n, n_neigh)] = idx;
                if(idx>=0)
                    ingr++;
            }
            d_out_ningroup[i_v] = ingr;
        }

    }
//needs a truncate functor, too? or do this with mallocs?
};

template<typename Device>
class LocalGroupOp : public OpKernel {
public:
    explicit LocalGroupOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                context->GetAttr("score_threshold", &score_threshold_));
    }

    void Compute(OpKernelContext *context) override {

        /*
         *
    .Input("neighbour_idxs: int32") // V x K
    .Input("hierarchy_idxs: int32") // V x 1  per group
    .Input("hierarchy_score: float32") // V x 1 score per group
    .Attr("score_threshold: float32")
    .Input("global_idxs: int32") // V x 1
    .Input("row_splits: int32") // RS
    .Output("out_row_splits: int32") //RS'
    .Output("selection_neighbour_idxs: int32") //V' x K'
    .Output("backscatter_idxs: int32"); //V

         */

        const Tensor &t_neighbour_idxs = context->input(0);
        const Tensor &t_hierarchy_idxs = context->input(1);
        const Tensor &t_score = context->input(2);
        const Tensor &t_global_idxs = context->input(3);
        const Tensor &t_row_splits = context->input(4);


        const int n_vert_in  = t_hierarchy_idxs.dim_size(0); //same as hierarch idxs, but not as global idxs
        const int n_neigh = t_neighbour_idxs.dim_size(1);
        const int n_global_idxs = t_global_idxs.dim_size(0);//just to throw exceptions
        const int n_rs = t_row_splits.dim_size(0);


        Tensor t_temp_out_sel_idxs;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,TensorShape({
            n_vert_in, n_neigh
        }),&t_temp_out_sel_idxs));

        set_defaults(t_temp_out_sel_idxs.flat<int>().data(),n_vert_in,n_neigh);

        Tensor t_temp_mask;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,TensorShape({
            n_vert_in
        }),&t_temp_mask));


        Tensor *t_out_row_splits = NULL;
        Tensor *t_out_backgather_idxs_g = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            n_rs
        }), &t_out_row_splits));

        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({
            n_global_idxs,1
        }), &t_out_backgather_idxs_g));


        int nseltotal=0;
        LocalGroupOpFunctor<Device, int>()(
                        context->eigen_device<Device>(),

                        t_neighbour_idxs.flat<int>().data(), // const int *d_neigh_idxs,
                        t_hierarchy_idxs.flat<int>().data(), //const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op
                        //the above will require an argsort on ragged (only with eager workaround so far)

                        t_score.flat<float>().data(),

                        score_threshold_,

                        t_global_idxs.flat<int>().data(), // const int * d_global_idxs, //global index of each vertex: V x 1, not global dimension!
                        t_row_splits.flat<int>().data(), //const int * d_row_splits,  //keeps dimensions: N_rs x 1

                        t_temp_mask.flat<int>().data(), //int * mask,
                        t_temp_out_sel_idxs.flat<int>().data(), //int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
                        &nseltotal,
                        t_out_row_splits->flat<int>().data(),

                        n_vert_in,
                        n_neigh,
                        n_rs,


                        //globals for bookkeeping. dimension n_global_vert_g!
                        t_out_backgather_idxs_g->flat<int>().data(), //which global index each vertex is associated to V x 1
                        n_global_idxs

                        );


        Tensor *t_selection_idxs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({
            nseltotal, n_neigh, 1
        }), &t_selection_idxs));


        Tensor *t_out_ningroup = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({
            nseltotal, 1
        }), &t_out_ningroup));

        LocalGroupTruncateOpFunctor<Device,int>()(
                context->eigen_device<Device>(),

                t_temp_out_sel_idxs.flat<int>().data(), //which ones to keep
                t_selection_idxs->flat<int>().data(),
                t_out_ningroup->flat<int>().data(),
                nseltotal,
                n_neigh
                );




    }

private:
    float score_threshold_;

};

REGISTER_KERNEL_BUILDER(Name("LocalGroup").Device(DEVICE_CPU), LocalGroupOp<CPUDevice>);

#ifdef NOOOOO_GOOGLE_CUDA
extern template struct LocalGroupOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("LocalGroup").Device(DEVICE_GPU), LocalGroupOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
