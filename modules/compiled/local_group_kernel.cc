
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

            const int * d_row_splits,  //keeps dimensions: N_rs x 1

            int * mask,
            int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
            int * d_out_dir_neighbours, // V x K
            int * n_sel_vtx,
            int * d_out_row_splits,

            const int n_in_vert,
            const int n_neigh,
            const int n_row_splits,

            //globals for bookkeeping. dimension n_global_vert_g!
            int *d_out_backgather //which global index each vertex is associated to V x 1
    ){

        bool big_first = true;

        *n_sel_vtx=0;
        for(int i_v=0;i_v<n_in_vert; i_v++){
            mask[i_v]=0;
            for(int i_n=0;i_n<n_neigh;i_n++){
                if(i_n)
                    d_out_dir_neighbours[I2D(i_v, i_n, n_neigh)]=-1;
                else
                    d_out_dir_neighbours[I2D(i_v, i_n, n_neigh)]=i_v;//self ref
            }
        }
        int nseltotal=0;


        d_out_row_splits[0]=0;

        std::vector<bool> touched(n_in_vert,false);

        for(int i_rs=0; i_rs<n_row_splits-1;i_rs++){
            //row splits only relevant for future CPU parallelisation since they don't interact

            for(int iteration=0;iteration<2;iteration++){
                if(!big_first)
                    iteration++; //skip first iteration

                for(int _i_v=d_row_splits[i_rs];_i_v<d_row_splits[i_rs+1]; _i_v++){

                    int i_v = d_hierarchy_idxs[_i_v];
                    if(i_v >= n_in_vert){
                        printf("local_group_kernel: invalid i_v\n");
                        continue;
                    }

                    if(!iteration && touched.at(i_v))
                        continue;

                    //below threshold is not allowed to merge its neighbours
                    bool can_accumulate_neighbours = d_hierarchy_score[i_v] >= score_threshold;

                    //int v_gl_idx = d_global_idxs[i_v];
                    //d_out_backgather[v_gl_idx] = nseltotal; //global self-associate

                    //if not masked -> select and mask, increment nseltotal
                    //if selected add to backgather with nseltotal-1

                    if(!mask[i_v]){
                        d_out_selection_idxs[nseltotal] = i_v;
                        nseltotal++;
                    }
                    else{//already masked can never accumulate more neighbours
                        can_accumulate_neighbours=false;
                    }

                    int curr_neigh=0;
                    for(int i_n=0;i_n<n_neigh;i_n++){

                        if(i_n && !can_accumulate_neighbours)
                            break;

                        int nidx = d_neigh_idxs[I2D(i_v,i_n,n_neigh)];

                        if(nidx<0)//not a neighbour, -1s defined to be at the end so break here
                            break;

                        if(mask[nidx])//already used, self is already added to d_out_dir_neighbours
                            continue;
                        //all the neighbours are also going to be touched
                        if(!iteration)
                            for(int i_nn=0;i_nn<n_neigh;i_nn++){
                                int nnidx = d_neigh_idxs[I2D(nidx,i_nn,n_neigh)];
                                if(nnidx<0)
                                    continue;
                                touched.at(nnidx) = true;
                            }

                        d_out_dir_neighbours[I2D(i_v,curr_neigh,n_neigh)] = nidx;
                        d_out_backgather[nidx] = nseltotal-1;
                        mask[nidx] = 1;

                        curr_neigh++;
                    }
                }
            }
            d_out_row_splits[i_rs+1] = nseltotal;
        }

        *n_sel_vtx=nseltotal;

    }


};



template<typename dummy>
struct LocalGroupTruncateOpFunctor<CPUDevice,dummy> {
    void operator()(
            const CPUDevice &d,

            const int *d_in_selection_idxs, //which ones to keep
            int *d_out_selection_idxs,
            int n_new_vert
    ){
        for(int i_v=0;i_v<n_new_vert;i_v++){
            d_out_selection_idxs[i_v] = d_in_selection_idxs[i_v];
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
        const Tensor &t_row_splits = context->input(3);


        const int n_vert_in  = t_hierarchy_idxs.dim_size(0); //same as hierarch idxs, but not as global idxs
        const int n_neigh = t_neighbour_idxs.dim_size(1);
        const int n_rs = t_row_splits.dim_size(0);



        Tensor t_temp_mask;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,TensorShape({
            n_vert_in
        }),&t_temp_mask));


        Tensor *t_out_row_splits = NULL;
        Tensor *t_out_backgather_idxs_g = NULL;
        Tensor *t_out_dir_nidx = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            n_rs
        }), &t_out_row_splits));

        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({
            n_vert_in,n_neigh
        }), &t_out_dir_nidx));

        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({
            n_vert_in,1
        }), &t_out_backgather_idxs_g));


        Tensor t_temp_out_sel_idxs;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_INT32 ,TensorShape({
            n_vert_in, 1
        }),&t_temp_out_sel_idxs));



        int nseltotal=0;
        LocalGroupOpFunctor<Device, int>()(
                        context->eigen_device<Device>(),

                        t_neighbour_idxs.flat<int>().data(), // const int *d_neigh_idxs,
                        t_hierarchy_idxs.flat<int>().data(), //const int *d_hierarchy_idxs, //maybe do this internally if op can be called by op
                        //the above will require an argsort on ragged (only with eager workaround so far)

                        t_score.flat<float>().data(),

                        score_threshold_,

                        t_row_splits.flat<int>().data(), //const int * d_row_splits,  //keeps dimensions: N_rs x 1

                        t_temp_mask.flat<int>().data(), //int * mask,
                        t_temp_out_sel_idxs.flat<int>().data(), //int * d_out_selection_idxs,    //which ones to keep  - here V x 1, finally: V' x 1
                        t_out_dir_nidx->flat<int>().data(),
                        &nseltotal,
                        t_out_row_splits->flat<int>().data(),

                        n_vert_in,
                        n_neigh,
                        n_rs,


                        //globals for bookkeeping. dimension n_global_vert_g!
                        t_out_backgather_idxs_g->flat<int>().data()//which global index each vertex is associated to V x 1

                        );


        Tensor *t_selection_idxs = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({
            nseltotal, 1
        }), &t_selection_idxs));


        LocalGroupTruncateOpFunctor<Device,int>()(
                context->eigen_device<Device>(),

                t_temp_out_sel_idxs.flat<int>().data(), //which ones to keep
                t_selection_idxs->flat<int>().data(),
                nseltotal
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
